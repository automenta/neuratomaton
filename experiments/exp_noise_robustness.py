#!/usr/bin/env python3
"""
Experiment: Noise Robustness Analysis
Test ANA under varying difficulty levels (noise amount)
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import random

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM

class NoiseDataset(data.Dataset):
    def __init__(self, size=400, num_kv=8, min_noise=3, max_noise=50):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, 30))
        for _ in range(size):
            kvs = [(random.choice(content), random.choice(content)) for _ in range(num_kv)]
            seq = []
            for k, v in kvs:
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            noise_len = random.randint(min_noise, max_noise)
            seq.extend([random.choice(content) for _ in range(noise_len)])
            ti = random.randint(0, num_kv-1)
            seq.extend([TOK_QUERY, kvs[ti][0], kvs[ti][1]])
            x = torch.tensor(seq[:-1])
            y = torch.tensor(seq[1:])
            m = torch.ones_like(y, dtype=torch.float) * 0.01
            m[-1] = 1.0
            self.data.append((x, y, m))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    xs, ys, ms = zip(*batch)
    ml = max(x.size(0) for x in xs)
    return (torch.stack([F.pad(x, (0, ml-x.size(0))) for x in xs]),
            torch.stack([F.pad(y, (0, ml-y.size(0))) for y in ys]),
            torch.stack([F.pad(m, (0, ml-m.size(0))) for m in ms]))

def train_eval(model, min_noise, max_noise, num_kv=8, epochs=20):
    ds = NoiseDataset(size=500, num_kv=num_kv, min_noise=min_noise, max_noise=max_noise)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    for _ in range(epochs):
        model.train()
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = (crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size()) * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, _ = model(x)
            for i in range(x.size(0)):
                pos = (m[i] > 0.5).nonzero(as_tuple=True)[0][0]
                if logits[i, pos].argmax().item() == y[i, pos].item():
                    correct += 1
                total += 1
    return correct / total if total else 0

NOISE_LEVELS = [
    ('easy', (0, 3)),
    ('medium', (3, 10)),
    ('hard', (10, 25)),
    ('extreme', (25, 50)),
]

CONFIGS = {
    'baseline': {'use_hololink': False, 'use_controller': False},
    'controller': {'use_hololink': False, 'use_controller': True},
    'hololink': {'use_hololink': True, 'use_controller': False},
    'full': {'use_hololink': True, 'use_controller': True},
}

print("="*70)
print("NOISE ROBUSTNESS ANALYSIS")
print("="*70)

results = {level: {} for level, _ in NOISE_LEVELS}

for level_name, (min_n, max_n) in NOISE_LEVELS:
    print(f"\n{'='*70}")
    print(f"Noise level: {level_name} ({min_n}-{max_n} tokens)")
    print(f"{'='*70}")
    
    for cfg_name, flags in CONFIGS.items():
        print(f"  {cfg_name}...", end=' ')
        accs = []
        for seed in [42, 123]:
            torch.manual_seed(seed)
            random.seed(seed)
            
            cfg = ANAConfig(d_model=64, num_layers=2, state_dim=64, vocab_size=30, **flags)
            
            if cfg_name == 'baseline':
                model = BaselineSSM(cfg).to(device)
            else:
                model = ANAModel(cfg).to(device)
            
            acc = train_eval(model, min_n, max_n, epochs=25)
            accs.append(acc)
        
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
        results[level_name][cfg_name] = {'mean': mean_acc, 'std': std_acc}
        print(f"{mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    
    single_best = max(results[level_name][c]['mean'] for c in ['controller', 'hololink'])
    full_mean = results[level_name]['full']['mean']
    synergy = full_mean - single_best
    results[level_name]['synergy'] = synergy
    print(f"  Synergy: +{synergy*100:.1f}%")

with open('archive/experiments/noise_robustness.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("NOISE ROBUSTNESS SUMMARY")
print("="*70)
print(f"{'Noise':>10} | {'Base':>8} | {'Ctrl':>8} | {'Holo':>8} | {'Full':>8} | {'Syn':>6}")
print("-"*60)
for level, _ in NOISE_LEVELS:
    r = results[level]
    print(f"{level:>10} | {r['baseline']['mean']*100:>7.1f}% | {r['controller']['mean']*100:>7.1f}% | {r['hololink']['mean']*100:>7.1f}% | {r['full']['mean']*100:>7.1f}% | {r['synergy']*100:>+5.1f}%")

print(f"\nResults saved to: archive/experiments/noise_robustness.json")
