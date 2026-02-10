#!/usr/bin/env python3
"""
Experiment: Synergy Analysis Across KV Counts
Find where synergy is strongest and why
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

class QuickMultiKV(data.Dataset):
    def __init__(self, size=500, num_kv=8, min_noise=3, max_noise=10):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, 50))
        for _ in range(size):
            kvs = [(random.choice(content), random.choice(content)) for _ in range(num_kv)]
            seq = []
            for k, v in kvs:
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            seq.extend([random.choice(content) for _ in range(random.randint(min_noise, max_noise))])
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

def train_eval(model, num_kv, epochs, lr):
    ds = QuickMultiKV(size=500, num_kv=num_kv)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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

CONFIGS = {
    'baseline': {'use_hololink': False, 'use_controller': False},
    'controller': {'use_hololink': False, 'use_controller': True},
    'hololink': {'use_hololink': True, 'use_controller': False},
    'full': {'use_hololink': True, 'use_controller': True},
}

KV_COUNTS = [1, 2, 4, 6, 8, 10, 12]
MODEL_CONFIG = {
    'd_model': 64,
    'num_layers': 2,
    'state_dim': 64,
    'vocab_size': 50,
    'max_seq_len': 2048,
    'lr': 1e-3,
    'epochs': 20
}

print("="*70)
print("SYNERGY ANALYSIS ACROSS KV COUNTS")
print("="*70)

results = {kv: {} for kv in KV_COUNTS}

for num_kv in KV_COUNTS:
    print(f"\n{'='*70}")
    print(f"Testing {num_kv} KV pairs")
    print(f"{'='*70}")
    
    results[num_kv] = {}
    
    for ablation_name, flags in CONFIGS.items():
        print(f"  {ablation_name}...", end=' ')
        accs = []
        for seed in [42, 123]:
            torch.manual_seed(seed)
            random.seed(seed)
            
            cfg = ANAConfig(
                d_model=MODEL_CONFIG['d_model'],
                num_layers=MODEL_CONFIG['num_layers'],
                state_dim=MODEL_CONFIG['state_dim'],
                vocab_size=MODEL_CONFIG['vocab_size'],
                max_seq_len=MODEL_CONFIG['max_seq_len'],
                **flags
            )
            
            if ablation_name == 'baseline':
                model = BaselineSSM(cfg).to(device)
            else:
                model = ANAModel(cfg).to(device)
            
            acc = train_eval(model, num_kv, MODEL_CONFIG['epochs'], MODEL_CONFIG['lr'])
            accs.append(acc)
        
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
        results[num_kv][ablation_name] = {'mean': mean_acc, 'std': std_acc}
        print(f"{mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    
    single_best = max(results[num_kv][c]['mean'] for c in ['controller', 'hololink'])
    full_mean = results[num_kv]['full']['mean']
    synergy = full_mean - single_best
    results[num_kv]['synergy'] = synergy
    print(f"  Synergy: +{synergy*100:.1f}%")

with open('archive/experiments/synergy_by_kv.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("SYNERGY BY KV COUNT SUMMARY")
print("="*70)
print(f"{'KV':>3} | {'Baseline':>10} | {'Controller':>10} | {'HoloLink':>10} | {'Full':>10} | {'Synergy':>10}")
print("-"*65)
for kv in KV_COUNTS:
    base = results[kv]['baseline']['mean']
    ctrl = results[kv]['controller']['mean']
    holo = results[kv]['hololink']['mean']
    full = results[kv]['full']['mean']
    syn = results[kv]['synergy']
    print(f"{kv:>3} | {base*100:>9.1f}% | {ctrl*100:>9.1f}% | {holo*100:>9.1f}% | {full*100:>9.1f}% | {syn*100:>+9.1f}%")

print(f"\nResults saved to: archive/experiments/synergy_by_kv.json")
