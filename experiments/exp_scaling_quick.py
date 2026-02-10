#!/usr/bin/env python3
"""
Phase A + D Combined: Quick validation of improvements
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
    def __init__(self, size=400, num_kv=8, min_noise=3, max_noise=10):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, 30))
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

def train_eval(model, num_kv, epochs=15, lr=1e-3):
    ds = QuickMultiKV(size=400, num_kv=num_kv)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    for _ in range(epochs):
        model.train()
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = (crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size()) * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

print("="*70)
print("QUICK SCALING TEST: Medium (128d, 3 layers) ~400K params")
print("="*70)

results = {'medium': {'ablations': {}}}

for ablation_name, flags in CONFIGS.items():
    print(f"\n  Testing {ablation_name}...")
    accs = []
    for seed in [42, 123]:
        torch.manual_seed(seed)
        random.seed(seed)
        
        cfg = ANAConfig(d_model=128, num_layers=3, state_dim=128, vocab_size=30, **flags)
        
        if ablation_name == 'baseline':
            model = BaselineSSM(cfg).to(device)
        else:
            model = ANAModel(cfg).to(device)
        
        params = sum(p.numel() for p in model.parameters())
        acc = train_eval(model, num_kv=8, epochs=15)
        accs.append(acc)
    
    mean_acc = sum(accs) / len(accs)
    results['medium']['ablations'][ablation_name] = {'mean': mean_acc, 'params': params}
    print(f"    {ablation_name}: {mean_acc*100:.1f}% ({params:,} params)")

single_best = max(results['medium']['ablations'][c]['mean'] for c in ['controller', 'hololink'])
full_mean = results['medium']['ablations']['full']['mean']
synergy = full_mean - single_best
results['medium']['synergy'] = synergy
print(f"\n  Synergy: +{synergy*100:.1f}%")

print("\n" + "="*70)
print("COMPARISON: Original vs New run")
print("="*70)
print(f"Original medium: Full=97.3%, Best=96.6%, Synergy=+0.7%")
print(f"New medium: Full={full_mean*100:.1f}%, Best={single_best*100:.1f}%, Synergy={synergy*100:+.1f}%")

with open('archive/experiments/phaseA_scaling_quick.json', 'w') as f:
    json.dump(results, f, indent=2)
