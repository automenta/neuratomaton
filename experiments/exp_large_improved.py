#!/usr/bin/env python3
"""
Phase A: Large Model with Better Training
- Lower learning rate
- More epochs
- Gradient clipping
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
    def __init__(self, size=600, num_kv=8, min_noise=3, max_noise=10):
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

def train_eval(model, num_kv, epochs=30, lr=5e-4, warmup=5):
    ds = QuickMultiKV(size=600, num_kv=num_kv)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr*2, epochs=epochs, steps_per_epoch=len(loader))
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    for epoch in range(epochs):
        model.train()
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = (crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size()) * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            sched.step()
    
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

print("="*70)
print("LARGE MODEL WITH BETTER TRAINING")
print("="*70)

CONFIGS = {
    'baseline': {'use_hololink': False, 'use_controller': False},
    'controller': {'use_hololink': False, 'use_controller': True},
    'hololink': {'use_hololink': True, 'use_controller': False},
    'full': {'use_hololink': True, 'use_controller': True},
}

results = {'ablations': {}}

for ablation_name, flags in CONFIGS.items():
    print(f"\n  Testing {ablation_name}...")
    accs = []
    for seed in [42, 123]:
        torch.manual_seed(seed)
        random.seed(seed)
        
        cfg = ANAConfig(
            d_model=256,
            num_layers=4,
            state_dim=256,
            vocab_size=30,
            max_seq_len=2048,
            **flags
        )
        
        if ablation_name == 'baseline':
            model = BaselineSSM(cfg).to(device)
        else:
            model = ANAModel(cfg).to(device)
        
        params = sum(p.numel() for p in model.parameters())
        acc = train_eval(model, num_kv=8, epochs=30, lr=5e-4)
        accs.append(acc)
    
    mean_acc = sum(accs) / len(accs)
    results['ablations'][ablation_name] = {'mean': mean_acc, 'params': params}
    print(f"    {ablation_name}: {mean_acc*100:.1f}% ({params:,} params)")

single_best = max(results['ablations'][c]['mean'] for c in ['controller', 'hololink'])
full_mean = results['ablations']['full']['mean']
synergy = full_mean - single_best
results['synergy'] = synergy
print(f"\n  Synergy: +{synergy*100:.1f}%")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Original large: Full=38.1%, Controller=47.1%, Synergy=-9.0%")
print(f"New large: Full={full_mean*100:.1f}%, Controller={results['ablations']['controller']['mean']*100:.1f}%, Synergy={synergy*100:+.1f}%")
