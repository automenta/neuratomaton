#!/usr/bin/env python3
"""
Experiment: Synergy Mechanism Analysis
Track gate activations and HoloLink memory usage
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
from ana.models import ANAModel

class MultiKVDataset(data.Dataset):
    def __init__(self, size=100, num_kv=8):
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, 30))
        self.data = []
        for _ in range(size):
            kvs = [(random.choice(content), random.choice(content)) for _ in range(num_kv)]
            seq = []
            for k, v in kvs:
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            seq.extend([random.choice(content) for _ in range(random.randint(3, 10))])
            ti = random.randint(0, num_kv-1)
            seq.extend([TOK_QUERY, kvs[ti][0], kvs[ti][1]])
            self.data.append((torch.tensor(seq[:-1]), torch.tensor(seq[1:])))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def analyze_model_behavior(model, num_kv):
    """Analyze gate activations and memory patterns"""
    ds = MultiKVDataset(size=50, num_kv=num_kv)
    
    model.eval()
    
    gate_alpha_values = []
    gate_beta_values = []
    ret_gate_values = []
    
    with torch.no_grad():
        for x, y in ds:
            x = x.unsqueeze(0).to(device)
            logits, info_log = model(x, return_info=True)
            
            for info in info_log:
                if 'ga_0' in info:
                    gate_alpha_values.append(info['ga_0'])
                if 'ret_gate' in info:
                    ret_gate_values.append(info['ret_gate'])
    
    return {
        'gate_alpha_mean': sum(gate_alpha_values) / len(gate_alpha_values) if gate_alpha_values else 0,
        'gate_alpha_std': (sum((g - sum(gate_alpha_values)/len(gate_alpha_values))**2 for g in gate_alpha_values) / len(gate_alpha_values))**0.5 if gate_alpha_values else 0,
        'ret_gate_mean': sum(ret_gate_values) / len(ret_gate_values) if ret_gate_values else 0,
        'ret_gate_std': (sum((g - sum(ret_gate_values)/len(ret_gate_values))**2 for g in ret_gate_values) / len(ret_gate_values))**0.5 if ret_gate_values else 0,
    }

print("="*70)
print("SYNERGY MECHANISM ANALYSIS")
print("="*70)

results = {}

for num_kv in [4, 8, 12]:
    print(f"\n{'='*70}")
    print(f"{num_kv} KV pairs analysis")
    print(f"{'='*70}")
    
    results[num_kv] = {}
    
    configs = {
        'controller': {'use_hololink': False, 'use_controller': True},
        'hololink': {'use_hololink': True, 'use_controller': False},
        'full': {'use_hololink': True, 'use_controller': True},
    }
    
    for name, flags in configs.items():
        print(f"\n  {name}:")
        
        torch.manual_seed(42)
        random.seed(42)
        
        model = ANAModel(ANAConfig(
            d_model=64,
            num_layers=2,
            state_dim=64,
            vocab_size=30,
            max_seq_len=1024,
            **flags
        )).to(device)
        
        # Simple training
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ds = MultiKVDataset(size=200, num_kv=num_kv)
        for epoch in range(15):
            for x, y in ds:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits, _ = model(x.unsqueeze(0))
                loss = F.cross_entropy(logits.view(-1, 30), y.view(-1), ignore_index=0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        
        # Analyze behavior
        behavior = analyze_model_behavior(model, num_kv)
        results[num_kv][name] = behavior
        
        if name in ['controller', 'full']:
            print(f"    Gate Alpha: {behavior['gate_alpha_mean']:.3f} ± {behavior['gate_alpha_std']:.3f}")
            print(f"    Retention Gate: {behavior['ret_gate_mean']:.3f} ± {behavior['ret_gate_std']:.3f}")
        
        params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {params:,}")

with open('archive/experiments/synergy_mechanism.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("MECHANISM ANALYSIS SUMMARY")
print("="*70)
print(f"{'KV':>3} | {'Config':>12} | {'Gate Alpha':>12} | {'Ret Gate':>10}")
print("-"*40)
for kv in [4, 8, 12]:
    for cfg in ['controller', 'full']:
        r = results[kv][cfg]
        print(f"{kv:>3} | {cfg:>12} | {r['gate_alpha_mean']:>6.3f} ± {r['gate_alpha_std']:<4.2f} | {r['ret_gate_mean']:>6.3f} ± {r['ret_gate_std']:<4.2f}")
    print("-"*40)

print(f"\nResults saved to: archive/experiments/synergy_mechanism.json")
