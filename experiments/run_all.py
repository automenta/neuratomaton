#!/usr/bin/env python3
"""
Run all experiments for ANA validation.
Total runtime: ~1-2 hours on commodity GPU.
"""
import os
import sys
import json
import time
import torch
import random

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM
import torch.nn.functional as F
from torch.utils import data

# ============================================================================
# Common utilities
# ============================================================================

class QuickMultiKV(data.Dataset):
    def __init__(self, size=400, num_kv=4, min_noise=3, max_noise=10):
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

def train_eval(model, num_kv, epochs=15):
    ds = QuickMultiKV(size=400, num_kv=num_kv)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
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

# ============================================================================
# Simple Transformer baseline (matched params)
# ============================================================================

class SimpleTransformer(torch.nn.Module):
    def __init__(self, vocab_size=30, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=d_model*4
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = torch.nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        h = self.embedding(x)
        h = self.transformer(h)
        return self.output(h), {}

# ============================================================================
# Experiment 1: ANA vs Transformer Comparison
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 1: ANA vs Transformer Comparison")
print("="*70)

results_exp1 = {}
for kv in [1, 2, 4, 8]:
    print(f"\n  Testing {kv} KV pairs...")
    
    ana = ANAModel(ANAConfig(d_model=64, vocab_size=30)).to(device)
    ana_params = sum(p.numel() for p in ana.parameters())
    ana_acc = train_eval(ana, kv)
    
    xformer = SimpleTransformer(vocab_size=30, d_model=64, n_heads=4, n_layers=2).to(device)
    xf_params = sum(p.numel() for p in xformer.parameters())
    xf_acc = train_eval(xformer, kv)
    
    results_exp1[kv] = {'ana': ana_acc, 'transformer': xf_acc, 'ana_params': ana_params, 'xf_params': xf_params}
    print(f"    ANA ({ana_params:,} params): {ana_acc*100:.1f}%")
    print(f"    Transformer ({xf_params:,} params): {xf_acc*100:.1f}%")

# ============================================================================
# Experiment 2: Synergy Reproducibility (3 seeds)
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 2: Synergy Reproducibility")
print("="*70)

configs = {
    'baseline': {'use_hololink': False, 'use_controller': False},
    'controller': {'use_hololink': False, 'use_controller': True},
    'hololink': {'use_hololink': True, 'use_controller': False},
    'full': {'use_hololink': True, 'use_controller': True},
}

results_exp2 = {kv: {c: [] for c in configs} for kv in [4, 8]}

for kv in [4, 8]:
    print(f"\n  {kv} KV pairs (3 seeds each):")
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        random.seed(seed)
        for name, flags in configs.items():
            if name == 'baseline':
                model = BaselineSSM(ANAConfig(d_model=64, vocab_size=30)).to(device)
            else:
                model = ANAModel(ANAConfig(d_model=64, vocab_size=30, **flags)).to(device)
            acc = train_eval(model, kv, epochs=15)
            results_exp2[kv][name].append(acc)
    
    for name in configs:
        accs = results_exp2[kv][name]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean)**2 for a in accs) / len(accs)) ** 0.5
        print(f"    {name:12s}: {mean*100:.1f}% ± {std*100:.1f}%")
    
    # Calculate synergy
    single_best = max(sum(results_exp2[kv][c])/3 for c in ['controller', 'hololink'])
    full_avg = sum(results_exp2[kv]['full'])/3
    synergy = full_avg - single_best
    print(f"    Synergy: +{synergy*100:.1f}%")

# ============================================================================
# Experiment 3: Inference Speed
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 3: Inference Speed Benchmark")
print("="*70)

results_exp3 = {}

for seq_len in [64, 128, 256, 512, 1024]:
    x = torch.randint(0, 30, (1, seq_len)).to(device)
    
    # ANA
    ana = ANAModel(ANAConfig(d_model=64, vocab_size=30)).to(device)
    ana.eval()
    with torch.no_grad():
        for _ in range(10): ana(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): ana(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ana_time = (time.time() - t0) / 100 * 1000
    
    # Transformer
    xformer = SimpleTransformer(vocab_size=30, d_model=64).to(device)
    xformer.eval()
    with torch.no_grad():
        for _ in range(10): xformer(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): xformer(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        xf_time = (time.time() - t0) / 100 * 1000
    
    results_exp3[seq_len] = {'ana_ms': ana_time, 'transformer_ms': xf_time}
    print(f"  Len {seq_len:4d}: ANA={ana_time:.2f}ms, Transformer={xf_time:.2f}ms, ratio={xf_time/ana_time:.2f}x")

# ============================================================================
# Save and Summarize
# ============================================================================
results = {
    'exp1_comparison': results_exp1,
    'exp2_synergy': {k: {c: [a*100 for a in accs] for c, accs in v.items()} for k, v in results_exp2.items()},
    'exp3_inference': results_exp3,
}

with open('archive/experiments/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n1. ANA vs Transformer Accuracy:")
print(f"   {'KV':>4} | {'ANA':>8} | {'Transformer':>12}")
print("   " + "-"*30)
for kv, r in results_exp1.items():
    print(f"   {kv:>4} | {r['ana']*100:>7.1f}% | {r['transformer']*100:>11.1f}%")

print("\n2. Synergy Effect at 8 KV:")
accs_8 = results_exp2[8]
for name in ['baseline', 'controller', 'hololink', 'full']:
    mean = sum(accs_8[name])/3 * 100
    print(f"   {name:12s}: {mean:.1f}%")

print("\n3. Inference Scaling (ms):")
print(f"   {'Length':>8} | {'ANA':>8} | {'Transformer':>12} | {'Ratio':>8}")
for seq_len, r in results_exp3.items():
    print(f"   {seq_len:>8} | {r['ana_ms']:>7.2f}ms | {r['transformer_ms']:>11.2f}ms | {r['transformer_ms']/r['ana_ms']:>7.2f}x")

print(f"\nResults saved to: archive/experiments/all_results.json")
