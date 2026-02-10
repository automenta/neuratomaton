#!/usr/bin/env python3
"""
Phase B: Long Sequence Benchmark
Demonstrate O(1) inference advantage for ANA vs O(n) for Transformer
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=30, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=d_model*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        h = self.embedding(x)
        h = self.transformer(h)
        return self.output(h), {}

print("="*70)
print("PHASE B: LONG SEQUENCE BENCHMARK")
print("="*70)

SEQ_LENGTHS = [512, 1024, 2048, 4096]
WARMUP_RUNS = 10
TIMED_RUNS = 100

ana_cfg = ANAConfig(d_model=64, vocab_size=30, num_layers=2, max_seq_len=8192)
ana = ANAModel(ana_cfg).to(device)
ana.eval()

xformer = SimpleTransformer(vocab_size=30, d_model=64, n_heads=4, n_layers=2).to(device)
xformer.eval()

results = {}

for seq_len in SEQ_LENGTHS:
    print(f"\nTesting sequence length: {seq_len}")
    x = torch.randint(0, 30, (1, seq_len)).to(device)
    
    with torch.no_grad():
        for _ in range(WARMUP_RUNS): ana(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TIMED_RUNS): ana(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ana_time = (time.perf_counter() - t0) / TIMED_RUNS * 1000
    
    with torch.no_grad():
        for _ in range(WARMUP_RUNS): xformer(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TIMED_RUNS): xformer(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        xf_time = (time.perf_counter() - t0) / TIMED_RUNS * 1000
    
    results[seq_len] = {'ana_ms': ana_time, 'transformer_ms': xf_time}
    print(f"  ANA: {ana_time:.2f}ms | Transformer: {xf_time:.2f}ms | Ratio: {xf_time/ana_time:.2f}x")

ana_memory = sum(p.numel() * 4 for p in ana.parameters()) / 1024 / 1024
xf_memory = sum(p.numel() * 4 for p in xformer.parameters()) / 1024 / 1024
results['model_memory_mb'] = {'ana': ana_memory, 'transformer': xf_memory}

with open('archive/experiments/phaseB_longseq.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("LONG SEQUENCE BENCHMARK RESULTS")
print("="*70)
print(f"{'Seq Len':>10} | {'ANA (ms)':>10} | {'Transformer (ms)':>16} | {'Speedup':>10}")
print("-"*60)
for seq_len, r in results.items():
    if isinstance(seq_len, int):
        print(f"{seq_len:>10} | {r['ana_ms']:>10.2f} | {r['transformer_ms']:>16.2f} | {r['transformer_ms']/r['ana_ms']:>9.2f}x")

print(f"\nModel memory: ANA={ana_memory:.2f}MB, Transformer={xf_memory:.2f}MB")
print(f"\nResults saved to: archive/experiments/phaseB_longseq.json")
