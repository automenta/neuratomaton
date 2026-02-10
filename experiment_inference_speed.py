#!/usr/bin/env python3
# Experiment 3: Inference Speed Benchmarking
import torch
from ana.config import ANAConfig
from ana.models import ANAModel
import time

# Setup
config = ANAConfig(d_model=64, state_dim=64, num_layers=2, track_count=2, vocab_size=50)
model = ANAModel(config)
model = model.to(torch.device('cuda'))
model.eval()

seq_lengths = [64, 128, 256, 512]
batch_size = 16
num_warmup = 10
num_iterations = 100

print('='*70)
print('INFERENCE SPEED BENCHMARKING')
print('='*70)
print()

results = {}

for seq_len in seq_lengths:
    print(f'Sequence Length: {seq_len}')
    print('-'*70)

    # Warmup
    for _ in range(num_warmup):
        x = torch.randint(0, 50, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            _ = model(x)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    total_tokens = 0

    for _ in range(num_iterations):
        x = torch.randint(0, 50, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            _ = model(x)
            total_tokens += batch_size * seq_len

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    tokens_per_sec = total_tokens / elapsed_time
    ms_per_seq = (elapsed_time / num_iterations) * 1000

    results[seq_len] = {
        'tokens_per_sec': tokens_per_sec,
        'ms_per_sequence': ms_per_seq,
    }

    print(f'  Tokens/sec: {tokens_per_sec:,.0f}')
    print(f'  ms/sequence: {ms_per_seq:.2f}')
    print()

# Get memory for inference
x = torch.randint(0, 50, (batch_size, 512)).cuda()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model(x)
peak_memory = torch.cuda.max_memory_allocated() / 1024**2

print('='*70)
print('INFERENCE SPEED SUMMARY')
print('='*70)
print('Seq Len   Tokens/sec    ms/seq')
print('-'*50)
for seq_len, res in results.items():
    print(f'{seq_len:>8} {res["tokens_per_sec"]:>12,.0f} {res["ms_per_sequence"]:>10.2f}')
print(f'Peak Memory (inference): {peak_memory:.1f} MB')
print()

import json
with open('archive/phase4_inference_speed.json', 'w') as f:
    json.dump({'results': results, 'peak_memory_mb': peak_memory}, f, indent=2)
print('Results saved to archive/phase4_inference_speed.json')
