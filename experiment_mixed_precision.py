#!/usr/bin/env python3
# Experiment 1: Mixed Precision Training
import torch
from ana.config import ANAConfig, TrainingConfig, DataConfig
from ana.models import ANAModel
from ana.train import run_training
import time
import json

results = {}

for dtype_name, dtype in [('FP32', torch.float32), ('FP16', torch.float16)]:
    print(f'\n{"="*60}')
    print(f'MIXED PRECISION: {dtype_name}')
    print(f'{"="*60}')

    config = ANAConfig(d_model=64, state_dim=64, num_layers=2, track_count=2, vocab_size=50)
    train_config = TrainingConfig(batch_size=16, epochs=10, stage='2a', save_checkpoints=False)
    data_config = DataConfig(vocab_size=50, min_noise=10, max_noise=50, dataset_size=1000)

    model = ANAModel(config)
    device = torch.device('cuda')

    if dtype == torch.float16:
        model = model.half()
    model = model.to(device)

    # Disable num_workers for FP16 compatibility
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    history = run_training(config, train_config, data_config, model_type='ana', num_workers=0)
    train_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    results[dtype_name] = {
        'train_time': train_time,
        'peak_memory_mb': peak_memory,
        'final_loss': history['loss'][-1],
        'final_acc': history['needle_acc'][-1],
    }

    print(f'Peak Memory: {peak_memory:.1f} MB')
    print(f'Training Time: {train_time:.1f}s')

print(f'\n{"="*60}')
print('MIXED PRECISION SUMMARY')
print(f'{"="*60}')
print('Type      Time (s)   Peak MB      Loss        Acc')
print('-'*55)
for dtype_name, res in results.items():
    print(f'{dtype_name:>8} {res["train_time"]:>12.1f} {res["peak_memory_mb"]:>12.1f} {res["final_loss"]:>10.4f} {res["final_acc"]:>10.2%}')

with open('archive/phase4_mixed_precision.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to archive/phase4_mixed_precision.json')
