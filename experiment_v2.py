#!/usr/bin/env python3
"""
Comprehensive experiments comparing ANA v1 vs v2 on associative recall.
Tests multiple improvements:
1. External memory
2. Selective attention
3. Query-gated routing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from ana.config import ANAConfig
from ana.models import ANAModel
from ana.models_v2 import ANAModelV2
from ana.data import AssociativeRecallDataset
from ana.train import col_fn


class MultiKVDataset(torch.utils.data.Dataset):
    """Multi-KV associative recall to force memory usage."""
    
    def __init__(self, size=1000, vocab_size=50, num_kv_pairs=3, min_noise=5, max_noise=15):
        self.size = size
        self.vocab_size = vocab_size
        self.num_kv_pairs = num_kv_pairs
        self.min_noise = min_noise
        self.max_noise = max_noise
        
        self.TOK_KEY = 1
        self.TOK_VAL = 2
        self.TOK_QUERY = 3
        self.SEP = 4
        self.content = list(range(5, vocab_size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        import random
        
        # Generate multiple KV pairs
        kv_pairs = []
        for _ in range(self.num_kv_pairs):
            k = random.choice(self.content)
            v = random.choice(self.content)
            kv_pairs.append((k, v))
        
        # Build sequence: [KEY K1 VAL V1] [KEY K2 VAL V2] ... noise ... [QUERY K_target]
        seq = []
        for k, v in kv_pairs:
            seq.extend([self.TOK_KEY, k, self.TOK_VAL, v])
        
        # Add noise
        noise_len = random.randint(self.min_noise, self.max_noise)
        noise = [random.choice(self.content) for _ in range(noise_len)]
        seq.extend(noise)
        
        # Query: pick one of the keys randomly
        target_idx = random.randint(0, self.num_kv_pairs - 1)
        target_k, target_v = kv_pairs[target_idx]
        seq.extend([self.TOK_QUERY, target_k])
        
        # Input/target for LM
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor(seq[1:] + [target_v], dtype=torch.long)
        
        # Mask: focus on final prediction
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[-1] = 1.0
        
        return x, y, mask


def train_model(model, loader, epochs, device, lr=1e-3):
    """Train model and return history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward'):
                logits, _ = model(x)
            else:
                logits, _ = model(x)
            
            loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss_raw = loss_raw.view(y.size())
            loss = (loss_raw * mask).sum() / mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Needle accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds[:, -1] == y[:, -1]).float().sum().item()
            total += x.size(0)
        
        history['loss'].append(total_loss / len(loader))
        history['acc'].append(correct / total)
    
    return history


def evaluate_model(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            logits, _ = model(x)
            preds = logits.argmax(dim=-1)
            
            correct += (preds[:, -1] == y[:, -1]).float().sum().item()
            total += x.size(0)
    
    return correct / total


def run_experiment(config_name, model_class, config, dataset_class, dataset_args, 
                   epochs=30, num_runs=3, device='cuda'):
    """Run multiple training runs and aggregate results."""
    results = []
    
    for run in range(num_runs):
        # Create dataset
        ds = dataset_class(**dataset_args)
        loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=col_fn, num_workers=0)
        
        # Create model
        model = model_class(config).to(device)
        
        # Train
        start_time = time.time()
        history = train_model(model, loader, epochs, device)
        train_time = time.time() - start_time
        
        # Evaluate
        final_acc = evaluate_model(model, loader, device)
        
        results.append({
            'run': run + 1,
            'final_loss': history['loss'][-1],
            'final_acc': final_acc,
            'train_time': train_time,
            'history': history
        })
    
    # Aggregate
    accs = [r['final_acc'] for r in results]
    losses = [r['final_loss'] for r in results]
    
    return {
        'config_name': config_name,
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'runs': results
    }


def main():
    print('=' * 70)
    print('ANA V2 COMPREHENSIVE EXPERIMENTS')
    print('=' * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    vocab = 30
    
    # Base config
    base_config = ANAConfig(
        d_model=64,
        state_dim=64,
        num_layers=2,
        track_count=2,
        key_dim=64,
        vocab_size=vocab,
        use_position_encoding=True
    )
    
    all_results = {}
    
    # EXPERIMENT 1: Single KV, increasing noise
    print('\n' + '=' * 70)
    print('EXPERIMENT 1: Single KV - Noise Scaling')
    print('=' * 70)
    
    noise_levels = [5, 10, 15, 20, 25]
    exp1_results = {'v1': {}, 'v2': {}}
    
    for max_noise in noise_levels:
        print(f'\n--- Noise 2-{max_noise} ---')
        
        # V1
        config_v1 = ANAConfig(**{**base_config.__dict__})
        result_v1 = run_experiment(
            f'v1_noise_{max_noise}',
            ANAModel,
            config_v1,
            AssociativeRecallDataset,
            {'size': 500, 'vocab_size': vocab, 'min_noise': 2, 'max_noise': max_noise},
            epochs=25,
            num_runs=2,
            device=device
        )
        exp1_results['v1'][max_noise] = result_v1
        print(f'  V1: {result_v1["mean_acc"]*100:.1f}% +/- {result_v1["std_acc"]*100:.1f}')
        
        # V2
        config_v2 = ANAConfig(**{**base_config.__dict__})
        result_v2 = run_experiment(
            f'v2_noise_{max_noise}',
            ANAModelV2,
            config_v2,
            AssociativeRecallDataset,
            {'size': 500, 'vocab_size': vocab, 'min_noise': 2, 'max_noise': max_noise},
            epochs=25,
            num_runs=2,
            device=device
        )
        exp1_results['v2'][max_noise] = result_v2
        print(f'  V2: {result_v2["mean_acc"]*100:.1f}% +/- {result_v2["std_acc"]*100:.1f}')
        print(f'  Delta: +{(result_v2["mean_acc"] - result_v1["mean_acc"])*100:.1f}%')
    
    all_results['exp1_noise_scaling'] = exp1_results
    
    # EXPERIMENT 2: Multi-KV task
    print('\n' + '=' * 70)
    print('EXPERIMENT 2: Multi-KV Recall')
    print('=' * 70)
    
    kv_counts = [1, 2, 3, 4]
    exp2_results = {'v1': {}, 'v2': {}}
    
    for num_kv in kv_counts:
        print(f'\n--- {num_kv} KV pairs ---')
        
        # V1
        result_v1 = run_experiment(
            f'v1_{num_kv}kv',
            ANAModel,
            config_v1,
            MultiKVDataset,
            {'size': 500, 'vocab_size': vocab, 'num_kv_pairs': num_kv, 'min_noise': 3, 'max_noise': 10},
            epochs=30,
            num_runs=2,
            device=device
        )
        exp2_results['v1'][num_kv] = result_v1
        print(f'  V1: {result_v1["mean_acc"]*100:.1f}%')
        
        # V2
        result_v2 = run_experiment(
            f'v2_{num_kv}kv',
            ANAModelV2,
            config_v2,
            MultiKVDataset,
            {'size': 500, 'vocab_size': vocab, 'num_kv_pairs': num_kv, 'min_noise': 3, 'max_noise': 10},
            epochs=30,
            num_runs=2,
            device=device
        )
        exp2_results['v2'][num_kv] = result_v2
        print(f'  V2: {result_v2["mean_acc"]*100:.1f}%')
        print(f'  Delta: +{(result_v2["mean_acc"] - result_v1["mean_acc"])*100:.1f}%')
    
    all_results['exp2_multi_kv'] = exp2_results
    
    # Print summary
    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)
    
    print('\nExperiment 1: Noise Scaling')
    print('-' * 50)
    print(f'{"Noise":<10} {"V1":>12} {"V2":>12} {"Delta":>10}')
    print('-' * 50)
    for noise in noise_levels:
        v1 = exp1_results['v1'][noise]['mean_acc'] * 100
        v2 = exp2_results['v2'].get(noise, exp1_results['v2'][noise])['mean_acc'] * 100
        # Get from exp1 for noise
        v1 = exp1_results['v1'][noise]['mean_acc'] * 100
        v2 = exp1_results['v2'][noise]['mean_acc'] * 100
        print(f'{f"2-{noise}":<10} {v1:>11.1f}% {v2:>11.1f}% {v2-v1:>+9.1f}%')
    
    print('\nExperiment 2: Multi-KV')
    print('-' * 50)
    print(f'{"KV Pairs":<10} {"V1":>12} {"V2":>12} {"Delta":>10}')
    print('-' * 50)
    for num_kv in kv_counts:
        v1 = exp2_results['v1'][num_kv]['mean_acc'] * 100
        v2 = exp2_results['v2'][num_kv]['mean_acc'] * 100
        print(f'{num_kv:<10} {v1:>11.1f}% {v2:>11.1f}% {v2-v1:>+9.1f}%')
    
    # Save results
    with open('archive/v2_experiments.json', 'w') as f:
        # Convert numpy arrays to lists for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)
    
    print('\nResults saved to archive/v2_experiments.json')


if __name__ == '__main__':
    main()
