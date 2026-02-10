#!/usr/bin/env python3
"""
Refined Multi-KV Capacity Study
- More epochs
- Param-matched baseline
- Single-KV sanity check
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import random
import os

from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM
from ana.train import col_fn


class MultiKVDataset(Dataset):
    def __init__(self, size=1000, vocab_size=50, num_kv_pairs=4, min_noise=3, max_noise=10):
        self.size = size
        self.vocab_size = vocab_size
        self.num_kv_pairs = num_kv_pairs
        self.min_noise = min_noise
        self.max_noise = max_noise
        
        self.TOK_KEY = 1
        self.TOK_VAL = 2
        self.TOK_QUERY = 3
        self.content = list(range(5, vocab_size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        kv_pairs = []
        for _ in range(self.num_kv_pairs):
            k = random.choice(self.content)
            v = random.choice(self.content)
            while any((k, v) == existing for existing in kv_pairs):
                k = random.choice(self.content)
                v = random.choice(self.content)
            kv_pairs.append((k, v))
        
        seq = []
        for k, v in kv_pairs:
            seq.extend([self.TOK_KEY, k, self.TOK_VAL, v])
        
        noise_len = random.randint(self.min_noise, self.max_noise)
        noise = [random.choice(self.content) for _ in range(noise_len)]
        seq.extend(noise)
        
        target_idx = random.randint(0, self.num_kv_pairs - 1)
        target_k, target_v = kv_pairs[target_idx]
        seq.extend([self.TOK_QUERY, target_k])
        
        full_seq = seq + [target_v]
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[-1] = 1.0
        
        return x, y, mask


class SingleKVDataset(Dataset):
    """Original single-KV task for sanity check."""
    
    def __init__(self, size=1000, vocab_size=30, min_noise=5, max_noise=15):
        self.size = size
        self.vocab_size = vocab_size
        self.min_noise = min_noise
        self.max_noise = max_noise
        
        self.TOK_KEY = 1
        self.TOK_VAL = 2
        self.TOK_QUERY = 3
        self.content = list(range(5, vocab_size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        k = random.choice(self.content)
        v = random.choice(self.content)
        
        seq = [self.TOK_KEY, k, self.TOK_VAL, v]
        
        noise_len = random.randint(self.min_noise, self.max_noise)
        noise = [random.choice(self.content) for _ in range(noise_len)]
        seq.extend(noise)
        seq.extend([self.TOK_QUERY, k])
        
        full_seq = seq + [v]
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[-1] = 1.0
        
        return x, y, mask


def train_and_evaluate(model, dataset, epochs=50, batch_size=16, lr=1e-3, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=col_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss_raw = loss_raw.view(y.size())
            loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
    
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    results = {'sanity_check': {}, 'capacity': {}}
    
    print("\n" + "="*70)
    print("SANITY CHECK: Single-KV (noise 5-15)")
    print("="*70)
    
    single_dataset = SingleKVDataset(size=1000, vocab_size=30, min_noise=5, max_noise=15)
    
    config = ANAConfig(
        d_model=64, state_dim=64, num_layers=2, track_count=2, key_dim=64, vocab_size=30,
        use_parallel_scan=True, use_hololink=True, use_controller=True
    )
    
    ana_model = ANAModel(config).to(device)
    ana_acc = train_and_evaluate(ana_model, single_dataset, epochs=50, device=device)
    print(f"ANA Single-KV: {ana_acc*100:.1f}% ({count_params(ana_model):,} params)")
    results['sanity_check']['ana'] = ana_acc
    
    baseline_config = ANAConfig(
        d_model=64, state_dim=64, num_layers=2, vocab_size=30,
        use_parallel_scan=True, use_hololink=False, use_controller=False
    )
    baseline_model = BaselineSSM(baseline_config).to(device)
    baseline_acc = train_and_evaluate(baseline_model, single_dataset, epochs=50, device=device)
    print(f"Baseline Single-KV: {baseline_acc*100:.1f}% ({count_params(baseline_model):,} params)")
    results['sanity_check']['baseline'] = baseline_acc
    
    print("\n" + "="*70)
    print("CAPACITY STUDY: Multi-KV")
    print("="*70)
    
    vocab_size = 50
    kv_pairs_list = [1, 2, 4, 8]
    
    for num_kv in kv_pairs_list:
        print(f"\n--- {num_kv} KV pairs ---")
        
        dataset = MultiKVDataset(
            size=1000, vocab_size=vocab_size,
            num_kv_pairs=num_kv, min_noise=3, max_noise=8
        )
        
        config = ANAConfig(
            d_model=64, state_dim=64, num_layers=2, track_count=2, key_dim=64,
            vocab_size=vocab_size, use_parallel_scan=True, use_hololink=True, use_controller=True
        )
        
        ana_model = ANAModel(config).to(device)
        ana_acc = train_and_evaluate(ana_model, dataset, epochs=50, device=device)
        
        baseline_config = ANAConfig(
            d_model=64, state_dim=64, num_layers=2, vocab_size=vocab_size,
            use_parallel_scan=True, use_hololink=False, use_controller=False
        )
        baseline_model = BaselineSSM(baseline_config).to(device)
        baseline_acc = train_and_evaluate(baseline_model, dataset, epochs=50, device=device)
        
        results['capacity'][num_kv] = {
            'ana': ana_acc,
            'baseline': baseline_acc,
            'delta': ana_acc - baseline_acc
        }
        
        print(f"  ANA: {ana_acc*100:.1f}% | Baseline: {baseline_acc*100:.1f}% | Delta: {(ana_acc-baseline_acc)*100:+.1f}%")
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nSanity Check (Single-KV, noise 5-15):")
    print(f"  ANA: {results['sanity_check']['ana']*100:.1f}%")
    print(f"  Baseline: {results['sanity_check']['baseline']*100:.1f}%")
    print(f"\nCapacity (Multi-KV):")
    print(f"{'KV':<5} {'ANA':>10} {'Baseline':>10} {'Delta':>10}")
    print("-"*40)
    for kv, r in results['capacity'].items():
        print(f"{kv:<5} {r['ana']*100:>9.1f}% {r['baseline']*100:>9.1f}% {r['delta']*100:>+9.1f}%")
    
    os.makedirs('archive', exist_ok=True)
    with open('archive/capacity_study_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to archive/capacity_study_v2.json")
    
    return results


if __name__ == '__main__':
    main()
