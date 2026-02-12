"""
Breakthrough Demonstration: Small ANA vs Larger Baselines

Core claim: A tiny ANA (~500K params) with HoloLink can match or beat
larger models on memory-intensive tasks due to constant-size associative memory.

Key comparisons:
1. Tiny ANA vs Larger SSM (same architecture, no HoloLink)
2. ANA scaling: small model with HoloLink vs large model without
3. Memory capacity: how many KV pairs can each handle?
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from ana import ANAConfig, ANAModel


def generate_kv_task(batch_size, num_pairs, vocab_size, noise_len=10):
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    content = list(range(4, vocab_size))
    
    inputs, targets = [], []
    for _ in range(batch_size):
        keys = random.sample(content, num_pairs)
        vals = random.sample([t for t in content if t not in keys], num_pairs)
        
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([TOK_KEY, k, TOK_VAL, v])
        
        seq.extend(random.choices(content, k=noise_len))
        
        q_idx = random.randint(0, num_pairs - 1)
        seq.extend([TOK_QUERY, keys[q_idx]])
        
        inputs.append(seq)
        targets.append(vals[q_idx])
    
    max_len = max(len(s) for s in inputs)
    x = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, s in enumerate(inputs):
        x[i, :len(s)] = torch.tensor(s)
    
    return x, torch.tensor(targets)


def evaluate(model, num_pairs, vocab_size, device, n_eval=200):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n_eval // 32):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits[:, -1].argmax(-1)
            correct += (pred == y).sum().item()
    model.train()
    return correct / n_eval


def get_component_params(model):
    holo_params, ctl_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if 'holo' in name:
            holo_params.append(p)
        elif 'controller' in name:
            ctl_params.append(p)
        else:
            other_params.append(p)
    return holo_params, ctl_params, other_params


def train_two_phase(model, vocab_size, device, curriculum, verbose=True):
    """Two-phase training."""
    holo_params, ctl_params, other_params = get_component_params(model)
    
    for p in ctl_params:
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(list(holo_params) + other_params, lr=1e-3)
    
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
        
        if verbose:
            acc = evaluate(model, num_pairs, vocab_size, device)
            print(f"  Phase 1 @ {num_pairs} pairs: {100*acc:.1f}%")
    
    for p in ctl_params:
        p.requires_grad = True
    for p in holo_params:
        p.requires_grad = False
    
    optimizer_ctl = torch.optim.Adam(ctl_params, lr=1e-4)
    target_pairs = curriculum[-1][0]
    
    for step in range(500):
        x, y = generate_kv_task(32, target_pairs, vocab_size)
        x, y = x.to(device), y.to(device)
        optimizer_ctl.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        loss.backward()
        optimizer_ctl.step()
    
    return evaluate(model, target_pairs, vocab_size, device)


def train_standard(model, vocab_size, device, curriculum, verbose=True):
    """Standard joint training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
        
        if verbose:
            acc = evaluate(model, num_pairs, vocab_size, device)
            print(f"  @ {num_pairs} pairs: {100*acc:.1f}%")
    
    target_pairs = curriculum[-1][0]
    return evaluate(model, target_pairs, vocab_size, device)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 100
    curriculum = [(1, 600), (2, 600), (4, 700), (6, 700), (8, 800), (10, 800), (12, 900), (16, 1000), (20, 1200)]
    
    print("=" * 70)
    print("BREAKTHROUGH DEMO: Small ANA vs Larger Baselines")
    print("=" * 70)
    print(f"Device: {device}\n")
    
    results = {}
    
    # ========== MODEL 1: Tiny ANA with HoloLink (~500K params) ==========
    print("-" * 70)
    print("MODEL 1: Tiny ANA with HoloLink (~500K params)")
    print("-" * 70)
    
    config_tiny = ANAConfig(
        d_model=64, state_dim=64, key_dim=64,
        vocab_size=vocab_size, use_hololink=True, use_controller=False,
        use_parallel_scan=True, track_count=1, num_layers=1
    )
    model_tiny = ANAModel(config_tiny).to(device)
    params_tiny = sum(p.numel() for p in model_tiny.parameters())
    print(f"Parameters: {params_tiny:,}")
    
    acc_tiny = train_standard(model_tiny, vocab_size, device, curriculum)
    results['tiny_ana'] = {'params': params_tiny, 'acc': acc_tiny}
    print(f"\nFinal accuracy at 20 pairs: {100*acc_tiny:.1f}%\n")
    
    # ========== MODEL 2: Medium SSM without HoloLink (~1M params) ==========
    print("-" * 70)
    print("MODEL 2: Medium SSM without HoloLink (~1M params)")
    print("-" * 70)
    
    config_medium = ANAConfig(
        d_model=96, state_dim=96, key_dim=64,
        vocab_size=vocab_size, use_hololink=False, use_controller=False,
        use_parallel_scan=True, track_count=1, num_layers=2
    )
    model_medium = ANAModel(config_medium).to(device)
    params_medium = sum(p.numel() for p in model_medium.parameters())
    print(f"Parameters: {params_medium:,} ({params_medium/params_tiny:.1f}x larger)")
    
    acc_medium = train_standard(model_medium, vocab_size, device, curriculum)
    results['medium_ssm'] = {'params': params_medium, 'acc': acc_medium}
    print(f"\nFinal accuracy at 20 pairs: {100*acc_medium:.1f}%\n")
    
    # ========== MODEL 3: Large SSM without HoloLink (~2M params) ==========
    print("-" * 70)
    print("MODEL 3: Large SSM without HoloLink (~2M params)")
    print("-" * 70)
    
    config_large = ANAConfig(
        d_model=128, state_dim=128, key_dim=64,
        vocab_size=vocab_size, use_hololink=False, use_controller=False,
        use_parallel_scan=True, track_count=2, num_layers=2
    )
    model_large = ANAModel(config_large).to(device)
    params_large = sum(p.numel() for p in model_large.parameters())
    print(f"Parameters: {params_large:,} ({params_large/params_tiny:.1f}x larger)")
    
    acc_large = train_standard(model_large, vocab_size, device, curriculum)
    results['large_ssm'] = {'params': params_large, 'acc': acc_large}
    print(f"\nFinal accuracy at 20 pairs: {100*acc_large:.1f}%\n")
    
    # ========== MODEL 4: Larger ANA with HoloLink (~1M params) ==========
    print("-" * 70)
    print("MODEL 4: Larger ANA with HoloLink (~1M params)")
    print("-" * 70)
    
    config_ana_lg = ANAConfig(
        d_model=96, state_dim=96, key_dim=96,
        vocab_size=vocab_size, use_hololink=True, use_controller=False,
        use_parallel_scan=True, track_count=1, num_layers=2
    )
    model_ana_lg = ANAModel(config_ana_lg).to(device)
    params_ana_lg = sum(p.numel() for p in model_ana_lg.parameters())
    print(f"Parameters: {params_ana_lg:,} ({params_ana_lg/params_tiny:.1f}x larger)")
    
    acc_ana_lg = train_standard(model_ana_lg, vocab_size, device, curriculum)
    results['large_ana'] = {'params': params_ana_lg, 'acc': acc_ana_lg}
    print(f"\nFinal accuracy at 20 pairs: {100*acc_ana_lg:.1f}%\n")
    
    # ========== RESULTS ==========
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'Params':>10} {'Accuracy':>10} {'Params/ Tiny':>12}")
    print("-" * 70)
    
    for name, data in results.items():
        ratio = data['params'] / params_tiny
        print(f"{name:<30} {data['params']:>10,} {100*data['acc']:>9.1f}% {ratio:>11.1f}x")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    
    if acc_tiny > acc_medium:
        improvement = acc_tiny - acc_medium
        print(f"""
Tiny ANA ({params_tiny:,} params) BEATS Medium SSM ({params_medium:,} params)
by {100*improvement:.1f}% absolute with {params_medium/params_tiny:.1f}x fewer parameters!

This demonstrates "punching above weight": HoloLink's constant-size
associative memory provides capabilities that would require much larger
models without this architecture.
""")
    else:
        print(f"""
Results need analysis:
- Tiny ANA: {100*acc_tiny:.1f}%
- Medium SSM: {100*acc_medium:.1f}%
- Large SSM: {100*acc_large:.1f}%
- Large ANA: {100*acc_ana_lg:.1f}%
""")
    
    # ========== SCALING TEST ==========
    print("=" * 70)
    print("MEMORY CAPACITY TEST: How many KV pairs can each handle?")
    print("=" * 70)
    
    test_pairs = [4, 8, 12, 16, 20, 24, 32]
    
    print(f"\n{'KV Pairs':<10} {'Tiny ANA':>12} {'Med SSM':>12} {'Large SSM':>12} {'Lg ANA':>12}")
    print("-" * 60)
    
    for n_pairs in test_pairs:
        tiny_acc = evaluate(model_tiny, n_pairs, vocab_size, device)
        med_acc = evaluate(model_medium, n_pairs, vocab_size, device)
        lg_acc = evaluate(model_large, n_pairs, vocab_size, device)
        ana_lg_acc = evaluate(model_ana_lg, n_pairs, vocab_size, device)
        
        print(f"{n_pairs:<10} {100*tiny_acc:>11.1f}% {100*med_acc:>11.1f}% {100*lg_acc:>11.1f}% {100*ana_lg_acc:>11.1f}%")
    
    return results


if __name__ == "__main__":
    results = main()
