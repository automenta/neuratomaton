"""
Quick KV Scaling Experiment for ANA

Validates the two-phase training advantage at larger scales.
Runs in ~10-15 minutes on 10GB VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from typing import List, Dict, Tuple

import sys
sys.path.insert(0, '/home/me/ana')

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


def train_two_phase(num_pairs_target, vocab_size, device, verbose=True):
    """Two-phase training to reach num_pairs_target KV pairs."""
    
    config = ANAConfig(
        d_model=64,
        vocab_size=vocab_size,
        state_dim=64,
        key_dim=64,
        use_hololink=True,
        use_controller=True,
        use_parallel_scan=True,
        track_count=1,
        num_layers=1
    )
    
    model = ANAModel(config).to(device)
    holo_params, ctl_params, other_params = get_component_params(model)
    
    if verbose:
        print(f"\n=== TWO-PHASE: Training to {num_pairs_target} KV pairs ===")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    curriculum = []
    n = 1
    while n <= num_pairs_target:
        steps = 400 + (n - 1) * 50
        curriculum.append((n, steps))
        n = min(n + 2, num_pairs_target) if n < 8 else n + 4
    if curriculum[-1][0] != num_pairs_target:
        curriculum.append((num_pairs_target, 800))
    
    for p in ctl_params:
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(list(holo_params) + other_params, lr=1e-3)
    
    if verbose:
        print("\nPhase 1: Training HoloLink...")
    
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model, num_pairs, vocab_size, device)
        if verbose:
            status = '✅' if acc > 0.9 else ('⚠️' if acc > 0.7 else '❌')
            print(f"  {num_pairs} pairs: {100*acc:.1f}% {status}")
    
    phase1_acc = evaluate(model, num_pairs_target, vocab_size, device)
    
    for p in ctl_params:
        p.requires_grad = True
    for p in holo_params:
        p.requires_grad = False
    
    optimizer_ctl = torch.optim.Adam(ctl_params, lr=1e-4)
    
    if verbose:
        print("\nPhase 2: Fine-tuning Controller...")
    
    for step in range(500):
        x, y = generate_kv_task(32, num_pairs_target, vocab_size)
        x, y = x.to(device), y.to(device)
        optimizer_ctl.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        loss.backward()
        optimizer_ctl.step()
        
        if verbose and (step + 1) % 100 == 0:
            acc = evaluate(model, num_pairs_target, vocab_size, device, n_eval=100)
            print(f"  Step {step+1}: {100*acc:.1f}%")
    
    phase2_acc = evaluate(model, num_pairs_target, vocab_size, device)
    
    if verbose:
        print(f"\nFinal: Phase 1={100*phase1_acc:.1f}%, Phase 2={100*phase2_acc:.1f}%")
    
    return model, phase1_acc, phase2_acc


def train_joint(num_pairs_target, vocab_size, device, verbose=True):
    """Joint training baseline."""
    
    config = ANAConfig(
        d_model=64,
        vocab_size=vocab_size,
        state_dim=64,
        key_dim=64,
        use_hololink=True,
        use_controller=True,
        use_parallel_scan=True,
        track_count=1,
        num_layers=1
    )
    
    model = ANAModel(config).to(device)
    
    if verbose:
        print(f"\n=== JOINT: Training to {num_pairs_target} KV pairs ===")
    
    curriculum = []
    n = 1
    while n <= num_pairs_target:
        steps = 400 + (n - 1) * 50
        curriculum.append((n, steps))
        n = min(n + 2, num_pairs_target) if n < 8 else n + 4
    if curriculum[-1][0] != num_pairs_target:
        curriculum.append((num_pairs_target, 800))
    
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
        
        acc = evaluate(model, num_pairs, vocab_size, device)
        if verbose:
            status = '✅' if acc > 0.9 else ('⚠️' if acc > 0.7 else '❌')
            print(f"  {num_pairs} pairs: {100*acc:.1f}% {status}")
    
    final_acc = evaluate(model, num_pairs_target, vocab_size, device)
    
    if verbose:
        print(f"\nFinal: {100*final_acc:.1f}%")
    
    return model, final_acc


def train_hololink_only(num_pairs_target, vocab_size, device, verbose=True):
    """HoloLink-only ablation (no controller)."""
    
    config = ANAConfig(
        d_model=64,
        vocab_size=vocab_size,
        state_dim=64,
        key_dim=64,
        use_hololink=True,
        use_controller=False,
        use_parallel_scan=True,
        track_count=1,
        num_layers=1
    )
    
    model = ANAModel(config).to(device)
    
    if verbose:
        print(f"\n=== HOLOLINK ONLY: Training to {num_pairs_target} KV pairs ===")
    
    curriculum = []
    n = 1
    while n <= num_pairs_target:
        steps = 400 + (n - 1) * 50
        curriculum.append((n, steps))
        n = min(n + 2, num_pairs_target) if n < 8 else n + 4
    if curriculum[-1][0] != num_pairs_target:
        curriculum.append((num_pairs_target, 800))
    
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
        
        acc = evaluate(model, num_pairs, vocab_size, device)
        if verbose:
            status = '✅' if acc > 0.9 else ('⚠️' if acc > 0.7 else '❌')
            print(f"  {num_pairs} pairs: {100*acc:.1f}% {status}")
    
    final_acc = evaluate(model, num_pairs_target, vocab_size, device)
    
    if verbose:
        print(f"\nFinal: {100*final_acc:.1f}%")
    
    return model, final_acc


def test_long_context_inference(model, vocab_size, device, max_length=64000):
    """Test that model can handle long sequences at inference."""
    model.eval()
    
    print(f"\n=== LONG CONTEXT INFERENCE TEST ===")
    
    for length in [4000, 8000, 16000, 32000, 64000]:
        if length > max_length:
            break
        try:
            torch.cuda.empty_cache()
            
            x = torch.randint(4, vocab_size, (1, length), device=device)
            
            start = time.time()
            with torch.no_grad():
                logits, _ = model(x)
            elapsed = time.time() - start
            
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {length:,} tokens: {elapsed:.2f}s, {mem:.1f}GB, {length/elapsed:.0f} tok/s")
            
        except RuntimeError as e:
            if "OOM" in str(e):
                print(f"  {length:,} tokens: OOM")
                break
            raise
    
    model.train()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    target_pairs = 24
    vocab_size = 80
    
    print("\n" + "="*60)
    print(f"KV SCALING EXPERIMENT: {target_pairs} pairs")
    print("="*60)
    
    model_tp, p1_acc, p2_acc = train_two_phase(target_pairs, vocab_size, device)
    
    model_joint, joint_acc = train_joint(target_pairs, vocab_size, device)
    
    model_holo, holo_acc = train_hololink_only(target_pairs, vocab_size, device)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Method':<25} {'Accuracy':>10}")
    print("-"*40)
    print(f"{'Two-Phase (Phase 1)':<25} {100*p1_acc:>9.1f}%")
    print(f"{'Two-Phase (Phase 2)':<25} {100*p2_acc:>9.1f}%")
    print(f"{'Joint Training':<25} {100*joint_acc:>9.1f}%")
    print(f"{'HoloLink Only':<25} {100*holo_acc:>9.1f}%")
    
    print(f"\n{'Synergy (Phase2 vs HoloLink):':<40} {100*(p2_acc - holo_acc):+.1f}%")
    print(f"{'Improvement (Phase2 vs Joint):':<40} {100*(p2_acc - joint_acc):+.1f}%")
    
    if p2_acc > 0.8 and p2_acc > joint_acc + 0.1:
        print("\n✅ SUCCESS: Two-phase training achieves high accuracy and beats joint!")
    elif p2_acc > joint_acc:
        print("\n⚠️ PARTIAL: Two-phase beats joint but accuracy could be higher")
    else:
        print("\n❌ Need investigation")
    
    test_long_context_inference(model_tp, vocab_size, device)


if __name__ == "__main__":
    main()
