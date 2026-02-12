"""
KV Scaling Experiment for ANA

Tests the core two-phase training claim at progressively larger scales.
Proves that ANA handles more KV pairs without performance collapse.

Key experiments:
1. Scale from 12 KV pairs to 64-128 pairs
2. Compare two-phase vs joint training
3. Show controller improves over HoloLink-only

Runs in <20 minutes on 10GB VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from torch.amp import autocast, GradScaler

import sys
sys.path.insert(0, '/home/me/ana')

from ana import ANAConfig, ANAModel


def generate_kv_task(
    batch_size: int,
    num_pairs: int,
    vocab_size: int,
    noise_len: int = 20,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate KV associative recall task."""
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
    x = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, s in enumerate(inputs):
        x[i, :len(s)] = torch.tensor(s, device=device)
    
    return x, torch.tensor(targets, device=device)


def get_component_params(model: nn.Module) -> Tuple[List, List, List]:
    holo_params, ctl_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if 'holo' in name:
            holo_params.append(p)
        elif 'controller' in name:
            ctl_params.append(p)
        else:
            other_params.append(p)
    return holo_params, ctl_params, other_params


def evaluate(model: nn.Module, num_pairs: int, vocab_size: int, device: str, n_eval: int = 200) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n_eval // 32):
            x, y = generate_kv_task(32, num_pairs, vocab_size, device=device)
            logits, _ = model(x)
            pred = logits[:, -1].argmax(-1)
            correct += (pred == y).sum().item()
    model.train()
    return correct / n_eval


def train_two_phase_curriculum(
    config: ANAConfig,
    device: str,
    curriculum: List[Tuple[int, int]] = None,
    phase2_steps: int = 500,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """Two-phase training with curriculum over KV pair counts."""
    
    model = ANAModel(config).to(device)
    holo_params, ctl_params, other_params = get_component_params(model)
    
    if curriculum is None:
        curriculum = [(1, 200), (2, 300), (4, 400), (8, 500), (12, 600), (16, 700), (24, 800)]
    
    results = {'phase1': {}, 'phase2': {}, 'final': {}}
    vocab_size = config.vocab_size
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TWO-PHASE TRAINING: KV SCALING")
        print(f"{'='*60}")
        print(f"Model: d={config.d_model}, tracks={config.track_count}")
        print(f"Curriculum: {curriculum}")
    
    for p in ctl_params:
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(list(holo_params) + other_params, lr=3e-4, weight_decay=0.01)
    scaler = GradScaler('cuda') if device == "cuda" else None
    
    if verbose:
        print(f"\n--- PHASE 1: Training HoloLink ---")
    
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size, device=device)
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=device == "cuda"):
                logits, _ = model(x)
                loss = F.cross_entropy(logits[:, -1, :].float(), y)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        acc = evaluate(model, num_pairs, vocab_size, device)
        results['phase1'][num_pairs] = acc
        
        if verbose:
            status = '✅' if acc > 0.85 else ('⚠️' if acc > 0.6 else '❌')
            print(f"  {num_pairs} pairs: {100*acc:.1f}% {status}")
    
    phase1_max = max(results['phase1'].keys())
    phase1_acc = results['phase1'][phase1_max]
    
    for p in ctl_params:
        p.requires_grad = True
    for p in holo_params:
        p.requires_grad = False
    
    optimizer_ctl = torch.optim.AdamW(ctl_params, lr=1e-4, weight_decay=0.01)
    
    if verbose:
        print(f"\n--- PHASE 2: Fine-tuning Controller ---")
    
    for step in range(phase2_steps):
        x, y = generate_kv_task(32, phase1_max, vocab_size, device=device)
        optimizer_ctl.zero_grad()
        
        with autocast('cuda', enabled=device == "cuda"):
            logits, _ = model(x)
            loss = F.cross_entropy(logits[:, -1, :].float(), y)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer_ctl)
            scaler.update()
        else:
            loss.backward()
            optimizer_ctl.step()
        
        if verbose and (step + 1) % 100 == 0:
            acc = evaluate(model, phase1_max, vocab_size, device, n_eval=100)
            print(f"  Step {step+1}: {100*acc:.1f}%")
    
    for num_pairs in results['phase1'].keys():
        acc = evaluate(model, num_pairs, vocab_size, device)
        results['phase2'][num_pairs] = acc
        results['final'][num_pairs] = acc
    
    phase2_acc = results['phase2'][phase1_max]
    
    if verbose:
        print(f"\n  Phase 1 @ {phase1_max}: {100*phase1_acc:.1f}%")
        print(f"  Phase 2 @ {phase1_max}: {100*phase2_acc:.1f}%")
        print(f"  Change: {100*(phase2_acc - phase1_acc):+.1f}%")
    
    return model, results


def train_joint_baseline(
    config: ANAConfig,
    device: str,
    curriculum: List[Tuple[int, int]] = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """Joint training baseline (same curriculum)."""
    
    model = ANAModel(config).to(device)
    
    if curriculum is None:
        curriculum = [(1, 200), (2, 300), (4, 400), (8, 500), (12, 600), (16, 700), (24, 800)]
    
    results = {'results': {}}
    vocab_size = config.vocab_size
    scaler = GradScaler('cuda') if device == "cuda" else None
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    if verbose:
        print(f"\n--- JOINT TRAINING BASELINE ---")
    
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size, device=device)
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=device == "cuda"):
                logits, _ = model(x)
                loss = F.cross_entropy(logits[:, -1, :].float(), y)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        acc = evaluate(model, num_pairs, vocab_size, device)
        results['results'][num_pairs] = acc
        
        if verbose:
            status = '✅' if acc > 0.85 else ('⚠️' if acc > 0.6 else '❌')
            print(f"  {num_pairs} pairs: {100*acc:.1f}% {status}")
    
    return model, results


def train_hololink_only(
    config: ANAConfig,
    device: str,
    curriculum: List[Tuple[int, int]] = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """HoloLink-only ablation (no controller)."""
    
    no_ctl_config = ANAConfig(
        **{k: v for k, v in config.__dict__.items()},
    )
    no_ctl_config.use_controller = False
    
    model = ANAModel(no_ctl_config).to(device)
    
    if curriculum is None:
        curriculum = [(1, 200), (2, 300), (4, 400), (8, 500), (12, 600), (16, 700), (24, 800)]
    
    results = {'results': {}}
    vocab_size = config.vocab_size
    scaler = GradScaler('cuda') if device == "cuda" else None
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    if verbose:
        print(f"\n--- HOLOLINK ONLY (No Controller) ---")
    
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size, device=device)
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=device == "cuda"):
                logits, _ = model(x)
                loss = F.cross_entropy(logits[:, -1, :].float(), y)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        acc = evaluate(model, num_pairs, vocab_size, device)
        results['results'][num_pairs] = acc
        
        if verbose:
            status = '✅' if acc > 0.85 else ('⚠️' if acc > 0.6 else '❌')
            print(f"  {num_pairs} pairs: {100*acc:.1f}% {status}")
    
    return model, results


def run_full_experiment(verbose: bool = True) -> Dict:
    """Run complete KV scaling experiment."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    vocab_size = 300
    
    config = ANAConfig(
        vocab_size=vocab_size,
        d_model=64,
        state_dim=64,
        key_dim=64,
        num_layers=1,
        track_count=1,
        use_hololink=True,
        use_controller=True,
        use_parallel_scan=True,
    )
    
    curriculum = [(1, 200), (2, 300), (4, 400), (8, 500), (12, 600), (16, 700), (24, 800)]
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: Two-Phase Training")
    print("="*70)
    model_tp, results_tp = train_two_phase_curriculum(
        config, device, curriculum, phase2_steps=500, verbose=verbose
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Joint Training Baseline")
    print("="*70)
    model_joint, results_joint = train_joint_baseline(
        config, device, curriculum, verbose=verbose
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: HoloLink-Only Ablation")
    print("="*70)
    model_holo, results_holo = train_hololink_only(
        config, device, curriculum, verbose=verbose
    )
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'KV Pairs':<10} {'Two-Phase':>12} {'Joint':>12} {'HoloLink':>12}")
    print("-"*50)
    
    for num_pairs in curriculum:
        n = num_pairs[0]
        tp_acc = results_tp['final'].get(n, 0)
        jt_acc = results_joint['results'].get(n, 0)
        hl_acc = results_holo['results'].get(n, 0)
        print(f"{n:<10} {100*tp_acc:>11.1f}% {100*jt_acc:>11.1f}% {100*hl_acc:>11.1f}%")
    
    max_pairs = curriculum[-1][0]
    tp_final = results_tp['final'].get(max_pairs, 0)
    jt_final = results_joint['results'].get(max_pairs, 0)
    hl_final = results_holo['results'].get(max_pairs, 0)
    
    print(f"\n{'At ' + str(max_pairs) + ' pairs:':<20}")
    print(f"  Two-Phase: {100*tp_final:.1f}%")
    print(f"  Joint:     {100*jt_final:.1f}%")
    print(f"  HoloLink:  {100*hl_final:.1f}%")
    
    synergy = tp_final - hl_final
    improvement = tp_final - jt_final
    
    print(f"\n{'Synergy (vs HoloLink-only):':<35} {100*synergy:+.1f}%")
    print(f"{'Improvement (vs Joint):':<35} {100*improvement:+.1f}%")
    
    if synergy > 0.05:
        print("\n✅ SUCCESS: Controller provides meaningful improvement!")
    elif tp_final > jt_final + 0.1:
        print("\n⚠️ PARTIAL: Two-phase beats joint but controller synergy unclear")
    else:
        print("\n❌ FAILED: Need to investigate")
    
    return {
        'two_phase': results_tp,
        'joint': results_joint,
        'hololink_only': results_holo,
        'synergy': synergy,
        'improvement': improvement
    }


if __name__ == "__main__":
    results = run_full_experiment(verbose=True)
