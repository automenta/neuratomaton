"""
Two-Phase Training Experiment

Hypothesis: Training order matters for modular architectures.
- Phase 1: Train HoloLink (freeze Controller)
- Phase 2: Fine-tune Controller (freeze HoloLink)

Expected: 95%+ accuracy, Controller enhances HoloLink
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
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
    holo_params = []
    ctl_params = []
    other_params = []
    
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
    return correct / n_eval


def train_two_phase(config, device, verbose=True):
    """
    Two-phase training protocol:
    Phase 1: Train HoloLink (freeze Controller)
    Phase 2: Fine-tune Controller (freeze HoloLink)
    """
    model = ANAModel(config).to(device)
    vocab_size = config.vocab_size
    
    holo_params, ctl_params, other_params = get_component_params(model)
    
    if verbose:
        print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  HoloLink: {sum(p.numel() for p in holo_params):,}")
        print(f"  Controller: {sum(p.numel() for p in ctl_params):,}")
        print(f"  Other: {sum(p.numel() for p in other_params):,}")
    
    curriculum = [(1, 800), (2, 800), (4, 800), (6, 800), (8, 800), (10, 800), (12, 1000)]
    
    # ====================
    # PHASE 1: Train HoloLink (freeze controller)
    # ====================
    if verbose:
        print("\n" + "="*60)
        print("PHASE 1: Training HoloLink (Controller frozen)")
        print("="*60)
    
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
        
        acc = evaluate(model, num_pairs, vocab_size, device)
        if verbose:
            status = '✅' if acc > 0.9 else ('⚠️' if acc > 0.7 else '❌')
            print(f"  {num_pairs} pairs: {100*acc:.1f}% {status}")
    
    phase1_acc = evaluate(model, 12, vocab_size, device)
    if verbose:
        print(f"\n  Phase 1 Final: {100*phase1_acc:.1f}%")
    
    # ====================
    # PHASE 2: Fine-tune Controller (freeze HoloLink)
    # ====================
    if verbose:
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning Controller (HoloLink frozen)")
        print("="*60)
    
    for p in ctl_params:
        p.requires_grad = True
    for p in holo_params:
        p.requires_grad = False
    
    optimizer_ctl = torch.optim.Adam(ctl_params, lr=1e-4)
    
    for step in range(500):
        x, y = generate_kv_task(32, 12, vocab_size)
        x, y = x.to(device), y.to(device)
        
        optimizer_ctl.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        loss.backward()
        optimizer_ctl.step()
        
        if verbose and (step + 1) % 100 == 0:
            acc = evaluate(model, 12, vocab_size, device, n_eval=100)
            print(f"  Step {step+1}: {100*acc:.1f}%")
    
    phase2_acc = evaluate(model, 12, vocab_size, device)
    if verbose:
        print(f"\n  Phase 2 Final: {100*phase2_acc:.1f}%")
    
    return model, phase1_acc, phase2_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 60
    
    print("="*70)
    print("TWO-PHASE TRAINING EXPERIMENT")
    print("="*70)
    print(f"Device: {device}")
    print()
    
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
    
    model, phase1_acc, phase2_acc = train_two_phase(config, device)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Phase 1 (HoloLink only): {100*phase1_acc:.1f}%")
    print(f"  Phase 2 (+ Controller):  {100*phase2_acc:.1f}%")
    print(f"  Change: {'+' if phase2_acc >= phase1_acc else ''}{100*(phase2_acc - phase1_acc):.1f}%")
    
    if phase2_acc > 0.9:
        print("\n  ✅ SUCCESS: Two-phase training works!")
    elif phase2_acc > phase1_acc:
        print("\n  ⚠️ PARTIAL: Controller helps but below target")
    else:
        print("\n  ❌ FAILED: Controller still degrades performance")
    
    return phase2_acc


if __name__ == "__main__":
    main()