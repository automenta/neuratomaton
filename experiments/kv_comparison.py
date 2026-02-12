"""
KV Scaling Experiment - Final Version

Compares Two-Phase Training vs Joint Training vs HoloLink-Only.
Demonstrates the key claim: Two-phase training beats joint training.
"""

import sys
sys.path.insert(0, '/home/me/ana')

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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 60
    
    print('='*60)
    print('KV SCALING EXPERIMENT: Two-Phase vs Joint Training')
    print('='*60)
    print(f'Device: {device}')
    
    curriculum = [(1, 800), (2, 800), (4, 800), (6, 800), (8, 800), (10, 800), (12, 1000)]
    
    # ============ TWO-PHASE TRAINING ============
    print('\n--- TWO-PHASE TRAINING ---')
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
    
    model_tp = ANAModel(config).to(device)
    holo_params, ctl_params, other_params = get_component_params(model_tp)
    
    # Phase 1
    for p in ctl_params:
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(list(holo_params) + other_params, lr=1e-3)
    
    tp_results = {}
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model_tp(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model_tp, num_pairs, vocab_size, device)
        tp_results[num_pairs] = acc
        status = '✅' if acc > 0.9 else ('⚠️' if acc > 0.7 else '❌')
        print(f'  Phase 1 @ {num_pairs} pairs: {100*acc:.1f}% {status}')
    
    phase1_acc = tp_results[12]
    
    # Phase 2
    for p in ctl_params:
        p.requires_grad = True
    for p in holo_params:
        p.requires_grad = False
    
    optimizer_ctl = torch.optim.Adam(ctl_params, lr=1e-4)
    
    for step in range(500):
        x, y = generate_kv_task(32, 12, vocab_size)
        x, y = x.to(device), y.to(device)
        optimizer_ctl.zero_grad()
        logits, _ = model_tp(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        loss.backward()
        optimizer_ctl.step()
    
    phase2_acc = evaluate(model_tp, 12, vocab_size, device)
    tp_results[12] = phase2_acc
    print(f'  Phase 2 Final: {100*phase2_acc:.1f}%')
    
    # ============ JOINT TRAINING ============
    print('\n--- JOINT TRAINING (Baseline) ---')
    model_joint = ANAModel(config).to(device)
    optimizer = torch.optim.Adam(model_joint.parameters(), lr=1e-3)
    
    joint_results = {}
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model_joint(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model_joint, num_pairs, vocab_size, device)
        joint_results[num_pairs] = acc
        status = '✅' if acc > 0.9 else ('⚠️' if acc > 0.7 else '❌')
        print(f'  @ {num_pairs} pairs: {100*acc:.1f}% {status}')
    
    joint_acc = joint_results[12]
    
    # ============ HOLOLINK-ONLY ============
    print('\n--- HOLOLINK ONLY (No Controller) ---')
    config_no_ctl = ANAConfig(
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
    
    model_holo = ANAModel(config_no_ctl).to(device)
    optimizer = torch.optim.Adam(model_holo.parameters(), lr=1e-3)
    
    holo_results = {}
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model_holo(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model_holo, num_pairs, vocab_size, device)
        holo_results[num_pairs] = acc
        status = '✅' if acc > 0.9 else ('⚠️' if acc > 0.7 else '❌')
        print(f'  @ {num_pairs} pairs: {100*acc:.1f}% {status}')
    
    holo_acc = holo_results[12]
    
    # ============ RESULTS ============
    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    
    print('\nKV Pairs     Two-Phase      Joint    HoloLink')
    print('-'*50)
    for n in [1, 2, 4, 6, 8, 10, 12]:
        print(f'{n:<12} {100*tp_results[n]:>10.1f}% {100*joint_results[n]:>10.1f}% {100*holo_results[n]:>10.1f}%')
    
    print('\nAt 12 pairs:')
    print(f'  Two-Phase: {100*phase2_acc:.1f}%')
    print(f'  Joint:     {100*joint_acc:.1f}%')
    print(f'  HoloLink:  {100*holo_acc:.1f}%')
    
    synergy = phase2_acc - holo_acc
    improvement = phase2_acc - joint_acc
    
    print(f'\nSynergy (vs HoloLink-only):   {100*synergy:+.1f}%')
    print(f'Improvement (vs Joint):       {100*improvement:+.1f}%')
    
    if improvement > 0.1:
        print('\n✅ SUCCESS: Two-phase training beats joint by >10%!')
    elif improvement > 0:
        print('\n⚠️ Two-phase beats joint but margin is small')
    else:
        print('\n❌ Need investigation')
    
    return {
        'two_phase': tp_results,
        'joint': joint_results,
        'hololink': holo_results,
        'synergy': synergy,
        'improvement': improvement
    }


if __name__ == "__main__":
    results = main()
