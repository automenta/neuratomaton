#!/usr/bin/env python3
"""
ANA v2: Quick Generalization Test

Fast version of Phase 1 experiment to verify viability.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
from torch.utils.data import DataLoader

from ana.v2.core import ANAConfig, ANAModel
from ana.v2.train import Trainer, SimpleDataset
from ana.v2.tasks import generate_copy_task, generate_reverse_task


def quick_test():
    """Quick test of generalization capability."""
    print("=" * 60)
    print("ANA v2: Quick Generalization Test")
    print("=" * 60)
    
    # Tiny reverse task
    print("\n1. Training on Reverse task...")
    task = generate_reverse_task(
        num_train=200, num_test=50,
        train_len=(3, 4), test_len=(5, 8),
        vocab_size=5
    )
    
    print(f"   Train: {task.train_seqs.shape}, Test: {task.test_seqs.shape}")
    
    # Tiny model
    config = ANAConfig(
        d_model=32, vocab_size=task.vocab_size,
        track_dims=(8, 16, 8), stack_depth=2,
        stack_dim=16, num_layers=1
    )
    
    print(f"   Model params: {sum(p.numel() for p in ANAModel(config).parameters()):,}")
    
    # Quick train
    dataset = SimpleDataset(task.train_seqs, task.train_targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    trainer = Trainer(config, lr=1e-3)
    
    print("\n2. Training for 10 epochs...")
    trainer.train(loader, num_epochs=10)
    
    # Evaluate generalization
    print("\n3. Evaluating generalization...")
    trainer.model.eval()
    
    results = {}
    for seq_len in [5, 6, 7, 8]:
        filtered_seqs = []
        filtered_targets = []
        for i, seq in enumerate(task.test_seqs):
            if (seq != 0).sum().item() == seq_len:
                filtered_seqs.append(seq)
                filtered_targets.append(task.test_targets[i])
        
        if len(filtered_seqs) > 0:
            filtered_seqs = torch.stack(filtered_seqs)
            filtered_targets = torch.stack(filtered_targets)
            
            with torch.no_grad():
                filtered_seqs = filtered_seqs.to(trainer.device)
                logits = trainer.model(filtered_seqs)
                preds = logits.argmax(dim=-1).cpu()
                
                exact = sum(1 for i in range(len(filtered_seqs)) 
                            if torch.equal(preds[i], filtered_targets[i])) / len(filtered_seqs)
                
                correct = sum(((preds[i] == filtered_targets[i]) & (filtered_targets[i] != 0)).sum().item()
                             for i in range(len(filtered_seqs)))
                total = sum((filtered_targets[i] != 0).sum().item() for i in range(len(filtered_seqs)))
                token = correct / total if total > 0 else 0
                
                results[seq_len] = {'exact': exact, 'token': token, 'count': len(filtered_seqs)}
    
    # Print results
    print("\n4. Results by test length:")
    print("   Length | Exact | Token | Count")
    print("   -------|-------|-------|------")
    for length, metrics in sorted(results.items()):
        ratio = length / 4  # Max train length
        print(f"   {length:5}   | {metrics['exact']:5.1%} | {metrics['token']:5.1%} | {metrics['count']:5}")
        print(f"          (ratio={ratio:.1f}x train)")
    
    # Check viability
    print("\n5. Viability Check:")
    if results[5]['exact'] > 0.5:
        print(f"   ✅ Some generalization (length 5: {results[5]['exact']:.1%})")
    else:
        print(f"   ❌ No generalization (length 5: {results[5]['exact']:.1%})")
    
    if results[8]['exact'] > 0.3:
        print(f"   ✅ Strong generalization (length 8: {results[8]['exact']:.1%})")
    else:
        print(f"   ⚠️  Weak generalization (length 8: {results[8]['exact']:.1%})")
    
    print("\n" + "=" * 60)
    if results[5]['exact'] > 0.4:
        print("VIABLE: Architecture shows generalization capability.")
        print("Proceed with full Phase 1 experiments.")
    else:
        print("NEEDS WORK: Architecture not generalizing well.")
        print("Consider: increase model size, more training, or fix bugs.")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = quick_test()
