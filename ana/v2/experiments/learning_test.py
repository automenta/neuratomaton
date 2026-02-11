#!/usr/bin/env python3
"""
ANA v2: Learning Test (Experiment 1.1)

Objective: Verify model can learn simple tasks with sufficient training.
Expected: Accuracy >50% after 1000 training steps.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from datetime import datetime

from ana.v2.core import ANAConfig, ANAModel
from ana.v2.tasks import generate_reverse_task

def main():
    print("="*60)
    print("ANA v2: Learning Test (Experiment 1.1)")
    print("="*60)
    
    # Generate task
    print("\n1. Generating reverse task...")
    task = generate_reverse_task(
        num_train=500, num_test=100,
        train_len=(3, 5), test_len=(6, 8),
        vocab_size=10
    )
    print(f"   Train: {task.train_seqs.shape}")
    print(f"   Test: {task.test_seqs.shape}")
    print(f"   Vocab size: {task.vocab_size}")
    
    # Create model
    print("\n2. Creating model...")
    config = ANAConfig(
        d_model=64, vocab_size=task.vocab_size,
        track_dims=(16, 32, 16), stack_depth=4,
        stack_dim=32, num_layers=2
    )
    model = ANAModel(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Train 1000 steps
    print("\n3. Training 1000 steps...")
    best_loss = float('inf')
    for step in range(1000):
        optimizer.zero_grad()
        logits = model(task.train_seqs)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            task.train_targets.view(-1),
            ignore_index=0
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (step + 1) % 100 == 0:
            print(f"   Step {step+1}/1000, Loss: {loss.item():.4f} (best: {best_loss:.4f})")
    
    # Evaluate
    print("\n4. Evaluating...")
    model.eval()
    
    with torch.no_grad():
        test_logits = model(task.test_seqs)
        test_preds = test_logits.argmax(dim=-1)
        
        # Exact match accuracy
        correct = sum(1 for i in range(len(task.test_seqs)) 
                       if torch.equal(test_preds[i], task.test_targets[i]))
        accuracy = correct / len(task.test_seqs)
        
        # Token accuracy
        correct_tokens = 0
        total_tokens = 0
        for i in range(len(task.test_seqs)):
            mask = task.test_targets[i] != 0
            correct_tokens += ((test_preds[i] == task.test_targets[i]) & mask).sum().item()
            total_tokens += mask.sum().item()
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    print(f"\n   Exact match accuracy: {accuracy:.2%}")
    print(f"   Token accuracy: {token_accuracy:.2%}")
    
    # Save results
    print("\n5. Saving results...")
    results = {
        "experiment": "exp1_1_learning",
        "date": datetime.now().isoformat(),
        "task": "reverse",
        "train_samples": 500,
        "test_samples": 100,
        "model_config": {
            "d_model": config.d_model,
            "vocab_size": config.vocab_size,
            "track_dims": config.track_dims,
            "num_layers": config.num_layers,
            "params": params
        },
        "training": {
            "steps": 1000,
            "learning_rate": 3e-4,
            "final_loss": loss.item(),
            "best_loss": best_loss
        },
        "results": {
            "test_accuracy": accuracy,
            "token_accuracy": token_accuracy
        },
        "status": "PASS" if accuracy > 0.5 else "FAIL",
        "notes": "Model shows learning capability" if accuracy > 0.5 else "Model needs more training or capacity"
    }
    
    Path("ana/v2/results").mkdir(parents=True, exist_ok=True)
    with open("ana/v2/results/exp1_1_learning.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model params: {params:,}")
    print(f"Training steps: 1000")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Token accuracy: {token_accuracy:.2%}")
    
    if accuracy > 0.5:
        print("\n✅ PASS: Model achieves >50% accuracy")
        print("Next: Run generalization test (Experiment 1.2)")
    elif accuracy > 0.2:
        print("\n⚠️  PARTIAL: Model learning but below target")
        print("Action: Try more training steps or larger model")
    else:
        print("\n❌ FAIL: Model not learning")
        print("Action: Debug architecture (stack, memory, tracks)")
    
    print("="*60)
    
    return results

if __name__ == "__main__":
    main()
