#!/usr/bin/env python3
"""Super quick test with progress output."""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F

print("Step 1: Imports... ✓")

from ana.v2.core import ANAConfig, ANAModel
print("Step 2: Import core... ✓")

from ana.v2.tasks import generate_reverse_task
print("Step 3: Import tasks... ✓")

print("\nStep 4: Generating task...")
task = generate_reverse_task(num_train=50, num_test=20, 
                             train_len=(3,3), test_len=(4,4),
                             vocab_size=5)
print(f"Step 4: Task generated - Train: {task.train_seqs.shape}, Test: {task.test_seqs.shape}")

print("\nStep 5: Creating model...")
config = ANAConfig(d_model=16, vocab_size=task.vocab_size,
                   track_dims=(4, 8, 4), stack_depth=2,
                   stack_dim=8, num_layers=1)
model = ANAModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"Step 5: Model created - {sum(p.numel() for p in model.parameters()):,} params")

print("\nStep 6: Training...")
for step in range(10):
    x = task.train_seqs
    targets = task.train_targets
    
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), 
                          targets.view(-1), ignore_index=0)
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {step+1}/10, Loss: {loss.item():.4f}")

print("\nStep 7: Evaluation...")
with torch.no_grad():
    test_logits = model(task.test_seqs)
    test_preds = test_logits.argmax(dim=-1)
    
    correct = 0
    for i in range(len(task.test_seqs)):
        if torch.equal(test_preds[i], task.test_targets[i]):
            correct += 1
    
    accuracy = correct / len(task.test_seqs)
    print(f"Test accuracy: {accuracy:.2%}")

print("\n" + "="*50)
print("COMPLETE: Training loop works!")
print("="*50)
