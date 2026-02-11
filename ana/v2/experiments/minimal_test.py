#!/usr/bin/env python3
"""Minimal test to verify training works."""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F

from ana.v2.core import ANAConfig, ANAModel
from ana.v2.tasks import generate_reverse_task

print("Minimal training test...")

# Tiny data
task = generate_reverse_task(num_train=20, num_test=10, 
                             train_len=(3,3), test_len=(4,4),
                             vocab_size=5)

print(f"Train: {task.train_seqs.shape}, Test: {task.test_seqs.shape}")

# Tiny model
config = ANAConfig(d_model=16, vocab_size=task.vocab_size,
                   track_dims=(4, 8, 4), stack_depth=2,
                   stack_dim=8, num_layers=1)

model = ANAModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Simple training loop
print("Training 5 steps...")
for step in range(5):
    x = task.train_seqs
    targets = task.train_targets
    
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), 
                          targets.view(-1), ignore_index=0)
    loss.backward()
    optimizer.step()
    
    print(f"  Step {step}, Loss: {loss.item():.4f}")

# Quick eval
with torch.no_grad():
    test_logits = model(task.test_seqs)
    test_preds = test_logits.argmax(dim=-1)
    
    correct = 0
    for i in range(len(task.test_seqs)):
        if torch.equal(test_preds[i], task.test_targets[i]):
            correct += 1
    
    accuracy = correct / len(task.test_seqs)
    print(f"\nTest accuracy: {accuracy:.2%}")

print("Done!")
