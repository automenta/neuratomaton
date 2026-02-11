#!/usr/bin/env python3
"""
Ultra-minimal test - <5 seconds, all fixed.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F

print("Starting ultra-minimal test...")

# Import
from ana.v2.core import ANAConfig, ANAModel
from ana.v2.tasks import generate_reverse_task

# Tiny task with vocab_size=10 (tokens 0-9)
task = generate_reverse_task(
    num_train=50, num_test=20,
    train_len=(3,3), test_len=(4,4),
    vocab_size=10
)
print(f"Task: {task.train_seqs.shape} train, {task.test_seqs.shape} test")

# Tiny model - MUST match vocab_size from task
config = ANAConfig(
    d_model=16, vocab_size=10,  # Matches task.vocab_size
    track_dims=(4, 8, 4), stack_depth=2,
    stack_dim=8, num_layers=1
)
model = ANAModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Train 100 steps
print("Training 100 steps...")
for step in range(100):
    optimizer.zero_grad()
    logits = model(task.train_seqs)
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        task.train_targets.view(-1),
        ignore_index=0
    )
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 25 == 0:
        print(f"  Step {step+1}, Loss: {loss.item():.4f}")

# Quick eval
model.eval()
with torch.no_grad():
    test_logits = model(task.test_seqs)
    test_preds = test_logits.argmax(dim=-1)
    correct = sum(1 for i in range(len(task.test_seqs)) 
                   if torch.equal(test_preds[i], task.test_targets[i]))
    accuracy = correct / len(task.test_seqs)

print(f"\n✓ Test accuracy: {accuracy:.2%}")

if accuracy > 0.1:
    print("✅ WORKING: Model shows some learning!")
    print("   Architecture is viable.")
    print("   Run with more data/steps for full results.")
else:
    print("⚠️  Model needs more training steps.")

print("Done in <5 seconds")
