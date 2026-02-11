#!/usr/bin/env python3
"""Fast test with minimal settings."""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F

print("Fast test starting...")

from ana.v2.core import ANAConfig, ANAModel
from ana.v2.tasks import generate_reverse_task

# Tiny task
task = generate_reverse_task(
    num_train=100, num_test=50,
    train_len=(3,3), test_len=(4,4),
    vocab_size=5
)
print(f"Task: Train {task.train_seqs.shape}, Test {task.test_seqs.shape}")

# Medium model
config = ANAConfig(
    d_model=32, vocab_size=task.vocab_size,
    track_dims=(8, 16, 8), stack_depth=2,
    stack_dim=16, num_layers=1
)
model = ANAModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Fast training
print("Training 20 epochs...")
for epoch in range(20):
    optimizer.zero_grad()
    logits = model(task.train_seqs)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), 
                          task.train_targets.view(-1), ignore_index=0)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Eval
model.eval()
with torch.no_grad():
    test_logits = model(task.test_seqs)
    test_preds = test_logits.argmax(dim=-1)
    correct = sum(1 for i in range(len(task.test_seqs)) 
                   if torch.equal(test_preds[i], task.test_targets[i]))
    acc = correct / len(task.test_seqs)

print(f"\nTest accuracy: {acc:.2%}")

if acc > 0.1:
    print("✅ Some learning occurring!")
else:
    print("❌ Not learning - may need more epochs or larger model")
