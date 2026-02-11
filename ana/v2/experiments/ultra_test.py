#!/usr/bin/env python3
"""Ultra-minimal test - should complete in <10 seconds."""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn as nn
import time

print("Ultra-minimal test...")

# Direct test of components without training
from ana.v2.core import ANAConfig, ANAModel

start = time.time()

# Tiny config
config = ANAConfig(
    d_model=16, vocab_size=5,
    track_dims=(4, 8, 4), stack_depth=2,
    stack_dim=8, num_layers=1
)

# Create model
model = ANAModel(config)
print(f"Model created: {sum(p.numel() for p in model.parameters()):,} params ({time.time()-start:.2f}s)")

# Forward pass
input_ids = torch.randint(0, 5, (2, 3))
logits = model(input_ids)
print(f"Forward pass: {logits.shape} ({time.time()-start:.2f}s)")

# Training step
targets = torch.randint(0, 5, (2, 3))
loss = nn.functional.cross_entropy(logits.view(-1, 5), targets.view(-1))
loss.backward()
print(f"Training step: loss={loss.item():.4f} ({time.time()-start:.2f}s)")

# 5 more steps
for i in range(5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, 5), targets.view(-1))
    loss.backward()
    optimizer.step()

print(f"5 more steps: {time.time()-start:.2f}s total")
print(f"\n✅ Training loop works and is fast!")
print(f"   Rate: {6/(time.time()-start):.1f} steps/second")
