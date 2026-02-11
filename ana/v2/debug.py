#!/usr/bin/env python3
"""
Debug the forward pass error.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
from ana.v2.core import ANAConfig, ANAModel

config = ANAConfig(
    d_model=64,
    vocab_size=20,
    track_dims=(16, 32, 16),
    stack_depth=3,
    stack_dim=32,
    num_layers=1
)

model = ANAModel(config)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

batch_size = 2
seq_len = 5
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print(f"Input shape: {input_ids.shape}")
print(f"d_model: {config.d_model}")
print(f"track_dims: {config.track_dims}")
print(f"stack_dim: {config.stack_dim}")
print(f"hologram_dim: {config.hologram_dim}")

try:
    logits = model(input_ids)
    print(f"Success! Logits shape: {logits.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
