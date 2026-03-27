#!/usr/bin/env python
"""
Example: Text Generation with ANA

This script demonstrates how to initialize the ANA model and run a forward pass
simulating text generation.
"""

import torch
from ana import ANAConfig, ANAModel

def main():
    print("=== ANA Text Generation Example ===")

    # 1. Create Configuration
    # Using small parameters for demonstration
    config = ANAConfig(
        vocab_size=1000,
        d_model=64,
        state_dim=32,
        num_layers=2,
        use_hololink=True,
        track_count=2
    )
    print(f"Configuration: {config}")

    # 2. Initialize Model
    model = ANAModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")

    # 3. Create Dummy Input
    # Batch size 2, Sequence length 16
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"Input shape: {input_ids.shape}")

    # 4. Forward Pass
    logits, info = model(input_ids, return_info=True)

    print(f"Logits shape: {logits.shape}")
    print("Forward pass successful!")

    # 5. Check if info contains expected keys
    if 'layers' in info:
        print(f"Info log contains {len(info['layers'])} layers.")

if __name__ == "__main__":
    main()
