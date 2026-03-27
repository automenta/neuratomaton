#!/usr/bin/env python
"""
Example: Training Text Generation with ANA

This script demonstrates how to train the ANA model on a simple text dataset
using the built-in Trainer class with checkpointing and metrics logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
import os
import shutil

from ana import ANAConfig, ANAModel
from ana.training.utils import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DummyTextDataset(Dataset):
    """
    Simple dataset that generates random sequences for demonstration.
    """
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Pre-generate data
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        # Target is input shifted by 1 (simple autoregressive task)
        # For random data, this is impossible to learn, but demonstrates the loop.
        self.targets = torch.roll(self.data, -1, dims=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def main():
    print("=== ANA Training Example ===")

    # 1. Configuration
    config = ANAConfig(
        vocab_size=100,
        d_model=64,
        state_dim=64,
        num_layers=2,
        track_count=2,
        batch_size=16,
        epochs=3,
        learning_rate=1e-3,
        use_hololink=True,
        use_controller=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Configuration: {config}")
    print(f"Device: {config.device}")

    # 2. Create Model
    model = ANAModel(config).to(config.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")

    # 3. Create Data
    print("Generating dummy data...")
    train_dataset = DummyTextDataset(config.vocab_size, seq_len=32, num_samples=1000)
    val_dataset = DummyTextDataset(config.vocab_size, seq_len=32, num_samples=100)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 4. Setup Trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    checkpoint_dir = "example_checkpoints"
    # Clean up previous run
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.device,
        checkpoint_dir=checkpoint_dir,
        log_interval=10
    )

    # 5. Train
    print("Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        val_every=1
    )

    print("\nTraining complete!")
    print(f"Checkpoints saved in {checkpoint_dir}")

    # 6. Load Best Model
    print("Loading best model for inference...")
    trainer.load_checkpoint("best_model.pt")

    # Simple Inference
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(config.device)
    with torch.no_grad():
        logits, _ = model(input_ids)
    print(f"Inference logits shape: {logits.shape}")

if __name__ == "__main__":
    main()
