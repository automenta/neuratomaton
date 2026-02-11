#!/usr/bin/env python3
"""
ANA v2: Quick demo - train on reverse task.

This demonstrates the core capability: learning an algorithm from examples.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
from torch.utils.data import DataLoader

from ana.v2.core import ANAConfig, ANAModel
from ana.v2.train import Trainer, SimpleDataset
from ana.v2.tasks import generate_reverse_task, evaluate_task

print("=" * 60)
print("ANA v2: Reverse Task Demo")
print("=" * 60)

# Generate task
print("\n1. Generating reverse task...")
task = generate_reverse_task(
    num_train=500,
    num_test=100,
    train_len=(3, 6),
    test_len=(7, 12),
    vocab_size=10
)
print(f"   Train samples: {task.train_seqs.shape}")
print(f"   Test samples: {task.test_seqs.shape}")
print(f"   Vocab size: {task.vocab_size}")

# Example
print(f"\n   Example:")
print(f"   Input:  {task.train_seqs[0].tolist()}")
print(f"   Target: {task.train_targets[0].tolist()}")

# Create model
print("\n2. Creating model...")
config = ANAConfig(
    d_model=64,
    vocab_size=task.vocab_size,
    track_dims=(16, 32, 16),
    stack_depth=4,
    stack_dim=32,
    num_layers=2
)
model = ANAModel(config)
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")

# Train
print("\n3. Training...")
dataset = SimpleDataset(task.train_seqs, task.train_targets)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

trainer = Trainer(config, lr=1e-3)

# Quick training
history = trainer.train(loader, num_epochs=20, eval_every=5)

# Evaluate
print("\n4. Evaluating on longer sequences (generalization test)...")
results = evaluate_task(trainer.model, task)

print(f"\n   Results:")
print(f"   Exact Match Accuracy: {results['exact_accuracy']:.2%}")
print(f"   Token Accuracy: {results['token_accuracy']:.2%}")

# Manual test
print("\n5. Manual test:")
trainer.model.eval()
with torch.no_grad():
    # Create a test sequence
    test_seq = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])  # Padded
    logits = trainer.model(test_seq)
    pred = logits.argmax(dim=-1)
    
    print(f"   Input:       {test_seq[0].tolist()}")
    print(f"   Predicted:   {pred[0].tolist()}")
    print(f"   Expected:    [5, 4, 3, 2, 1, 0, 0]")

print("\n" + "=" * 60)
if results['token_accuracy'] > 0.5:
    print("SUCCESS! Model is learning the reverse algorithm!")
else:
    print("Model needs more training or architecture tuning.")
print("=" * 60)
