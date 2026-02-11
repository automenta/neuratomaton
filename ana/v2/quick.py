#!/usr/bin/env python3
"""Quick sanity check for ANA v2."""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
from torch.utils.data import DataLoader

from ana.v2.core import ANAConfig, ANAModel
from ana.v2.train import Trainer, SimpleDataset
from ana.v2.tasks import generate_reverse_task

print("ANA v2: Quick Sanity Check")
print("=" * 40)

# Tiny task
task = generate_reverse_task(num_train=50, num_test=20, 
                             train_len=(3,4), test_len=(5,6),
                             vocab_size=5)

print(f"Train: {task.train_seqs.shape}, Test: {task.test_seqs.shape}")

# Tiny model
config = ANAConfig(d_model=32, vocab_size=task.vocab_size,
                   track_dims=(8, 16, 8), stack_depth=2,
                   stack_dim=16, num_layers=1)

print(f"Model params: {sum(p.numel() for p in ANAModel(config).parameters()):,}")

# Quick train
dataset = SimpleDataset(task.train_seqs, task.train_targets)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
trainer = Trainer(config, lr=1e-3)

print("\nTraining 5 epochs...")
trainer.train(loader, num_epochs=5)

# Quick test
trainer.model.eval()
with torch.no_grad():
    test_x = task.test_seqs[:3].to(trainer.device)
    logits = trainer.model(test_x)
    preds = logits.argmax(dim=-1)
    
    print("\nSample predictions:")
    for i in range(3):
        inp = task.test_seqs[i].tolist()
        pred = preds[i].tolist()
        tgt = task.test_targets[i].tolist()
        print(f"  In: {inp[:5]}...")
        print(f"  Pred: {pred[:5]}...")
        print(f"  Target: {tgt[:5]}...")
        print()

print("=" * 40)
print("Sanity check complete!")
