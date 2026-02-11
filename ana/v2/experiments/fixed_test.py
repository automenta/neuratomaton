#!/usr/bin/env python3
"""Fixed test with longer training and larger model."""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F
import random

print("="*60)
print("ANA v2: Fixed Test")
print("="*60)

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Import
from ana.v2.core import ANAConfig, ANAModel
from ana.v2.tasks import generate_reverse_task

# Generate task
print("\n1. Generating reverse task...")
task = generate_reverse_task(
    num_train=500,  # More data
    num_test=100,
    train_len=(3, 5),  # Train on 3-5
    test_len=(6, 8),   # Test on 6-8 (2× max train)
    vocab_size=10
)
print(f"   Train: {task.train_seqs.shape}")
print(f"   Test: {task.test_seqs.shape}")

# Larger model
print("\n2. Creating larger model...")
config = ANAConfig(
    d_model=64,  # Bigger
    vocab_size=task.vocab_size,
    track_dims=(16, 32, 16),  # Bigger tracks
    stack_depth=4,  # Deeper stack
    stack_dim=32,  # Bigger stack
    num_layers=2  # 2 layers
)
model = ANAModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
params = sum(p.numel() for p in model.parameters())
print(f"   Model params: {params:,}")

# Show example
print("\n3. Example data:")
idx = 0
print(f"   Input:  {task.train_seqs[idx].tolist()}")
print(f"   Target: {task.train_targets[idx].tolist()}")

# Training
print("\n4. Training...")
best_loss = float('inf')
for epoch in range(100):
    # Train on all data
    optimizer.zero_grad()
    logits = model(task.train_seqs)
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        task.train_targets.view(-1),
        ignore_index=0
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if loss.item() < best_loss:
        best_loss = loss.item()
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}/100, Loss: {loss.item():.4f} (best: {best_loss:.4f})")

# Evaluate
print("\n5. Evaluating...")
model.eval()

with torch.no_grad():
    # Train set
    train_logits = model(task.train_seqs)
    train_preds = train_logits.argmax(dim=-1)
    
    train_correct = sum(1 for i in range(len(task.train_seqs)) 
                        if torch.equal(train_preds[i], task.train_targets[i]))
    train_acc = train_correct / len(task.train_seqs)
    
    # Test set
    test_logits = model(task.test_seqs)
    test_preds = test_logits.argmax(dim=-1)
    
    test_correct = sum(1 for i in range(len(task.test_seqs)) 
                       if torch.equal(test_preds[i], task.test_targets[i]))
    test_acc = test_correct / len(task.test_seqs)

print(f"\n   Train accuracy: {train_acc:.2%}")
print(f"   Test accuracy:  {test_acc:.2%}")

# Check generalization by length
print("\n6. Generalization by test length:")
for length in [6, 7, 8]:
    filtered = []
    filtered_targets = []
    for i, seq in enumerate(task.test_seqs):
        if (seq != 0).sum().item() == length:
            filtered.append(seq)
            filtered_targets.append(task.test_targets[i])
    
    if filtered:
        filtered = torch.stack(filtered)
        filtered_targets = torch.stack(filtered_targets)
        
        with torch.no_grad():
            logits = model(filtered)
            preds = logits.argmax(dim=-1)
            correct = sum(1 for i in range(len(filtered)) 
                         if torch.equal(preds[i], filtered_targets[i]))
            acc = correct / len(filtered)
        
        ratio = length / 5  # Max train length
        print(f"   Length {length} ({ratio:.1f}x train): {acc:.2%} ({len(filtered)} samples)")

# Show some predictions
print("\n7. Sample predictions:")
model.eval()
with torch.no_grad():
    sample_logits = model(task.test_seqs[:5])
    sample_preds = sample_logits.argmax(dim=-1)
    
    for i in range(5):
        inp = task.test_seqs[i].tolist()
        pred = sample_preds[i].tolist()
        tgt = task.test_targets[i].tolist()
        correct = "✓" if torch.equal(sample_preds[i], task.test_targets[i]) else "✗"
        print(f"   {correct} In: {inp[:5]}... | Pred: {pred[:5]}... | Tgt: {tgt[:5]}...")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Model params: {params:,}")
print(f"Train loss:   {best_loss:.4f}")
print(f"Train accuracy: {train_acc:.2%}")
print(f"Test accuracy:  {test_acc:.2%}")

if train_acc > 0.8:
    print("\n✅ MODEL LEARNING: Train accuracy >80%")
    if test_acc > 0.5:
        print("✅ GENERALIZATION: Test accuracy >50%")
        print("STATUS: Ready for Phase 1 generalization experiments")
    else:
        print("⚠️  PARTIAL: Learning but not generalizing well")
        print("STATUS: Try longer training or larger model")
else:
    print("\n❌ NOT LEARNING: Train accuracy <80%")
    print("STATUS: Architecture issue - needs debugging")
    
print("="*60)
