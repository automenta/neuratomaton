#!/usr/bin/env python3
"""
ANA v2: GENERALIZATION TEST - Can it learn the ALGORITHM or just patterns?

Tests if the model generalizes to sequences LONGER than training.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F
import numpy as np

print("="*70)
print("ANA v2: GENERALIZATION TEST - Algorithm Learning vs Pattern Matching")
print("="*70)

from ana.v2.core import ANAConfig, ANAModel

# Training data: SHORT sequences (3-5 tokens)
train_seqs = torch.tensor([
    [1, 2, 3, 0, 0, 0],
    [2, 3, 4, 0, 0, 0],
    [1, 3, 5, 0, 0, 0],
    [4, 5, 6, 0, 0, 0],
    [2, 4, 6, 0, 0, 0],
    [1, 2, 3, 4, 0, 0],
    [2, 3, 4, 5, 0, 0],
    [1, 3, 5, 7, 0, 0],
])

train_targets = torch.tensor([
    [3, 2, 1, 0, 0, 0],
    [4, 3, 2, 0, 0, 0],
    [5, 3, 1, 0, 0, 0],
    [6, 5, 4, 0, 0, 0],
    [6, 4, 2, 0, 0, 0],
    [4, 3, 2, 1, 0, 0],
    [5, 4, 3, 2, 0, 0],
    [7, 5, 3, 1, 0, 0],
])

vocab_size = 8

print(f"\n📚 TRAINING DATA (length 3-5):")
print(f"   Samples: {len(train_seqs)}")
print(f"   Max train length: 4")

# Create model
config = ANAConfig(
    d_model=32, vocab_size=vocab_size,
    track_dims=(8, 16, 8), stack_depth=3,
    stack_dim=16, num_layers=1
)
model = ANAModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

print(f"\n🧠 MODEL: {sum(p.numel() for p in model.parameters()):,} parameters")

# Train
print(f"\n🎓 TRAINING (100 steps):")
for step in range(100):
    optimizer.zero_grad()
    logits = model(train_seqs)
    loss = F.cross_entropy(logits.view(-1, vocab_size), train_targets.view(-1), ignore_index=0)
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 20 == 0:
        print(f"   Step {step+1:3}: Loss = {loss.item():.4f}")

# Test on LONGER sequences (6-8 tokens) - beyond training!
test_cases = [
    ([1, 2, 3, 4, 5, 6, 0, 0], [6, 5, 4, 3, 2, 1, 0, 0], "6 tokens (1.5× train)"),
    ([1, 2, 3, 4, 5, 6, 7, 0], [7, 6, 5, 4, 3, 2, 1, 0], "7 tokens (1.75× train)"),
    ([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1], "8 tokens (2× train)"),
]

print(f"\n🔬 GENERALIZATION TEST:")
print(f"{'Test Case':<30} {'Predicted':<25} {'Target':<25} {'Result':<10}")
print(f"{'-'*70}")

model.eval()
with torch.no_grad():
    for test_seq, target_seq, description in test_cases:
        test_tensor = torch.tensor([test_seq])
        target_tensor = torch.tensor([target_seq])
        
        logits = model(test_tensor)
        preds = logits.argmax(dim=-1)[0]
        
        # Show actual predicted tokens (up to non-zero)
        pred_list = preds.tolist()
        pred_clean = pred_list[:pred_list.index(0) if 0 in pred_list else len(pred_list)]
        
        target_list = target_tensor[0].tolist()
        target_clean = target_list[:target_list.index(0) if 0 in target_list else len(target_list)]
        
        # Check accuracy
        correct = (preds == target_tensor[0]).sum().item()
        total = (target_tensor[0] != 0).sum().item()
        accuracy = correct / total if total > 0 else 0
        
        result = "✅ PASS" if accuracy >= 0.8 else "⚠️  PARTIAL" if accuracy >= 0.5 else "❌ FAIL"
        
        print(f"{description:<30} {str(pred_clean):<25} {str(target_clean):<25} {result:<10}")

# Detailed analysis
print(f"\n📊 ANALYSIS:")
print(f"   If model generalizes, it learned the REVERSE ALGORITHM.")
print(f"   If model fails, it only memorized training patterns.")
print(f"   ")
print(f"   The BEAST uses:")
print(f"     • Stack to store sequence")
print(f"     • POP to retrieve in reverse order")
print(f"     • Dynamic α,β modulation for timing")
print(f"     • Holographic memory for binding")

# Success criteria
print(f"\n🎯 SUCCESS CRITERIA:")
print(f"   • 6 tokens (1.5×): >80% accuracy")
print(f"   • 7 tokens (1.75×): >70% accuracy")
print(f"   • 8 tokens (2×): >50% accuracy")
print(f"   ")
print(f"   If ANY of these pass: Algorithm learning works!")
print(f"   If ALL fail: Model only memorized patterns")

print(f"\n" + "="*70)
print(f"   This is THE test that proves the thesis.")
print(f"   Generalization beyond training = Algorithm learning")
print(f"   Pattern matching only = Limited generalization")
print(f"="*70)
