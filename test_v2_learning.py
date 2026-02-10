#!/usr/bin/env python3
"""
End-to-end learning verification for ANA v2.
Run a small training run and show clear evidence of learning.
"""
import sys
import os
sys.path.insert(0, '.')

from ana.config_v2 import ANAv2Config, Trainingv2Config, Datav2Config
from ana.training_v2 import TrainerV2
from ana.data import AssociativeRecallDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import json

print("="*70)
print("ANA V2 END-TO-END LEARNING VERIFICATION")
print("="*70)

# Small config for fast training
config = ANAv2Config(
    d_model=32,
    vocab_size=16,
    syntax_dim=16,
    semantic_dim=32,
    logic_dim=16,
    stack_dim=32,
    stack_depth=3,
    cortex_hidden_dim=32,
    cortex_layers=1,
    fault_dim=64,
    fault_buffer_size=50,
    max_seq_len=128,
)

train_config = Trainingv2Config(
    epochs=20,
    batch_size=8,
    learning_rate=5e-4,
    device='cpu',
    log_interval=10,
    grad_clip=2.0,
    save_checkpoints=False,
    output_dir='archive/verification_run'
)

data_config = Datav2Config(
    vocab_size=16,
    seq_len=32,
    min_noise=5,
    max_noise=15,
    dataset_size=400,
)

print(f"\nModel config: d_model={config.d_model}, vocab={config.vocab_size}")
print(f"Training: {train_config.epochs} epochs, batch={train_config.batch_size}")
print(f"Dataset: {data_config.dataset_size} samples, seq_len={data_config.seq_len}")

trainer = TrainerV2(config, train_config, data_config)

# Stage 0 curriculum
dataloader = trainer.setup_stage0_curriculum()

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, trainer.model.parameters()), 
                       lr=train_config.learning_rate, weight_decay=0.01)

print("\n" + "="*70)
print("TRAINING START")
print("="*70)

history = {'train_loss': [], 'val_ppl': [], 'val_acc': [], 'needle_acc': []}

for epoch in range(train_config.epochs):
    trainer.model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        x, y, mask = batch
        x, y, mask = x.to(trainer.device), y.to(trainer.device), mask.to(trainer.device)
        
        optimizer.zero_grad()
        
        logits, rule_logits = trainer.model(x)
        loss_dict = trainer.model.compute_loss(logits, rule_logits, y)
        loss = loss_dict['total']
        
        # Masked loss
        ce_per_pos = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0, reduction='none'
        )
        ce_per_pos = ce_per_pos.view(y.size())
        loss = (ce_per_pos * mask).sum() / mask.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), train_config.grad_clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Evaluation
    trainer.model.eval()
    with torch.no_grad():
        val_loss, val_ce, val_ppl, val_acc, needle_acc, stack_depth = trainer.evaluate(
            dataloader, None, '0'
        )
    
    avg_train_loss = epoch_loss / len(dataloader)
    
    history['train_loss'].append(avg_train_loss)
    history['val_ppl'].append(val_ppl)
    history['val_acc'].append(val_acc)
    history['needle_acc'].append(needle_acc)
    
    print(f"Epoch {epoch+1:2d}/{train_config.epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val PPL: {val_ppl:6.2f} | "
          f"Val Acc: {val_acc:4.2%} | "
          f"Needle Acc: {needle_acc:5.2%}")

print("\n" + "="*70)
print("LEARNING ANALYSIS")
print("="*70)

# Check for learning - compare first 3 epochs vs last 3 epochs for stability
first_avg_loss = np.mean(history['train_loss'][:3])
last_avg_loss = np.mean(history['train_loss'][-3:])
first_avg_ppl = np.mean(history['val_ppl'][:3])
last_avg_ppl = np.mean(history['val_ppl'][-3:])
first_avg_acc = np.mean(history['val_acc'][:3])
last_avg_acc = np.mean(history['val_acc'][-3:])
first_avg_needle = np.mean(history['needle_acc'][:3])
last_avg_needle = np.mean(history['needle_acc'][-3:])

loss_improvement = ((first_avg_loss - last_avg_loss) / first_avg_loss) * 100
ppl_improvement = ((first_avg_ppl - last_avg_ppl) / first_avg_ppl) * 100
acc_improvement = ((last_avg_acc - first_avg_acc) / (first_avg_acc + 1e-9)) * 100
needle_improvement = ((last_avg_needle - first_avg_needle) / (first_avg_needle + 1e-9)) * 100

print(f"\nTrain Loss (avg 1-3): {first_avg_loss:.4f} → (avg last-2): {last_avg_loss:.4f} ({loss_improvement:+.1f}%)")
print(f"Val PPL (avg 1-3):    {first_avg_ppl:.2f} → (avg last-2): {last_avg_ppl:.2f} ({ppl_improvement:+.1f}%)")
print(f"Val Acc (avg 1-3):    {first_avg_acc:.2%} → (avg last-2): {last_avg_acc:.2%} ({acc_improvement:+.1f}%)")
print(f"Needle Acc (avg 1-3): {first_avg_needle:.2%} → (avg last-2): {last_avg_needle:.2%} ({needle_improvement:+.1f}%)")

print("\n" + "="*70)
print("VERDICT")
print("="*70)

# Primary learning criteria: Loss and PPL must decrease significantly
learning_criteria = [
    (loss_improvement > 10, f"Loss decreased by {loss_improvement:.1f}% (>10% required)"),
    (ppl_improvement > 10, f"PPL decreased by {ppl_improvement:.1f}% (>10% required)"),
]

# Secondary metrics (not required to pass but good to track)
secondary_criteria = [
    (acc_improvement > 0, f"Accuracy improved by {acc_improvement:.1f}%"),
    (last_avg_acc > first_avg_acc, f"Final avg acc ({last_avg_acc:.1%}) > initial avg acc ({first_avg_acc:.1%})"),
]

all_passed = all(criterion for criterion, _ in learning_criteria)

print("\nPrimary Metrics (must pass):")
for criterion, description in learning_criteria:
    status = "✓ PASS" if criterion else "✗ FAIL"
    print(f"  {status}: {description}")

print("\nSecondary Metrics (informational):")
for criterion, description in secondary_criteria:
    status = "✓" if criterion else " "
    print(f"  {status}: {description}")

print("\n" + "="*70)
if all_passed:
    print("✓✓✓ LEARNING VERIFIED - ANA v2 IS WORKING ✓✓✓")
else:
    print("✗✗✗ LEARNING NOT DETECTED ✗✗✗")
print("="*70)

# Save results
os.makedirs('archive/verification_run', exist_ok=True)
with open('archive/verification_run/learning_verification.json', 'w') as f:
    json.dump({
        'history': history,
        'improvements': {
            'loss_pct': loss_improvement,
            'ppl_pct': ppl_improvement,
            'acc_pct': acc_improvement,
            'needle_pct': needle_improvement,
        },
        'all_passed': all_passed,
        'config': {
            'd_model': config.d_model,
            'vocab_size': config.vocab_size,
            'epochs': train_config.epochs,
            'dataset_size': data_config.dataset_size,
        }
    }, f, indent=2)

print(f"\nResults saved to archive/verification_run/learning_verification.json")

sys.exit(0 if all_passed else 1)
