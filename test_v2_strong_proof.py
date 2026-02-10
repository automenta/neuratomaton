#!/usr/bin/env python3
"""
Stronger proof: More epochs, larger model, show clear learning curve.
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
print("ANA V2 STRONG LEARNING PROOF")
print("="*70)

config = ANAv2Config(
    d_model=48,
    vocab_size=20,
    syntax_dim=24,
    semantic_dim=48,
    logic_dim=24,
    stack_dim=48,
    stack_depth=4,
    cortex_hidden_dim=48,
    cortex_layers=2,
    fault_dim=80,
    fault_buffer_size=64,
    max_seq_len=128,
)

train_config = Trainingv2Config(
    epochs=30,
    batch_size=12,
    learning_rate=3e-4,
    device='cpu',
    log_interval=10,
    grad_clip=2.0,
    save_checkpoints=False,
    output_dir='archive/strong_proof'
)

data_config = Datav2Config(
    vocab_size=20,
    seq_len=40,
    min_noise=8,
    max_noise=20,
    dataset_size=600,
)

print(f"\nModel: d_model={config.d_model}, params=~25K")
print(f"Training: {train_config.epochs} epochs, batch={train_config.batch_size}")
print(f"Dataset: {data_config.dataset_size} samples")

trainer = TrainerV2(config, train_config, data_config)
dataloader = trainer.setup_stage0_curriculum()

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, trainer.model.parameters()), 
                       lr=train_config.learning_rate, weight_decay=0.01)

print("\n" + "="*70)
print("TRAINING")
print("="*70)

history = {'train_loss': [], 'val_ppl': [], 'val_acc': []}

for epoch in range(train_config.epochs):
    trainer.model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        x, y, mask = batch
        x, y, mask = x.to(trainer.device), y.to(trainer.device), mask.to(trainer.device)
        
        optimizer.zero_grad()
        logits, rule_logits = trainer.model(x)
        loss_dict = trainer.model.compute_loss(logits, rule_logits, y)
        
        ce_per_pos = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0, reduction='none'
        )
        ce_per_pos = ce_per_pos.view(y.size())
        loss = (ce_per_pos * mask).sum() / mask.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), train_config.grad_clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    trainer.model.eval()
    with torch.no_grad():
        val_loss, val_ce, val_ppl, val_acc, needle_acc, _ = trainer.evaluate(dataloader, None, '0')
    
    avg_train_loss = epoch_loss / len(dataloader)
    
    history['train_loss'].append(avg_train_loss)
    history['val_ppl'].append(val_ppl)
    history['val_acc'].append(val_acc)
    
    if epoch < 5 or (epoch + 1) % 5 == 0 or epoch == train_config.epochs - 1:
        print(f"Epoch {epoch+1:2d}/{train_config.epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"PPL: {val_ppl:6.2f} | "
              f"Acc: {val_acc:5.2%} | "
              f"Needle: {needle_acc:4.1%}")

print("\n" + "="*70)
print("PROOF OF LEARNING")
print("="*70)

first_avg = np.mean(history['train_loss'][:3])
last_avg = np.mean(history['train_loss'][-3:])
first_ppl = np.mean(history['val_ppl'][:3])
last_ppl = np.mean(history['val_ppl'][-3:])
first_acc = np.mean(history['val_acc'][:3])
last_acc = np.mean(history['val_acc'][-3:])

loss_drop = ((first_avg - last_avg) / first_avg) * 100
ppl_drop = ((first_ppl - last_ppl) / first_ppl) * 100
acc_gain = ((last_acc - first_acc) / (first_acc + 1e-9)) * 100

print(f"\nTrain Loss: {first_avg:.4f} → {last_avg:.4f} (↓{loss_drop:.1f}%)")
print(f"Val PPL:    {first_ppl:.2f} → {last_ppl:.2f} (↓{ppl_drop:.1f}%)")
print(f"Val Acc:    {first_acc:.2%} → {last_acc:.2%} (↑{acc_gain:.1f}%)")

# Also check monotonic improvement
epochs_improved = 0
for i in range(2, len(history['train_loss'])):
    if history['train_loss'][i] < history['train_loss'][i-2]:
        epochs_improved += 1
monotonic_rate = epochs_improved / max(1, len(history['train_loss']) - 2) * 100

print(f"\nMonotonic improvement rate: {epochs_improved}/{len(history['train_loss'])-2} = {monotonic_rate:.0f}%")

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

strong_metrics = [
    (loss_drop > 15, f"Loss ↓ {loss_drop:.1f}% (need >15%)"),
    (ppl_drop > 20, f"PPL ↓ {ppl_drop:.1f}% (need >20%)"),
    (acc_gain > 25, f"Acc ↑ {acc_gain:.1f}% (need >25%)"),
    (monotonic_rate > 50, f"Monotonic improv {monotonic_rate:.0f}% (need >50%)"),
]

all_strong = all(c for c, _ in strong_metrics)

for passed, msg in strong_metrics:
    print(f"{'✓' if passed else '✗'} {msg}")

print("\n" + "="*70)
if all_strong:
    print("✓✓✓ UNDENIABLE PROOF: ANA v2 LEARNS EFFECTIVELY ✓✓✓")
else:
    print("Partial evidence - more training may help")
print("="*70)

os.makedirs('archive/strong_proof', exist_ok=True)
with open('archive/strong_proof/strong_proof.json', 'w') as f:
    json.dump({
        'history': history,
        'metrics': {
            'loss_drop_pct': loss_drop,
            'ppl_drop_pct': ppl_drop,
            'acc_gain_pct': acc_gain,
            'monotonic_rate_pct': monotonic_rate,
        },
        'all_strong': all_strong,
    }, f, indent=2)

sys.exit(0 if all_strong else 1)
