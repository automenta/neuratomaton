#!/usr/bin/env python3
# Experiment 5: Sequence Extrapolation
import torch
from ana.config import ANAConfig, TrainingConfig, DataConfig
from ana.models import ANAModel
from ana.train import run_training, evaluate
import json

print('='*70)
print('SEQUENCE EXTRAPOLATION TEST')
print('='*70)
print()

# Train on seq_len=64
config = ANAConfig(
    d_model=64,
    state_dim=64,
    num_layers=2,
    track_count=2,
    vocab_size=50,
)

train_config = TrainingConfig(
    batch_size=16,
    epochs=10,
    stage='2a',
    save_checkpoints=True,  # Need to save for extrapolation test
)

data_config = DataConfig(
    vocab_size=50,
    min_noise=10,
    max_noise=50,
    dataset_size=1000,
)

print('Training on seq_len=64...')
history = run_training(config, train_config, data_config, model_type='ana', num_workers=0)
print(f'Training complete. Final loss: {history["loss"][-1]:.4f}')
print()

# Load trained model
model = ANAModel(config).to(torch.device('cuda'))
model.load_state_dict(torch.load('archive/results/model_stage2a_ana.pt'))
model.eval()

# Test on longer sequences
from ana.data import AssociativeRecallDataset
from torch.utils.data import DataLoader
from ana.train import col_fn
import torch.nn as nn

test_seq_lengths = [64, 128, 256, 512]
results = {}

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
device = torch.device('cuda')

for seq_len in test_seq_lengths:
    print(f'Testing on seq_len={seq_len}...')

    # Adjust noise range for longer sequences
    test_min_noise = min(10, seq_len // 4)
    test_max_noise = min(50, seq_len // 2)

    test_dataset = AssociativeRecallDataset(
        size=500,
        vocab_size=50,
        min_noise=test_min_noise,
        max_noise=test_max_noise,
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=col_fn, num_workers=0)

    total_loss = 0
    total_correct = 0
    total_samples = 0
    needle_correct = 0
    needle_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            logits, _ = model(x)

            loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss_raw = loss_raw.view(y.size())
            loss = (loss_raw * mask).sum() / mask.sum()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            last_pred = preds[:, -1]
            last_target = y[:, -1]
            needle_correct += (last_pred == last_target).float().sum().item()
            needle_samples += x.size(0)

            total_correct += (preds == y).float().sum().item()
            total_samples += y.numel()

    avg_loss = total_loss / len(test_loader)
    acc = total_correct / total_samples
    needle_acc = needle_correct / needle_samples

    results[seq_len] = {
        'loss': avg_loss,
        'acc': acc,
        'needle_acc': needle_acc,
    }

    print(f'  Loss: {avg_loss:.4f}, Acc: {acc:.2%}, Needle Acc: {needle_acc:.2%}')

print()
print('='*70)
print('EXTRAPOLATION RESULTS')
print('='*70)
print('Seq Len   Loss       Acc        Needle Acc')
print('-'*60)
for seq_len, res in results.items():
    print(f'{seq_len:>8} {res["loss"]:>10.4f} {res["acc"]:>10.2%} {res["needle_acc"]:>12.2%}')

with open('archive/phase5_extrapolation.json', 'w') as f:
    json.dump(results, f, indent=2)
print()
print('Results saved to archive/phase5_extrapolation.json')
