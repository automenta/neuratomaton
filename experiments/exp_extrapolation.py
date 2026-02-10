#!/usr/bin/env python3
"""
Phase D: Extrapolation Test (v3 - properly fixed)
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import random

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM

class KVDataset(data.Dataset):
    """Single KV pair: KEY K VAL V ... noise ... QUERY K [target=V]"""
    def __init__(self, size=500, min_noise=5, max_noise=20, vocab_size=30):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, vocab_size))
        for _ in range(size):
            k, v = random.choice(content), random.choice(content)
            noise_len = random.randint(min_noise, max_noise)
            noise = [random.choice(content) for _ in range(noise_len)]
            # Input: KEY K VAL V ... noise ... QUERY K
            # Target: (shifted) K VAL V ... noise ... QUERY K [V]
            inp = [TOK_KEY, k, TOK_VAL, v] + noise + [TOK_QUERY, k]
            tgt = [k, TOK_VAL, v] + noise + [TOK_QUERY, k, v]
            
            x = torch.tensor(inp, dtype=torch.long)
            y = torch.tensor(tgt, dtype=torch.long)
            # Only final position matters (predicting V after QUERY K)
            m = torch.zeros_like(y, dtype=torch.float)
            m[-1] = 1.0
            self.data.append((x, y, m, v))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i][:3]

def collate(batch):
    xs, ys, ms = zip(*batch)
    ml = max(x.size(0) for x in xs)
    return (torch.stack([F.pad(x, (0, ml-x.size(0)), value=0) for x in xs]),
            torch.stack([F.pad(y, (0, ml-y.size(0)), value=0) for y in ys]),
            torch.stack([F.pad(m, (0, ml-m.size(0)), value=0) for m in ms]))

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=30, d_model=64, n_heads=4, n_layers=2, max_seq_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=d_model*4, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        h = self.embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = h + self.pos_embedding(positions)
        h = self.transformer(h)
        return self.output(h), {}

def train_model(model, epochs=20):
    ds = KVDataset(size=500, min_noise=10, max_noise=30)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss_raw = crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size())
            loss = (loss_raw * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, _ = model(x)
            for i in range(x.size(0)):
                positions = (m[i] > 0.5).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    pos = positions[0]
                    if pos < logits.size(1) and y[i, pos].item() != 0:
                        pred = logits[i, pos].argmax().item()
                        target = y[i, pos].item()
                        if pred == target:
                            correct += 1
                        total += 1
    return correct / total if total > 0 else 0

def evaluate_model(model, min_noise, max_noise):
    ds = KVDataset(size=300, min_noise=min_noise, max_noise=max_noise)
    loader = data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, _ = model(x)
            for i in range(x.size(0)):
                positions = (m[i] > 0.5).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    pos = positions[0]
                    if pos < logits.size(1) and y[i, pos].item() != 0:
                        pred = logits[i, pos].argmax().item()
                        target = y[i, pos].item()
                        if pred == target:
                            correct += 1
                        total += 1
    return correct / total if total > 0 else 0

print("="*70)
print("PHASE D: EXTRAPOLATION TEST (v3)")
print("="*70)

TRAIN_NOISE = (10, 30)
TEST_CONFIGS = [
    ('train', (10, 30)),
    ('2x', (30, 70)),
    ('4x', (70, 150)),
]

results = {'ana': {}, 'transformer': {}, 'baseline': {}}

print("\nTraining models...")

torch.manual_seed(42)
random.seed(42)
ana = ANAModel(ANAConfig(d_model=64, vocab_size=30, num_layers=2, max_seq_len=1024)).to(device)
train_acc = train_model(ana, epochs=25)
print(f"ANA train acc: {train_acc*100:.1f}%")

torch.manual_seed(42)
random.seed(42)
xformer = SimpleTransformer(vocab_size=30, d_model=64, n_heads=4, n_layers=2, max_seq_len=1024).to(device)
train_acc = train_model(xformer, epochs=25)
print(f"Transformer train acc: {train_acc*100:.1f}%")

torch.manual_seed(42)
random.seed(42)
baseline = BaselineSSM(ANAConfig(d_model=64, vocab_size=30, num_layers=2, max_seq_len=1024)).to(device)
train_acc = train_model(baseline, epochs=25)
print(f"Baseline train acc: {train_acc*100:.1f}%")

print("\nTesting extrapolation...")

for name, (min_n, max_n) in TEST_CONFIGS:
    print(f"\n  {name} (noise {min_n}-{max_n}):")
    results['ana'][name] = evaluate_model(ana, min_n, max_n)
    results['transformer'][name] = evaluate_model(xformer, min_n, max_n)
    results['baseline'][name] = evaluate_model(baseline, min_n, max_n)
    print(f"    ANA={results['ana'][name]*100:.1f}%, XF={results['transformer'][name]*100:.1f}%, Base={results['baseline'][name]*100:.1f}%")

with open('archive/experiments/phaseD_extrapolation.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"{'Config':>8} | {'ANA':>8} | {'Transformer':>12} | {'Baseline':>10}")
print("-"*50)
for name, _ in TEST_CONFIGS:
    print(f"{name:>8} | {results['ana'][name]*100:>7.1f}% | {results['transformer'][name]*100:>11.1f}% | {results['baseline'][name]*100:>9.1f}%")
