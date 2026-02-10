#!/usr/bin/env python3
"""
Large Model Investigation - Find the right training settings
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import random

sys.path.insert(0, '.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel

class QuickMultiKV(data.Dataset):
    def __init__(self, size=600, num_kv=8, min_noise=3, max_noise=10):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, 30))
        for _ in range(size):
            kvs = [(random.choice(content), random.choice(content)) for _ in range(num_kv)]
            seq = []
            for k, v in kvs:
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            seq.extend([random.choice(content) for _ in range(random.randint(min_noise, max_noise))])
            ti = random.randint(0, num_kv-1)
            seq.extend([TOK_QUERY, kvs[ti][0], kvs[ti][1]])
            x = torch.tensor(seq[:-1])
            y = torch.tensor(seq[1:])
            m = torch.ones_like(y, dtype=torch.float) * 0.01
            m[-1] = 1.0
            self.data.append((x, y, m))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    xs, ys, ms = zip(*batch)
    ml = max(x.size(0) for x in xs)
    return (torch.stack([F.pad(x, (0, ml-x.size(0))) for x in xs]),
            torch.stack([F.pad(y, (0, ml-y.size(0))) for y in ys]),
            torch.stack([F.pad(m, (0, ml-m.size(0))) for m in ms]))

def train_and_track(model, epochs=50, lr=1e-4):
    ds = QuickMultiKV(size=600, num_kv=8)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = (crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size()) * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, m in loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                logits, _ = model(x)
                for i in range(x.size(0)):
                    pos = (m[i] > 0.5).nonzero(as_tuple=True)[0][0]
                    if logits[i, pos].argmax().item() == y[i, pos].item():
                        correct += 1
                    total += 1
        acc = correct / total if total else 0
        history.append((epoch, total_loss/len(loader), acc))
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: loss={total_loss/len(loader):.4f}, acc={acc*100:.1f}%")
    
    return history

print("="*70)
print("LARGE MODEL INVESTIGATION")
print("="*70)

print("\nTesting different learning rates for large ANA model:")
for lr in [1e-4, 3e-4, 1e-3]:
    print(f"\nLR={lr}:")
    torch.manual_seed(42)
    random.seed(42)
    cfg = ANAConfig(d_model=256, num_layers=4, state_dim=256, vocab_size=30, max_seq_len=2048,
                    use_hololink=True, use_controller=True)
    model = ANAModel(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,}")
    history = train_and_track(model, epochs=40, lr=lr)
    final_acc = history[-1][2]
    best_acc = max(h[2] for h in history)
    print(f"  Final acc: {final_acc*100:.1f}%, Best acc: {best_acc*100:.1f}%")
