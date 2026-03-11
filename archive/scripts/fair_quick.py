#!/usr/bin/env python
"""
DEFINITIVE FAIR COMPARISON: ANA vs Transformer

Tests ANA against Transformers at multiple parameter counts to answer:
Is HoloLink's advantage universal, or just because the Transformer was too large?

Based on previous run:
- ANA (58K): 79.0% ± 19.8%
- Transformer 32K: 6.4% ± 1.0%
- Transformer 64K: 8.3% ± 0.3%
- Transformer 128K: 7.6% ± 0.6%
- Transformer 256K: 7.5% ± 0.3%
- Transformer 512K: 7.9% ± 0.5%

ANA beats ALL Transformers at ALL sizes by 10x!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gen_kv(batch, pairs, vocab=60):
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    content = list(range(4, vocab))
    x, y = [], []
    for _ in range(batch):
        keys = random.sample(content, pairs)
        vals = random.sample([t for t in content if t not in keys], pairs)
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([TOK_KEY, k, TOK_VAL, v])
        seq.extend(random.choices(content, k=8))
        q = random.randint(0, pairs-1)
        seq.extend([TOK_QUERY, keys[q]])
        x.append(seq)
        y.append(vals[q])
    mx = max(len(s) for s in x)
    t = torch.zeros(batch, mx, dtype=torch.long)
    for i, s in enumerate(x):
        t[i, :len(s)] = torch.tensor(s)
    return t.to(device), torch.tensor(y).to(device)

def eval_acc(model, pairs, n=30):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            bx, by = gen_kv(32, pairs)
            logits = model(bx)
            if isinstance(logits, tuple):
                logits = logits[0]
            correct += (logits[:, -1].argmax(-1) == by).sum().item()
    model.train()
    return correct / (n * 32)

class ANALayer(nn.Module):
    def __init__(self, d_model, state_dim, key_dim):
        super().__init__()
        self.in_proj = nn.Linear(d_model, state_dim)
        self.out_proj = nn.Linear(state_dim, d_model)
        self.alpha = nn.Parameter(torch.zeros(state_dim))
        self.beta = nn.Parameter(torch.zeros(state_dim))
        self.k_proj = nn.Linear(state_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(state_dim, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.bind = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        B, S, D = x.shape
        u = self.in_proj(x)
        a = torch.sigmoid(self.alpha).view(1, 1, -1)
        b = torch.sigmoid(self.beta).view(1, 1, -1)
        h = torch.zeros(B, self.alpha.shape[0], device=x.device)
        hs = []
        for t in range(S):
            h = a.squeeze(1) * h + b.squeeze(1) * u[:, t]
            hs.append(h)
        h = torch.stack(hs, dim=1)
        y = self.out_proj(h)
        k = F.normalize(self.k_proj(h), p=2, dim=-1)
        v = self.v_proj(h)
        M = torch.cumsum(F.softplus(self.bind) * k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        retrieved = self.norm(self.out((q.unsqueeze(-2) @ M).squeeze(-2)))
        return x + y + retrieved

class ANA(nn.Module):
    def __init__(self, d_model=64, state_dim=64, key_dim=32, n_layers=2, vocab=60, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([ANALayer(d_model, state_dim, key_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    
    def forward(self, x):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.hd), dim=-1)
        x = x + self.out((attn @ v).permute(0, 2, 1, 3).reshape(B, S, D))
        return x + self.ff(self.norm2(x))

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, vocab=60, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_model(model, steps=400, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for pairs in [1, 2, 4, 6, 8, 10, 12]:
        for _ in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            F.cross_entropy(model(bx)[:, -1], by).backward()
            opt.step()
    return eval_acc(model, 12)

# Quick single-seed comparison
print('='*70)
print('QUICK COMPARISON: ANA vs Transformers at Multiple Sizes')
print('='*70)

set_seed(42)

# ANA
print('\nTraining ANA...')
ana = ANA().to(device)
ana_params = count_params(ana)
ana_acc = train_model(ana)
print(f'ANA ({ana_params:,} params): {100*ana_acc:.1f}%')
del ana
torch.cuda.empty_cache()

# Transformers at different sizes
configs = [
    ('32K', 32, 1, 1),    # d_model, n_heads, n_layers
    ('64K', 48, 2, 2),
    ('128K', 64, 2, 3),
    ('256K', 80, 4, 4),
    ('512K', 112, 4, 5),
    ('1M', 160, 8, 5),
    ('2M', 224, 8, 6),
    ('4M', 256, 8, 6),
]

print('\nTraining Transformers...')
results = []
for name, d_model, n_heads, n_layers in configs:
    set_seed(42)
    trans = Transformer(d_model, n_heads, n_layers, d_model*4).to(device)
    params = count_params(trans)
    acc = train_model(trans)
    results.append((name, params, acc))
    print(f'Transformer {name} ({params:,}): {100*acc:.1f}%')
    del trans
    torch.cuda.empty_cache()

print('\n' + '='*70)
print('VERDICT')
print('='*70)
print(f'\nANA (58K params): {100*ana_acc:.1f}%')
print('Transformers:')
for name, params, acc in results:
    print(f'  {name} ({params:,}): {100*acc:.1f}%')

best_trans = max(results, key=lambda x: x[2])
if ana_acc > best_trans[2]:
    print(f'\n✅ BREAKTHROUGH: ANA beats ALL Transformers!')
    print(f'   ANA: {100*ana_acc:.1f}%')
    print(f'   Best Transformer ({best_trans[0]}): {100*best_trans[2]:.1f}%')
    print(f'   Gap: {100*(ana_acc - best_trans[2]):.1f}%')
