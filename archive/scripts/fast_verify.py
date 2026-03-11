#!/usr/bin/env python
"""
FAST VERIFICATION: Quick but reliable test of the breakthrough result.

Uses shorter curriculum but enough to demonstrate the effect.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

# ========== ANA Model ==========

class LRU(nn.Module):
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.in_proj = nn.Linear(d_model, state_dim)
        self.out_proj = nn.Linear(state_dim, d_model)
        self.alpha = nn.Parameter(torch.zeros(state_dim))
        self.beta = nn.Parameter(torch.zeros(state_dim))
    
    def forward(self, x):
        B, S, D = x.shape
        u = self.in_proj(x)
        a = torch.sigmoid(self.alpha).view(1, 1, -1)
        b = torch.sigmoid(self.beta).view(1, 1, -1)
        h = torch.zeros(B, self.state_dim, device=x.device)
        hs = []
        for t in range(S):
            h = a.squeeze(1) * h + b.squeeze(1) * u[:, t]
            hs.append(h)
        return self.out_proj(torch.stack(hs, dim=1)), torch.stack(hs, dim=1)

class HoloLink(nn.Module):
    def __init__(self, d_model, state_dim, key_dim):
        super().__init__()
        self.k_proj = nn.Linear(state_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(state_dim, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.bind = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, h):
        k = F.normalize(self.k_proj(h), p=2, dim=-1)
        v = self.v_proj(h)
        M = torch.cumsum(F.softplus(self.bind) * k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        return self.norm(self.out((q.unsqueeze(-2) @ M).squeeze(-2)))

class ANA(nn.Module):
    def __init__(self, vocab_size=60, d_model=64, state_dim=64, key_dim=32, n_layers=2, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({'lru': LRU(d_model, state_dim),
                          'holo': HoloLink(d_model, state_dim, key_dim)})
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            y, h = layer['lru'](x)
            x = x + y + layer['holo'](x, h)
        return self.head(self.norm(x))

# ========== Transformer ==========

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.hd), dim=-1)
        x = x + self.out((attn @ v).permute(0, 2, 1, 3).reshape(B, S, D))
        return x + self.ff(self.norm2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size=60, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

# ========== SINGLE SEED TEST ==========

def run_test(seed=42):
    print(f'\n=== Seed {seed} ===')
    set_seed(seed)
    
    # ANA
    print('Training ANA...')
    ana = ANA().to(device)
    ana_params = count_params(ana)
    opt = torch.optim.Adam(ana.parameters(), lr=1e-3)
    
    curriculum = [(p, 500) for p in [1, 2, 4, 6, 8, 10, 12]]
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            F.cross_entropy(ana(bx)[:, -1], by).backward()
            opt.step()
    
    ana_acc = eval_acc(ana, 12)
    print(f'ANA ({ana_params//1000}K params): {100*ana_acc:.1f}%')
    
    # Transformer
    print('Training Transformer...')
    set_seed(seed)
    trans = Transformer().to(device)
    trans_params = count_params(trans)
    opt = torch.optim.Adam(trans.parameters(), lr=1e-3)
    
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            F.cross_entropy(trans(bx)[:, -1], by).backward()
            opt.step()
    
    trans_acc = eval_acc(trans, 12)
    print(f'Transformer ({trans_params//1000000}M params): {100*trans_acc:.1f}%')
    
    eff = (ana_acc / (ana_params/1e6)) / (trans_acc / (trans_params/1e6))
    print(f'Efficiency ratio: {eff:.0f}x')
    
    del ana, trans
    torch.cuda.empty_cache()
    
    return ana_acc, trans_acc, eff

if __name__ == '__main__':
    print('='*60)
    print('FAST VERIFICATION: Breakthrough Result')
    print('='*60)
    
    # Run single test with seed 42
    ana_acc, trans_acc, eff = run_test(42)
    
    print('\n' + '='*60)
    print('RESULT')
    print('='*60)
    if ana_acc > 0.85 and trans_acc < 0.15:
        print(f'✅ VERIFIED: ANA ({100*ana_acc:.1f}%) >> Transformer ({100*trans_acc:.1f}%)')
        print(f'   Efficiency: {eff:.0f}x advantage')
    else:
        print('Needs investigation')
