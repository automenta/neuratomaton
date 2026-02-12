#!/usr/bin/env python
"""
ROBUST VERIFICATION: Stable training protocol with proper curriculum.

Key fixes:
1. Longer curriculum training per KV level
2. Learning rate scheduling
3. Gradient clipping
4. More evaluations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def eval_acc(model, pairs, n=50):
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

@dataclass
class ANAConfig:
    vocab_size: int = 60
    d_model: int = 64
    state_dim: int = 64
    key_dim: int = 32
    n_layers: int = 2
    max_seq_len: int = 128

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
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({'lru': LRU(config.d_model, config.state_dim),
                          'holo': HoloLink(config.d_model, config.state_dim, config.key_dim)})
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    
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
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq):
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

# ========== TRAINING ==========

def train_ana(model, verbose=False):
    """Train ANA with extended curriculum."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    
    curriculum = [(1, 600), (2, 600), (4, 800), (6, 800), (8, 1000), (10, 1000), (12, 1200)]
    
    for pairs, steps in curriculum:
        for step in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx)[:, -1], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        
        if verbose:
            acc = eval_acc(model, pairs)
            print(f'    {pairs} pairs: {100*acc:.1f}%')
    
    return eval_acc(model, 12)

def train_transformer(model, verbose=False):
    """Train Transformer with same curriculum."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    
    curriculum = [(1, 600), (2, 600), (4, 800), (6, 800), (8, 1000), (10, 1000), (12, 1200)]
    
    for pairs, steps in curriculum:
        for step in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx)[:, -1], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        
        if verbose:
            acc = eval_acc(model, pairs)
            print(f'    {pairs} pairs: {100*acc:.1f}%')
    
    return eval_acc(model, 12)

# ========== MAIN ==========

def main():
    print('\n' + '='*70)
    print('ROBUST VERIFICATION: Multi-Seed Test with Extended Training')
    print('='*70)
    
    seeds = [42, 123, 456]
    results = []
    
    for seed in seeds:
        print(f'\n--- Seed {seed} ---')
        set_seed(seed)
        
        # ANA
        print('  Training ANA...')
        ana = ANA(ANAConfig()).to(device)
        ana_params = count_params(ana)
        ana_acc = train_ana(ana, verbose=False)
        print(f'  ANA ({ana_params//1000}K): {100*ana_acc:.1f}%')
        
        # Transformer
        print('  Training Transformer...')
        set_seed(seed)
        trans = Transformer(60, 256, 8, 6, 1024, 128).to(device)
        trans_params = count_params(trans)
        trans_acc = train_transformer(trans, verbose=False)
        print(f'  Transformer ({trans_params//1000000}M): {100*trans_acc:.1f}%')
        
        eff = (ana_acc / (ana_params/1e6)) / (trans_acc / (trans_params/1e6))
        print(f'  Efficiency: {eff:.0f}x')
        
        results.append({
            'seed': seed,
            'ana_params': ana_params,
            'trans_params': trans_params,
            'ana_acc': ana_acc,
            'trans_acc': trans_acc,
            'efficiency': eff
        })
        
        del ana, trans
        torch.cuda.empty_cache()
    
    # Summary
    ana_accs = [r['ana_acc'] for r in results]
    trans_accs = [r['trans_acc'] for r in results]
    effs = [r['efficiency'] for r in results]
    
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'\nANA Accuracy: {100*np.mean(ana_accs):.1f}% ± {100*np.std(ana_accs):.1f}%')
    print(f'Transformer Accuracy: {100*np.mean(trans_accs):.1f}% ± {100*np.std(trans_accs):.1f}%')
    print(f'Efficiency Ratio: {np.mean(effs):.0f}x ± {np.std(effs):.0f}x')
    
    if np.mean(ana_accs) > 0.85 and np.mean(trans_accs) < 0.15:
        print('\n✅ VERIFIED: Robust results across seeds!')
    else:
        print('\n⚠️ Results need investigation.')

if __name__ == '__main__':
    main()
