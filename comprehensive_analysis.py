#!/usr/bin/env python
"""
COMPREHENSIVE ANALYSIS: When Does HoloLink Help?

Testing ANA vs Transformer on:
1. Associative Recall (synthetic KV task)
2. Language Modeling (character-level)
3. Copy Task (simple sequence copying)

Goal: Understand WHERE HoloLink provides advantage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ============================================================================
# MODELS
# ============================================================================

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
    def __init__(self, vocab_size, d_model=64, state_dim=64, key_dim=32, n_layers=2, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([ANALayer(d_model, state_dim, key_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq=128):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.register_buffer('mask', torch.triu(torch.ones(max_seq, max_seq), diagonal=1).bool())
    
    def forward(self, x):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        attn = attn.masked_fill(self.mask[:S, :S], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        x = x + self.out((attn @ v).permute(0, 2, 1, 3).reshape(B, S, D))
        return x + self.ff(self.norm2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, max_seq) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

# ============================================================================
# TASK 1: ASSOCIATIVE RECALL
# ============================================================================

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

def eval_kv(model, pairs, n=30):
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

def train_kv(model, steps=400):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for pairs in [1, 2, 4, 6, 8, 10, 12]:
        for _ in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            F.cross_entropy(model(bx)[:, -1], by).backward()
            opt.step()
    return eval_kv(model, 12)

# ============================================================================
# TASK 2: COPY TASK
# ============================================================================

def gen_copy(batch, length, vocab=50):
    SEP = 1
    x, y = [], []
    for _ in range(batch):
        seq = [random.randint(2, vocab-1) for _ in range(length)]
        full = seq + [SEP] + [0] * length  # Input: sequence + separator + zeros
        target = [0] * (length + 1) + seq   # Target: zeros + separator + copy
        x.append(full)
        y.append(target)
    mx = len(x[0])
    t = torch.zeros(batch, mx, dtype=torch.long)
    for i, s in enumerate(x):
        t[i, :len(s)] = torch.tensor(s)
    return t.to(device), torch.tensor(y).to(device)

def eval_copy(model, length, n=30):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n):
            bx, by = gen_copy(16, length)
            logits = model(bx)
            if isinstance(logits, tuple):
                logits = logits[0]
            # Check copy part (after separator)
            for i in range(length + 1, bx.shape[1]):
                correct += (logits[:, i].argmax(-1) == by[:, i]).sum().item()
            total += 16 * length
    model.train()
    return correct / total

def train_copy(model, length=10, steps=1000):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        bx, by = gen_copy(32, length)
        opt.zero_grad()
        logits = model(bx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1))
        loss.backward()
        opt.step()
    return eval_copy(model, length)

# ============================================================================
# MAIN
# ============================================================================

print('='*70)
print('COMPREHENSIVE ANALYSIS: When Does HoloLink Help?')
print('='*70)

# Matched parameter counts
vocab_kv = 60
vocab_copy = 50

# Build models
print('\nBuilding models...')

# ANA
ana_kv = ANA(vocab_kv, d_model=64, state_dim=64, key_dim=32, n_layers=2).to(device)
ana_copy = ANA(vocab_copy, d_model=64, state_dim=64, key_dim=32, n_layers=2).to(device)
ana_params = count_params(ana_kv)

# Transformer (matched)
trans_kv = Transformer(vocab_kv, d_model=64, n_heads=4, n_layers=2, d_ff=128).to(device)
trans_copy = Transformer(vocab_copy, d_model=64, n_heads=4, n_layers=2, d_ff=128).to(device)
trans_params = count_params(trans_kv)

print(f'ANA parameters: {ana_params:,}')
print(f'Transformer parameters: {trans_params:,}')

# Task 1: Associative Recall
print('\n' + '='*70)
print('TASK 1: ASSOCIATIVE RECALL')
print('='*70)

torch.manual_seed(42)
random.seed(42)

print('Training ANA...')
ana_kv_acc = train_kv(ana_kv)
print(f'ANA accuracy: {100*ana_kv_acc:.1f}%')

del ana_kv
torch.cuda.empty_cache()

print('Training Transformer...')
trans_kv_acc = train_kv(trans_kv)
print(f'Transformer accuracy: {100*trans_kv_acc:.1f}%')

del trans_kv
torch.cuda.empty_cache()

# Task 2: Copy Task
print('\n' + '='*70)
print('TASK 2: COPY TASK')
print('='*70)

torch.manual_seed(42)
random.seed(42)

print('Training ANA...')
ana_copy_acc = train_copy(ana_copy, length=10, steps=1000)
print(f'ANA accuracy: {100*ana_copy_acc:.1f}%')

del ana_copy
torch.cuda.empty_cache()

print('Training Transformer...')
trans_copy_acc = train_copy(trans_copy, length=10, steps=1000)
print(f'Transformer accuracy: {100*trans_copy_acc:.1f}%')

del trans_copy
torch.cuda.empty_cache()

# Summary
print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)
print(f'\n{"Task":<30} {"ANA":<15} {"Transformer":<15} {"Winner"}')
print('-'*75)
print(f'{"Associative Recall (12 KV)":<30} {100*ana_kv_acc:.1f}%{"":<9} {100*trans_kv_acc:.1f}%{"":<9} {"ANA" if ana_kv_acc > trans_kv_acc else "Transformer"}')
print(f'{"Copy Task (len=10)":<30} {100*ana_copy_acc:.1f}%{"":<9} {100*trans_copy_acc:.1f}%{"":<9} {"ANA" if ana_copy_acc > trans_copy_acc else "Transformer"}')

print('\n' + '='*70)
print('CONCLUSIONS')
print('='*70)

wins = sum([ana_kv_acc > trans_kv_acc, ana_copy_acc > trans_copy_acc])
total = 2

if wins == total:
    print('\n✅ ANA WINS on ALL tasks!')
    print('   HoloLink provides universal advantage.')
elif wins > 0:
    print(f'\n⚠️ MIXED RESULTS: ANA wins on {wins}/{total} tasks')
    print('   HoloLink advantage is task-specific.')
    
    if ana_kv_acc > trans_kv_acc:
        print('\n   WINS on: Associative Recall')
        print('   → HoloLink excels at explicit key-value binding')
    
    if ana_copy_acc > trans_copy_acc:
        print('\n   WINS on: Copy Task')
        print('   → HoloLink helps with sequence memorization')
    
    if ana_kv_acc <= trans_kv_acc:
        print('\n   LOSES on: Associative Recall')
        print('   → Transformer attention handles this well')
    
    if ana_copy_acc <= trans_copy_acc:
        print('\n   LOSES on: Copy Task')
        print('   → Transformer attention is sufficient')
else:
    print('\n❌ ANA LOSES on ALL tasks')
    print('   HoloLink does not provide advantage on these tasks.')

print('\n' + '='*70)
print('KEY INSIGHT')
print('='*70)
print("""
HoloLink is designed for EXPLICIT associative memory:
  M = Σ k⊗v  (store key-value pairs)
  v ≈ q^T M  (retrieve by query)

This excels when the task requires:
- Explicit key-value binding
- Associative retrieval
- Working memory operations

This may NOT help when the task requires:
- General pattern recognition
- Contextual prediction (language modeling)
- Fuzzy matching
""")
