#!/usr/bin/env python
"""
QUICK VERIFICATION: 30-second test of breakthrough result.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def gen_kv(batch, pairs):
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    content = list(range(4, 60))
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

def eval_acc(model, pairs, n=20):
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

# ANA with HoloLink
class ANA(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(60, 64)
        self.pos = nn.Embedding(128, 64)
        self.in_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 32, bias=False)
        self.v_proj = nn.Linear(64, 64, bias=False)
        self.q_proj = nn.Linear(64, 32, bias=False)
        self.out = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 60)
        self.alpha = nn.Parameter(torch.zeros(64))
        self.beta = nn.Parameter(torch.zeros(64))
        self.bind = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        
        # LRU
        u = self.in_proj(x)
        a = torch.sigmoid(self.alpha).view(1, 1, -1)
        b = torch.sigmoid(self.beta).view(1, 1, -1)
        h = torch.zeros(B, 64, device=ids.device)
        hs = []
        for t in range(S):
            h = a.squeeze(1) * h + b.squeeze(1) * u[:, t]
            hs.append(h)
        h = torch.stack(hs, dim=1)
        
        # HoloLink
        k = F.normalize(self.k_proj(h), p=2, dim=-1)
        v = self.v_proj(h)
        M = torch.cumsum(F.softplus(self.bind) * k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        retrieved = self.norm(self.out((q.unsqueeze(-2) @ M).squeeze(-2)))
        
        x = x + retrieved
        return self.head(self.norm(x))

# Transformer
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(60, 256)
        self.pos = nn.Embedding(128, 256)
        self.layers = nn.ModuleList()
        for _ in range(6):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(256),
                'qkv': nn.Linear(256, 768, bias=False),
                'out': nn.Linear(256, 256, bias=False),
                'norm2': nn.LayerNorm(256),
                'ff': nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))
            }))
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 60)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for L in self.layers:
            h = L['norm1'](x)
            qkv = L['qkv'](h).view(B, S, 3, 8, 32).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = F.softmax((q @ k.transpose(-2, -1)) / 5.66, dim=-1)
            x = x + L['out']((attn @ v).permute(0, 2, 1, 3).reshape(B, S, 256))
            x = x + L['ff'](L['norm2'](x))
        return self.head(self.norm(x))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('='*60)
print('QUICK VERIFICATION (30 seconds)')
print('='*60)

set_seed(42)

# ANA
print('\nTraining ANA...')
ana = ANA().to(device)
print(f'Parameters: {count_params(ana):,}')
opt = torch.optim.Adam(ana.parameters(), lr=1e-3)
for pairs in [1, 2, 4, 6, 8, 10, 12]:
    for _ in range(300):
        bx, by = gen_kv(32, pairs)
        opt.zero_grad()
        F.cross_entropy(ana(bx)[:, -1], by).backward()
        opt.step()

ana_acc = eval_acc(ana, 12)
print(f'ANA 12-KV: {100*ana_acc:.1f}%')

# Transformer
print('\nTraining Transformer...')
set_seed(42)
trans = Transformer().to(device)
print(f'Parameters: {count_params(trans):,}')
opt = torch.optim.Adam(trans.parameters(), lr=1e-3)
for pairs in [1, 2, 4, 6, 8, 10, 12]:
    for _ in range(300):
        bx, by = gen_kv(32, pairs)
        opt.zero_grad()
        F.cross_entropy(trans(bx)[:, -1], by).backward()
        opt.step()

trans_acc = eval_acc(trans, 12)
print(f'Transformer 12-KV: {100*trans_acc:.1f}%')

# Result
ana_p = count_params(ana)
trans_p = count_params(trans)
eff = (ana_acc / (ana_p/1e6)) / (trans_acc / (trans_p/1e6))

print('\n' + '='*60)
print('RESULT')
print('='*60)
print(f'ANA:         {ana_p:,} params, {100*ana_acc:.1f}% accuracy')
print(f'Transformer: {trans_p:,} params, {100*trans_acc:.1f}% accuracy')
print(f'Size ratio:  {trans_p//ana_p}x')
print(f'Efficiency:  {eff:.0f}x')

if ana_acc > 0.85 and trans_acc < 0.15:
    print('\n✅ BREAKTHROUGH VERIFIED')
