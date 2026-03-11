#!/usr/bin/env python
"""
FAIR COMPARISON: ANA vs Transformer at MATCHED parameter counts.

The critical question: Does ANA beat a Transformer of the SAME size?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}\n')

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
        # LRU
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
        
        # HoloLink
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

# ========== Transformer (scalable) ==========

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

def make_transformer_for_params(target_params, vocab=60, max_seq=128):
    """Create Transformer with approximately target_params parameters."""
    configs = []
    for d_model in [16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 224, 256]:
        for n_layers in [1, 2, 3, 4, 5, 6]:
            for n_heads in [1, 2, 4, 8]:
                if d_model % n_heads != 0:
                    continue
                d_ff = d_model * 4
                
                emb = vocab * d_model
                pos = max_seq * d_model
                head = d_model * vocab
                
                block = (d_model * 3 * d_model + d_model * d_model +
                        d_model * d_ff + d_ff * d_model +
                        4 * d_model)
                total = emb + pos + n_layers * block + head + d_model
                
                configs.append((abs(total - target_params), d_model, n_layers, n_heads, total))
    
    configs.sort()
    _, d_model, n_layers, n_heads, actual = configs[0]
    return Transformer(d_model, n_heads, n_layers, d_model * 4), actual

def train_model(model, steps_per_level=400, lr=1e-3):
    """Train with curriculum."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    for pairs in [1, 2, 4, 6, 8, 10, 12]:
        for _ in range(steps_per_level):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx)[:, -1], by)
            loss.backward()
            opt.step()
    
    return eval_acc(model, 12)

# ========== MAIN EXPERIMENT ==========

print('='*70)
print('FAIR COMPARISON: ANA vs Transformer at Matched Parameter Counts')
print('='*70)

set_seed(42)

# Build ANA (our baseline)
print('\nBuilding ANA (~32K params)...')
ana = ANA(d_model=64, state_dim=64, key_dim=32, n_layers=2).to(device)
ana_params = count_params(ana)
print(f'ANA parameters: {ana_params:,}')

# Train ANA multiple times to get baseline
print('\nTraining ANA (3 seeds)...')
ana_accs = []
for seed in [42, 123, 456]:
    set_seed(seed)
    ana = ANA(d_model=64, state_dim=64, key_dim=32, n_layers=2).to(device)
    acc = train_model(ana)
    ana_accs.append(acc)
    print(f'  Seed {seed}: {100*acc:.1f}%')
    del ana
    torch.cuda.empty_cache()

ana_mean = np.mean(ana_accs)
ana_std = np.std(ana_accs)
print(f'ANA average: {100*ana_mean:.1f}% ± {100*ana_std:.1f}%')

# Test Transformers at different sizes
target_sizes = [32000, 64000, 128000, 256000, 512000, 1000000, 2000000, 4000000]
results = []

print('\n' + '='*70)
print('TRANSFORMER BASELINES')
print('='*70)

for target in target_sizes:
    print(f'\nTarget ~{target//1000}K params...')
    
    trans_accs = []
    actual_params_list = []
    
    for seed in [42, 123, 456]:
        set_seed(seed)
        trans, actual = make_transformer_for_params(target)
        trans = trans.to(device)
        actual_params_list.append(actual)
        
        acc = train_model(trans)
        trans_accs.append(acc)
        print(f'  Seed {seed}: {actual:,} params → {100*acc:.1f}%')
        
        del trans
        torch.cuda.empty_cache()
    
    trans_mean = np.mean(trans_accs)
    trans_std = np.std(trans_accs)
    actual_mean = np.mean(actual_params_list)
    
    results.append({
        'target': target,
        'actual': actual_mean,
        'accuracy': trans_mean,
        'std': trans_std
    })

# Summary
print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)
print(f'\n{"Model":<25} {"Params":<12} {"Accuracy":<18} {"vs ANA"}')
print('-'*75)
print(f'{"ANA (HoloLink)":<25} {ana_params:<12,} {100*ana_mean:.1f}% ± {100*ana_std:.1f}%     {"baseline"}')

for r in results:
    diff = r['accuracy'] - ana_mean
    sig = '+' if diff > 0 else ''
    name = f"Transformer {r['target']//1000}K"
    print(f'{name:<25} {int(r["actual"]):<12,} {100*r["accuracy"]:.1f}% ± {100*r["std"]:.1f}%     {sig}{100*diff:.1f}%')

# Verdict
print('\n' + '='*70)
print('VERDICT')
print('='*70)

best_trans = max(results, key=lambda x: x['accuracy'])
ana_wins = ana_mean > best_trans['accuracy']

if ana_wins:
    print(f'\n✅ UNIVERSAL BREAKTHROUGH CONFIRMED!')
    print(f'   ANA ({ana_params:,} params, {100*ana_mean:.1f}%) beats ALL Transformers')
    print(f'   Best Transformer: {int(best_trans["actual"]):,} params, {100*best_trans["accuracy"]:.1f}%')
else:
    print(f'\n⚠️ NO UNIVERSAL ADVANTAGE')
    print(f'   Transformer ({int(best_trans["actual"]):,} params) matches or beats ANA')
    print(f'   Transformer: {100*best_trans["accuracy"]:.1f}%')
    print(f'   ANA: {100*ana_mean:.1f}%')

# Check same-size comparison
same_size = [r for r in results if abs(r['actual'] - ana_params) < ana_params * 0.5]
if same_size:
    print(f'\nSame-size comparison (~{ana_params//1000}K params):')
    for r in same_size:
        if r['accuracy'] < ana_mean:
            print(f'  ANA wins by {100*(ana_mean - r["accuracy"]):.1f}%')
        else:
            print(f'  Transformer wins by {100*(r["accuracy"] - ana_mean):.1f}%')
