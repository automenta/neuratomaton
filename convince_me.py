#!/usr/bin/env python
"""
CONVINCE ME: The Definitive ANA vs Transformer Comparison

Run this script to see the truth about HoloLink:
1. Where it wins (associative recall)
2. Where it doesn't (language modeling)

No cherry-picking. Honest comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('='*70)
print('CONVINCE ME: ANA vs Transformer - The Honest Comparison')
print('='*70)
print(f'\nDevice: {device}')

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

def train_kv(model, steps=400):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for pairs in [1, 2, 4, 6, 8, 10, 12]:
        for _ in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            F.cross_entropy(model(bx)[:, -1], by).backward()
            opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(30):
            bx, by = gen_kv(32, 12)
            logits = model(bx)
            correct += (logits[:, -1].argmax(-1) == by).sum().item()
    return correct / (30 * 32)

# ============================================================================
# TASK 2: LANGUAGE MODELING
# ============================================================================

def create_text():
    text = """
To be, or not to be, that is the question:
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 20
    chars = sorted(list(set(text)))
    vocab = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text])
    return data, vocab, c2i

def train_lm(model, data, vocab, epochs=15):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq_len = 64
    batch_size = 16
    
    for _ in range(epochs):
        for i in range(0, len(data) - seq_len - 1, batch_size * seq_len):
            batch_x = []
            batch_y = []
            for j in range(batch_size):
                start = i + j * seq_len
                if start + seq_len < len(data):
                    batch_x.append(data[start:start+seq_len])
                    batch_y.append(data[start+1:start+seq_len+1])
            if batch_x:
                x = torch.stack(batch_x).to(device)
                y = torch.stack(batch_y).to(device)
                opt.zero_grad()
                logits = model(x)
                F.cross_entropy(logits.view(-1, vocab), y.view(-1)).backward()
                opt.step()
    
    # Compute perplexity
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(0, min(1000, len(data) - seq_len - 1), seq_len):
            x = data[i:i+seq_len].unsqueeze(0).to(device)
            y = data[i+1:i+seq_len+1].unsqueeze(0).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
            total_loss += loss.item() * seq_len
            count += seq_len
    return math.exp(total_loss / count)

# ============================================================================
# MAIN
# ============================================================================

print('\n' + '='*70)
print('TEST 1: ASSOCIATIVE RECALL (HoloLink should win)')
print('='*70)

torch.manual_seed(42)
random.seed(42)

print('\nTraining ANA...')
ana = ANA(60).to(device)
ana_params = count_params(ana)
ana_kv_acc = train_kv(ana)
print(f'ANA ({ana_params:,} params): {100*ana_kv_acc:.1f}%')

del ana
torch.cuda.empty_cache()

print('\nTraining Transformer...')
trans = Transformer(60, d_model=64, n_heads=4, n_layers=2, d_ff=128).to(device)
trans_params = count_params(trans)
trans_kv_acc = train_kv(trans)
print(f'Transformer ({trans_params:,} params): {100*trans_kv_acc:.1f}%')

del trans
torch.cuda.empty_cache()

print('\n' + '='*70)
print('TEST 2: LANGUAGE MODELING (Transformer should win)')
print('='*70)

torch.manual_seed(42)

print('\nLoading text data...')
data, vocab, c2i = create_text()

print('\nTraining ANA...')
ana = ANA(vocab).to(device)
ana_ppl = train_lm(ana, data, vocab)
print(f'ANA perplexity: {ana_ppl:.2f}')

del ana
torch.cuda.empty_cache()

print('\nTraining Transformer...')
trans = Transformer(vocab, d_model=64, n_heads=4, n_layers=2, d_ff=128).to(device)
trans_ppl = train_lm(trans, data, vocab)
print(f'Transformer perplexity: {trans_ppl:.2f}')

del trans
torch.cuda.empty_cache()

# VERDICT
print('\n' + '='*70)
print('VERDICT')
print('='*70)

print(f'\n{"Task":<30} {"ANA":<15} {"Transformer":<15} {"Winner"}')
print('-'*75)
print(f'{"Associative Recall":<30} {100*ana_kv_acc:.1f}%{"":<9} {100*trans_kv_acc:.1f}%{"":<9} {"ANA ✓" if ana_kv_acc > trans_kv_acc else "Transformer"}')
print(f'{"Language Modeling (PPL)":<30} {ana_ppl:.2f}{"":<11} {trans_ppl:.2f}{"":<11} {"Transformer ✓" if trans_ppl < ana_ppl else "ANA"}')

print('\n' + '='*70)
print('THE HONEST CONCLUSION')
print('='*70)

if ana_kv_acc > trans_kv_acc and trans_ppl < ana_ppl:
    print("""
✅ BREAKTHROUGH CONFIRMED (but task-specific)

HoloLink provides MAJOR advantage for associative memory:
  - ANA wins associative recall by {:.0f}x ({:.1f}% vs {:.1f}%)
  
Transformer is better for language modeling:
  - Transformer wins by {:.1f}% lower perplexity ({:.2f} vs {:.2f})

The lesson: Use the right architecture for the right task.
  - ANA/HoloLink for: Retrieval, KV lookup, working memory
  - Transformers for: Language modeling, contextual prediction
""".format(ana_kv_acc/max(trans_kv_acc, 0.01), 100*ana_kv_acc, 100*trans_kv_acc,
           (ana_ppl - trans_ppl)/trans_ppl*100, trans_ppl, ana_ppl))
else:
    print("Results need further investigation.")
