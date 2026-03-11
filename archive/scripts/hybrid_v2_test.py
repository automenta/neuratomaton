#!/usr/bin/env python
"""
HYBRID v2: Transformer with Auxiliary HoloLink Memory

Key insight: Instead of mixing layers, use HoloLink as an auxiliary memory
module that the Transformer can query when needed.

Architecture:
  Input → Transformer Layers → [Optional: Query HoloLink Memory] → Output
  
The Transformer learns WHEN to use HoloLink memory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ============================================================================
# HOLOLINK MEMORY MODULE (Auxiliary)
# ============================================================================

class HoloLinkMemory(nn.Module):
    """Auxiliary memory that can be queried."""
    def __init__(self, d_model, key_dim=32):
        super().__init__()
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.bind = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """Returns memory output for all positions."""
        B, S, D = x.shape
        k = F.normalize(self.k_proj(x), p=2, dim=-1)
        v = self.v_proj(x)
        M = torch.cumsum(F.softplus(self.bind) * k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        retrieved = (q.unsqueeze(-2) @ M).squeeze(-2)
        return self.norm(self.out(retrieved))

# ============================================================================
# TRANSFORMER WITH AUXILIARY MEMORY
# ============================================================================

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
    
    def forward(self, x, mask=None):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None:
            attn = attn.masked_fill(mask[:S, :S], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        x = x + self.out((attn @ v).permute(0, 2, 1, 3).reshape(B, S, D))
        return x + self.ff(self.norm2(x))

class TransformerWithMemory(nn.Module):
    """Transformer with learnable access to HoloLink memory."""
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, d_ff=256, key_dim=32, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        
        # Auxiliary HoloLink memory
        self.memory = HoloLinkMemory(d_model, key_dim)
        
        # Learnable gate for memory access
        self.memory_gate = nn.Parameter(torch.zeros(1))
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer('mask', torch.triu(torch.ones(max_seq, max_seq), diagonal=1).bool())
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        
        # Transformer processing
        for layer in self.layers:
            x = layer(x, self.mask)
        
        # Optional memory retrieval (gated)
        gate = torch.sigmoid(self.memory_gate)
        mem_out = self.memory(x)
        x = x + gate * mem_out
        
        return self.head(self.norm(x))

class Transformer(nn.Module):
    """Baseline Transformer without memory."""
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, d_ff=256, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer('mask', torch.triu(torch.ones(max_seq, max_seq), diagonal=1).bool())
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x, self.mask)
        return self.head(self.norm(x))

# ============================================================================
# TASKS
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

def train_kv(model, steps=500):
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

def train_lm(model, data, vocab, epochs=20):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq_len = 64
    batch_size = 16
    
    for _ in range(epochs):
        for i in range(0, len(data) - seq_len - 1, batch_size * seq_len):
            batch_x, batch_y = [], []
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
    
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
print('HYBRID v2: Transformer with Auxiliary HoloLink Memory')
print('='*70)

# TEST 1: ASSOCIATIVE RECALL
print('\n' + '-'*70)
print('TEST 1: ASSOCIATIVE RECALL')
print('-'*70)

torch.manual_seed(42)
random.seed(42)

print('\nTraining Transformer (baseline)...')
trans = Transformer(60, n_layers=4).to(device)
trans_params = count_params(trans)
trans_kv = train_kv(trans)
print(f'Transformer ({trans_params:,}): {100*trans_kv:.1f}%')
del trans
torch.cuda.empty_cache()

print('\nTraining Transformer + Memory...')
trans_mem = TransformerWithMemory(60, n_layers=4).to(device)
trans_mem_params = count_params(trans_mem)
trans_mem_kv = train_kv(trans_mem)
gate_val = torch.sigmoid(trans_mem.memory_gate).item()
print(f'Trans+Mem ({trans_mem_params:,}): {100*trans_mem_kv:.1f}%')
print(f'Memory gate value: {gate_val:.3f}')
del trans_mem
torch.cuda.empty_cache()

# TEST 2: LANGUAGE MODELING
print('\n' + '-'*70)
print('TEST 2: LANGUAGE MODELING')
print('-'*70)

torch.manual_seed(42)
data, vocab, c2i = create_text()

print('\nTraining Transformer (baseline)...')
trans = Transformer(vocab, n_layers=4).to(device)
trans_ppl = train_lm(trans, data, vocab)
print(f'Transformer perplexity: {trans_ppl:.2f}')
del trans
torch.cuda.empty_cache()

print('\nTraining Transformer + Memory...')
trans_mem = TransformerWithMemory(vocab, n_layers=4).to(device)
trans_mem_ppl = train_lm(trans_mem, data, vocab)
gate_val = torch.sigmoid(trans_mem.memory_gate).item()
print(f'Trans+Mem perplexity: {trans_mem_ppl:.2f}')
print(f'Memory gate value: {gate_val:.3f}')
del trans_mem
torch.cuda.empty_cache()

# SUMMARY
print('\n' + '='*70)
print('RESULTS')
print('='*70)

print(f'\n{"Task":<25} {"Transformer":<15} {"+ Memory":<15} {"Improvement"}')
print('-'*70)

kv_diff = trans_mem_kv - trans_kv
ppl_diff = trans_ppl - trans_mem_ppl

print(f'{"Assoc Recall (acc)":<25} {100*trans_kv:.1f}%{"":<9} {100*trans_mem_kv:.1f}%{"":<9} {"+" if kv_diff > 0 else ""}{100*kv_diff:.1f}%')
print(f'{"Lang Model (ppl)":<25} {trans_ppl:.2f}{"":<11} {trans_mem_ppl:.2f}{"":<11} {"+" if ppl_diff > 0 else ""}{ppl_diff:.2f}')

print('\n' + '='*70)
print('VERDICT')
print('='*70)

if trans_mem_kv > trans_kv and trans_mem_ppl < trans_ppl:
    print('\n✅ Memory helps on BOTH tasks!')
    print('   Adding HoloLink memory improves Transformer universally.')
elif trans_mem_kv > trans_kv or trans_mem_ppl < trans_ppl:
    print('\n⚠️ Memory helps on one task')
    if trans_mem_kv > trans_kv:
        print('   Memory improves associative recall')
    if trans_mem_ppl < trans_ppl:
        print('   Memory improves language modeling')
else:
    print('\n❌ Memory does not help')
    print('   Simple gating is not enough - need smarter integration')
