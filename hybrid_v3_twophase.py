#!/usr/bin/env python
"""
HYBRID v3: Two-Phase Training with Memory

Hypothesis: The key is training order. We should:
1. Train memory module first (like we did with ANA)
2. Then train Transformer to use the frozen memory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class HoloLinkMemory(nn.Module):
    def __init__(self, d_model, key_dim=32):
        super().__init__()
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.bind = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        B, S, D = x.shape
        k = F.normalize(self.k_proj(x), p=2, dim=-1)
        v = self.v_proj(x)
        M = torch.cumsum(F.softplus(self.bind) * k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        return self.norm(self.out((q.unsqueeze(-2) @ M).squeeze(-2)))

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

class HybridModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, d_ff=256, key_dim=32, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.memory = HoloLinkMemory(d_model, key_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer('mask', torch.triu(torch.ones(max_seq, max_seq), diagonal=1).bool())
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x, self.mask)
        x = x + torch.sigmoid(self.gate) * self.memory(x)
        return self.head(self.norm(x))
    
    def freeze_memory(self):
        for p in self.memory.parameters():
            p.requires_grad = False
    
    def unfreeze_memory(self):
        for p in self.memory.parameters():
            p.requires_grad = True
    
    def freeze_transformer(self):
        for p in self.emb.parameters():
            p.requires_grad = False
        for p in self.pos.parameters():
            p.requires_grad = False
        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
    
    def unfreeze_transformer(self):
        for p in self.emb.parameters():
            p.requires_grad = True
        for p in self.pos.parameters():
            p.requires_grad = True
        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

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

def eval_kv(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(30):
            bx, by = gen_kv(32, 12)
            logits = model(bx)
            correct += (logits[:, -1].argmax(-1) == by).sum().item()
    model.train()
    return correct / (30 * 32)

def train_joint(model, steps=500):
    """Standard joint training."""
    model.train()
    model.unfreeze_memory()
    model.unfreeze_transformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for pairs in [1, 2, 4, 6, 8, 10, 12]:
        for _ in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            F.cross_entropy(model(bx)[:, -1], by).backward()
            opt.step()
    return eval_kv(model)

def train_two_phase(model, steps=500):
    """Two-phase training: Memory first, then Transformer."""
    model.train()
    
    # Phase 1: Train memory only
    model.freeze_transformer()
    model.unfreeze_memory()
    mem_params = list(model.memory.parameters()) + [model.gate]
    opt = torch.optim.Adam(mem_params, lr=1e-3)
    
    for pairs in [1, 2, 4, 6, 8, 10, 12]:
        for _ in range(steps):
            bx, by = gen_kv(32, pairs)
            opt.zero_grad()
            F.cross_entropy(model(bx)[:, -1], by).backward()
            opt.step()
    
    phase1_acc = eval_kv(model)
    
    # Phase 2: Train Transformer only
    model.freeze_memory()
    model.unfreeze_transformer()
    trans_params = []
    for p in model.emb.parameters(): trans_params.append(p)
    for p in model.pos.parameters(): trans_params.append(p)
    for layer in model.layers:
        trans_params.extend(layer.parameters())
    trans_params.extend(model.head.parameters())
    
    opt = torch.optim.Adam(trans_params, lr=1e-4)  # Lower LR
    
    for _ in range(steps * 2):
        bx, by = gen_kv(32, 12)
        opt.zero_grad()
        F.cross_entropy(model(bx)[:, -1], by).backward()
        opt.step()
    
    return eval_kv(model), phase1_acc

print('='*70)
print('HYBRID v3: Two-Phase Training')
print('='*70)

torch.manual_seed(42)
random.seed(42)

# Joint training
print('\n[1] Joint Training...')
model_joint = HybridModel(60).to(device)
joint_params = count_params(model_joint)
joint_acc = train_joint(model_joint)
print(f'Joint ({joint_params:,}): {100*joint_acc:.1f}%')
del model_joint
torch.cuda.empty_cache()

# Two-phase training
print('\n[2] Two-Phase Training...')
model_two = HybridModel(60).to(device)
two_phase_acc, phase1_acc = train_two_phase(model_two)
print(f'Phase 1 (Memory): {100*phase1_acc:.1f}%')
print(f'Phase 2 (+Trans): {100*two_phase_acc:.1f}%')

# Compare gate values
gate_val = torch.sigmoid(model_two.gate).item()
print(f'Memory gate: {gate_val:.3f}')

del model_two
torch.cuda.empty_cache()

# Control: Train memory-only model
print('\n[3] Memory-Only Control...')
model_mem = HybridModel(60).to(device)
model_mem.freeze_transformer()
model_mem.unfreeze_memory()
opt = torch.optim.Adam(list(model_mem.memory.parameters()) + [model_mem.gate], lr=1e-3)
for pairs in [1, 2, 4, 6, 8, 10, 12]:
    for _ in range(500):
        bx, by = gen_kv(32, pairs)
        opt.zero_grad()
        F.cross_entropy(model_mem(bx)[:, -1], by).backward()
        opt.step()
mem_only_acc = eval_kv(model_mem)
print(f'Memory-Only: {100*mem_only_acc:.1f}%')
del model_mem
torch.cuda.empty_cache()

print('\n' + '='*70)
print('RESULTS')
print('='*70)
print(f'\nJoint Training:      {100*joint_acc:.1f}%')
print(f'Two-Phase Training:  {100*two_phase_acc:.1f}%')
print(f'Memory-Only:         {100*mem_only_acc:.1f}%')

if two_phase_acc > joint_acc:
    print(f'\n✅ Two-phase training improves by {100*(two_phase_acc - joint_acc):.1f}%')
else:
    print(f'\n❌ Two-phase does not help')
