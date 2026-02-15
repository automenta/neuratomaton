#!/usr/bin/env python
"""
BREAKTHROUGH DEMONSTRATION: Compact ANA vs Large Transformer

Hypothesis: A small ANA model with HoloLink can outperform significantly 
larger Transformer models on associative recall and in-context learning tasks.

This demonstrates "parameter efficiency" - the key metric for edge deployment
and sustainable AI.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import time
from dataclasses import dataclass
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ============================================================================
# TASK SUITE: Multiple associative tasks to prove generalization
# ============================================================================

TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gen_kv_recall(batch, pairs, vocab_size=60, noise_len=10):
    content = list(range(4, vocab_size))
    x, y = [], []
    for _ in range(batch):
        keys = random.sample(content, pairs)
        vals = random.sample([t for t in content if t not in keys], pairs)
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([TOK_KEY, k, TOK_VAL, v])
        seq.extend(random.choices(content, k=noise_len))
        q = random.randint(0, pairs-1)
        seq.extend([TOK_QUERY, keys[q]])
        x.append(seq)
        y.append(vals[q])
    mx = max(len(s) for s in x)
    t = torch.zeros(batch, mx, dtype=torch.long)
    for i, s in enumerate(x):
        t[i, :len(s)] = torch.tensor(s)
    return t, torch.tensor(y)

def gen_associative_scan(batch, length, vocab_size=100):
    """Remember first element and apply transformations."""
    x, y = [], []
    for _ in range(batch):
        start = random.randint(10, vocab_size-10)
        seq = [start]
        current = start
        for _ in range(length):
            op = random.choice(['add', 'sub', 'mul'])
            val = random.randint(1, 5)
            if op == 'add':
                seq.append(1)
                seq.append(val)
                current = (current + val) % vocab_size
            elif op == 'sub':
                seq.append(2)
                seq.append(val)
                current = (current - val) % vocab_size
            else:
                seq.append(3)
                seq.append(val)
                current = (current * val) % vocab_size
        x.append(seq)
        y.append(current)
    mx = max(len(s) for s in x)
    t = torch.zeros(batch, mx, dtype=torch.long)
    for i, s in enumerate(x):
        t[i, :len(s)] = torch.tensor(s)
    return t, torch.tensor(y)

def gen_pattern_completion(batch, pattern_len, vocab_size=50):
    """Complete a repeated pattern."""
    x, y = [], []
    for _ in range(batch):
        pattern = [random.randint(4, vocab_size) for _ in range(pattern_len)]
        seq = pattern * 3 + pattern[:pattern_len//2]
        target = pattern[pattern_len//2]
        x.append(seq)
        y.append(target)
    mx = max(len(s) for s in x)
    t = torch.zeros(batch, mx, dtype=torch.long)
    for i, s in enumerate(x):
        t[i, :len(s)] = torch.tensor(s)
    return t, torch.tensor(y)

def eval_model(model, task, task_args, n=50, batch=32):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            bx, by = task(batch, *task_args)
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            if isinstance(logits, tuple):
                logits = logits[0]
            correct += (logits[:, -1].argmax(-1) == by).sum().item()
    model.train()
    return correct / (n * batch)

# ============================================================================
# BASELINE: Standard Transformer (scaled variants)
# ============================================================================

@dataclass
class TransformerConfig:
    vocab_size: int = 100
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.ff_up = nn.Linear(config.d_model, config.d_ff)
        self.ff_down = nn.Linear(config.d_ff, config.d_model)
    
    def forward(self, x):
        B, S, D = x.shape
        
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        x = x + self.o_proj(out)
        
        h = self.norm2(x)
        x = x + self.ff_down(F.gelu(self.ff_up(h)))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_emb(torch.arange(S, device=input_ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

# ============================================================================
# ANA: Our compact model with HoloLink
# ============================================================================

@dataclass
class ANAConfig:
    vocab_size: int = 100
    d_model: int = 64
    state_dim: int = 64
    key_dim: int = 32
    n_layers: int = 2
    max_seq_len: int = 256

class LinearRecurrentUnit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.d_model, config.state_dim)
        self.output_proj = nn.Linear(config.state_dim, config.d_model)
        self.alpha_logit = nn.Parameter(torch.zeros(config.state_dim))
        self.beta_logit = nn.Parameter(torch.zeros(config.state_dim))
    
    def forward_sequence(self, x):
        B, S, D = x.shape
        u = self.input_proj(x)
        alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1)
        beta = torch.sigmoid(self.beta_logit).view(1, 1, -1)
        
        h = torch.zeros(B, config.state_dim if hasattr(self, 'config') else 64, device=x.device)
        h_list = []
        for t in range(S):
            h = alpha * h + beta * u[:, t]
            h_list.append(h)
        
        h_seq = torch.stack(h_list, dim=1)
        return self.output_proj(h_seq), h_seq

class HoloLink(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_dim = config.key_dim
        self.d_model = config.d_model
        
        self.k_proj = nn.Linear(config.state_dim if hasattr(config, 'state_dim') else config.d_model, config.key_dim, bias=False)
        self.v_proj = nn.Linear(config.state_dim if hasattr(config, 'state_dim') else config.d_model, config.d_model, bias=False)
        self.q_proj = nn.Linear(config.d_model, config.key_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.binding = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, h):
        B, S, D = x.shape
        
        k = F.normalize(self.k_proj(h), p=2, dim=-1)
        v = self.v_proj(h)
        
        updates = F.softplus(self.binding) * torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        M = torch.cumsum(updates, dim=1)
        
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        retrieved = torch.matmul(q.unsqueeze(-2), M).squeeze(-2)
        
        return self.norm(self.out_proj(retrieved))

class ANA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(nn.ModuleDict({
                'lru': LinearRecurrentUnit(config),
                'holo': HoloLink(config)
            }))
        
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_emb(torch.arange(S, device=input_ids.device))
        
        for layer in self.layers:
            y, h = layer['lru'].forward_sequence(x)
            q = layer['holo'](x, h)
            x = x + y + q
        
        return self.head(self.norm(x))

# Fix the LinearRecurrentUnit for proper config access
class LinearRecurrentUnit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_dim = config.state_dim
        self.input_proj = nn.Linear(config.d_model, config.state_dim)
        self.output_proj = nn.Linear(config.state_dim, config.d_model)
        self.alpha_logit = nn.Parameter(torch.zeros(config.state_dim))
        self.beta_logit = nn.Parameter(torch.zeros(config.state_dim))
    
    def forward_sequence(self, x):
        B, S, D = x.shape
        u = self.input_proj(x)
        alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1)
        beta = torch.sigmoid(self.beta_logit).view(1, 1, -1)
        
        h = torch.zeros(B, self.state_dim, device=x.device)
        h_list = []
        for t in range(S):
            h = alpha.squeeze(1) * h + beta.squeeze(1) * u[:, t]
            h_list.append(h)
        
        h_seq = torch.stack(h_list, dim=1)
        return self.output_proj(h_seq), h_seq

class HoloLink(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_dim = config.key_dim
        self.d_model = config.d_model
        
        self.k_proj = nn.Linear(config.state_dim, config.key_dim, bias=False)
        self.v_proj = nn.Linear(config.state_dim, config.d_model, bias=False)
        self.q_proj = nn.Linear(config.d_model, config.key_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.binding = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, h):
        B, S, D = x.shape
        
        k = F.normalize(self.k_proj(h), p=2, dim=-1)
        v = self.v_proj(h)
        
        updates = F.softplus(self.binding) * torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        M = torch.cumsum(updates, dim=1)
        
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        retrieved = torch.matmul(q.unsqueeze(-2), M).squeeze(-2)
        
        return self.norm(self.out_proj(retrieved))

# ============================================================================
# TRAINING PROTOCOLS
# ============================================================================

def train_model(model, task, task_args, steps=2000, lr=1e-3, batch=32, verbose=True):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for step in range(steps):
        bx, by = task(batch, *task_args)
        bx, by = bx.to(device), by.to(device)
        
        opt.zero_grad()
        logits = model(bx)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits[:, -1, :], by)
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        
        if verbose and (step + 1) % 500 == 0:
            print(f'    Step {step+1}: loss={loss.item():.4f}')
    
    return losses

def train_with_curriculum(model, task_name, max_pairs=12, verbose=True):
    """Train with curriculum for KV recall task."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    curriculum = [(p, 400 if p < 8 else 600) for p in [1, 2, 4, 6, 8, 10, max_pairs]]
    
    for pairs, steps in curriculum:
        for step in range(steps):
            bx, by = gen_kv_recall(32, pairs)
            bx, by = bx.to(device), by.to(device)
            
            opt.zero_grad()
            logits = model(bx)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits[:, -1, :], by)
            loss.backward()
            opt.step()
        
        if verbose:
            acc = eval_model(model, gen_kv_recall, (pairs,))
            print(f'    {pairs} pairs: {100*acc:.1f}%')

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print('\n' + '='*70)
    print('BREAKTHROUGH DEMONSTRATION: Compact ANA vs Large Transformers')
    print('='*70)
    
    results = defaultdict(dict)
    
    # Define models with varying sizes
    models_config = {
        'ANA (0.5M)': ANAConfig(d_model=64, state_dim=64, key_dim=32, n_layers=2),
        'ANA (1M)': ANAConfig(d_model=96, state_dim=96, key_dim=48, n_layers=3),
        'Transformer (2M)': TransformerConfig(d_model=128, n_heads=4, n_layers=4, d_ff=512),
        'Transformer (5M)': TransformerConfig(d_model=192, n_heads=6, n_layers=6, d_ff=768),
        'Transformer (10M)': TransformerConfig(d_model=256, n_heads=8, n_layers=8, d_ff=1024),
    }
    
    # Build and count parameters
    print('\n[1] Model Sizes')
    print('-' * 50)
    models = {}
    for name, config in models_config.items():
        if 'ANA' in name:
            models[name] = ANA(config).to(device)
        else:
            models[name] = Transformer(config).to(device)
        params = count_parameters(models[name])
        print(f'  {name}: {params/1e6:.2f}M parameters')
    
    # Task 1: KV Recall (scaling)
    print('\n[2] Task: KV Associative Recall')
    print('-' * 50)
    
    for name, model in models.items():
        print(f'\n  Training {name}...')
        train_with_curriculum(model, 'kv_recall', max_pairs=12, verbose=True)
        
        for pairs in [4, 8, 12]:
            acc = eval_model(model, gen_kv_recall, (pairs,), n=50)
            results[name][f'kv_{pairs}'] = acc
            print(f'    Eval {pairs} pairs: {100*acc:.1f}%')
        
        del model
        torch.cuda.empty_cache()
    
    # Rebuild models for Task 2
    models = {}
    for name, config in models_config.items():
        if 'ANA' in name:
            models[name] = ANA(config).to(device)
        else:
            models[name] = Transformer(config).to(device)
    
    # Task 2: Pattern Completion
    print('\n[3] Task: Pattern Completion')
    print('-' * 50)
    
    for name, model in models.items():
        print(f'\n  Training {name}...')
        train_model(model, gen_pattern_completion, (4,), steps=1500, verbose=True)
        
        for plen in [3, 4, 5]:
            acc = eval_model(model, gen_pattern_completion, (plen,), n=50)
            results[name][f'pattern_{plen}'] = acc
            print(f'    Pattern len {plen}: {100*acc:.1f}%')
        
        del model
        torch.cuda.empty_cache()
    
    # Rebuild for Task 3
    models = {}
    for name, config in models_config.items():
        if 'ANA' in name:
            models[name] = ANA(config).to(device)
        else:
            models[name] = Transformer(config).to(device)
    
    # Task 3: Associative Scan
    print('\n[4] Task: Associative Scan')
    print('-' * 50)
    
    for name, model in models.items():
        print(f'\n  Training {name}...')
        train_model(model, gen_associative_scan, (5,), steps=2000, verbose=True)
        
        for length in [5, 10, 15]:
            acc = eval_model(model, gen_associative_scan, (length,), n=50)
            results[name][f'scan_{length}'] = acc
            print(f'    Scan len {length}: {100*acc:.1f}%')
        
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print('\n' + '='*70)
    print('BREAKTHROUGH RESULTS')
    print('='*70)
    
    print('\nKV Associative Recall (12 pairs):')
    print('-' * 50)
    for name in models_config.keys():
        if f'kv_12' in results[name]:
            params = count_parameters(ANA(models_config[name]) if 'ANA' in name else Transformer(models_config[name]))
            print(f'  {name} ({params/1e6:.1f}M): {100*results[name]["kv_12"]:.1f}%')
    
    print('\nParameter Efficiency (Accuracy per Million Parameters):')
    print('-' * 50)
    efficiencies = []
    for name in models_config.keys():
        if f'kv_12' in results[name]:
            config = models_config[name]
            params = count_parameters(ANA(config) if 'ANA' in name else Transformer(config))
            eff = results[name]['kv_12'] / (params / 1e6)
            efficiencies.append((name, params/1e6, results[name]['kv_12'], eff))
            print(f'  {name}: {100*eff:.1f}% per M params')
    
    # Find the winner
    print('\n' + '='*70)
    print('WINNER: Highest Accuracy-to-Parameter Ratio')
    print('='*70)
    winner = max(efficiencies, key=lambda x: x[3])
    print(f'  {winner[0]} ({winner[1]:.1f}M params): {100*winner[2]:.1f} accuracy, {100*winner[3]:.1f}% per M params')
    
    if 'ANA' in winner[0]:
        print('\n  🎯 BREAKTHROUGH: Compact ANA with HoloLink achieves superior')
        print('     parameter efficiency compared to much larger Transformers!')
    
    return results

if __name__ == '__main__':
    results = main()
