#!/usr/bin/env python
"""
Practical Two-Phase Training: LoRA Fine-Tuning

Tests the most common modular training scenario:
1. Pre-train base model
2. Add LoRA adapters
3. Compare: Joint fine-tuning vs Two-Phase fine-tuning

This mirrors real-world usage where you have a pre-trained model
and want to add adapters efficiently.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3

def gen_kv_data(batch, pairs):
    content = list(range(4, 60))
    x, y = [], []
    for _ in range(batch):
        keys = random.sample(content, pairs)
        vals = random.sample([t for t in content if t not in keys], pairs)
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([TOK_KEY, k, TOK_VAL, v])
        seq.extend(random.choices(content, k=10))
        q = random.randint(0, pairs-1)
        seq.extend([TOK_QUERY, keys[q]])
        x.append(seq)
        y.append(vals[q])
    mx = max(len(s) for s in x)
    t = torch.zeros(batch, mx, dtype=torch.long)
    for i, s in enumerate(x):
        t[i, :len(s)] = torch.tensor(s)
    return t, torch.tensor(y)

def eval_model(model, pairs, n=50, batch=32):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            bx, by = gen_kv_data(batch, pairs)
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            if isinstance(logits, tuple):
                logits = logits[0]
            correct += (logits[:, -1].argmax(-1) == by).sum().item()
    model.train()
    return correct / (n * batch)

@dataclass
class Config:
    vocab_size: int = 60
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    max_seq_len: int = 128
    d_ff: int = 512
    lora_rank: int = 16
    lora_alpha: float = 32.0

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, alpha=32.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.lora_enabled = True
    
    def forward(self, x):
        base_out = self.linear(x)
        if self.lora_enabled and self.lora_A.requires_grad:
            return base_out + (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out
    
    def disable_lora(self):
        self.lora_enabled = False
    
    def enable_lora(self):
        self.lora_enabled = True

class AttentionBlock(nn.Module):
    def __init__(self, config: Config, use_lora=False):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        if use_lora:
            self.q_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
            self.k_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
            self.v_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
            self.o_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
        else:
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.ff_up = nn.Linear(config.d_model, config.d_ff)
        self.ff_down = nn.Linear(config.d_ff, config.d_model)
        
        self.use_lora = use_lora
    
    def forward(self, x):
        B, S, D = x.shape
        
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        out = self.o_proj(out)
        x = x + out
        
        h = self.norm2(x)
        ff = F.gelu(self.ff_up(h))
        ff = self.ff_down(ff)
        x = x + ff
        return x

class TransformerWithLoRA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.base_layers = nn.ModuleList([
            AttentionBlock(config, use_lora=False) for _ in range(config.n_layers)
        ])
        self.lora_layers = nn.ModuleList([
            AttentionBlock(config, use_lora=True) for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_emb(torch.arange(S, device=input_ids.device))
        
        for base, lora in zip(self.base_layers, self.lora_layers):
            x = base(x) + lora(x)
        
        x = self.norm(x)
        return self.head(x)
    
    def get_base_params(self):
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.pos_emb.parameters())
        for layer in self.base_layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.head.parameters())
        return params
    
    def get_lora_params(self):
        params = []
        for layer in self.lora_layers:
            for name, module in layer.named_modules():
                if isinstance(module, LoRALinear):
                    params.extend([module.lora_A, module.lora_B])
        return params
    
    def freeze_base(self):
        for p in self.get_base_params():
            p.requires_grad = False
    
    def unfreeze_base(self):
        for p in self.get_base_params():
            p.requires_grad = True
    
    def freeze_lora(self):
        for p in self.get_lora_params():
            p.requires_grad = False
    
    def unfreeze_lora(self):
        for p in self.get_lora_params():
            p.requires_grad = True
    
    def disable_lora(self):
        for layer in self.lora_layers:
            for name, module in layer.named_modules():
                if isinstance(module, LoRALinear):
                    module.disable_lora()
    
    def enable_lora(self):
        for layer in self.lora_layers:
            for name, module in layer.named_modules():
                if isinstance(module, LoRALinear):
                    module.enable_lora()

def main():
    print('\n' + '='*70)
    print('PRACTICAL TWO-PHASE TRAINING: LoRA Fine-Tuning')
    print('='*70)
    
    config = Config()
    curriculum = [(1, 500), (2, 500), (4, 500), (6, 500), (8, 500), (10, 500), (12, 600)]
    
    results = {}
    
    # =========================================================================
    # STEP 1: Pre-train base model
    # =========================================================================
    print('\n[1] Pre-training Base Model (without LoRA)...')
    model = TransformerWithLoRA(config).to(device)
    model.disable_lora()
    model.freeze_lora()
    
    opt = torch.optim.Adam(model.get_base_params(), lr=1e-3)
    
    for pairs, steps in curriculum:
        for step in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
        
        if pairs in [1, 4, 8, 12]:
            acc = eval_model(model, pairs, n=30)
            print(f'  {pairs:2d} pairs: {100*acc:.1f}%')
    
    base_acc = eval_model(model, 12)
    print(f'  >>> Base Model: {100*base_acc:.1f}%')
    
    # Save base weights
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # STEP 2: Joint Fine-tuning (Base + LoRA together)
    # =========================================================================
    print('\n[2] Joint Fine-tuning (Base + LoRA together)...')
    model = TransformerWithLoRA(config).to(device)
    model.load_state_dict(base_state)
    model.enable_lora()
    model.unfreeze_base()
    model.unfreeze_lora()
    
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    for pairs, steps in [(12, 1000)]:
        for step in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
            
            if (step + 1) % 200 == 0:
                acc = eval_model(model, pairs, n=30)
                print(f'  Step {step+1}: {100*acc:.1f}%')
    
    results['joint'] = eval_model(model, 12)
    print(f'  >>> Joint Fine-tuning: {100*results["joint"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # STEP 3: Two-Phase Fine-tuning
    # =========================================================================
    print('\n[3] Two-Phase Fine-tuning...')
    model = TransformerWithLoRA(config).to(device)
    model.load_state_dict(base_state)
    model.enable_lora()
    
    # Phase 1: Train LoRA only (Base frozen)
    print('  Phase 1: Training LoRA (Base frozen)...')
    model.freeze_base()
    model.unfreeze_lora()
    
    opt = torch.optim.Adam(model.get_lora_params(), lr=1e-3)
    
    for step in range(500):
        bx, by = gen_kv_data(32, 12)
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        logits = model(bx)
        F.cross_entropy(logits[:, -1, :], by).backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            acc = eval_model(model, 12, n=30)
            print(f'    Step {step+1}: {100*acc:.1f}%')
    
    phase1_acc = eval_model(model, 12)
    print(f'  >>> Phase 1 (LoRA only): {100*phase1_acc:.1f}%')
    
    # Phase 2: Fine-tune Base (LoRA frozen)
    print('  Phase 2: Fine-tuning Base (LoRA frozen)...')
    model.freeze_lora()
    model.unfreeze_base()
    
    opt = torch.optim.Adam(model.get_base_params(), lr=1e-5)
    
    for step in range(500):
        bx, by = gen_kv_data(32, 12)
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        logits = model(bx)
        F.cross_entropy(logits[:, -1, :], by).backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            acc = eval_model(model, 12, n=30)
            print(f'    Step {step+1}: {100*acc:.1f}%')
    
    results['two_phase'] = eval_model(model, 12)
    print(f'  >>> Phase 2 (+Base): {100*results["two_phase"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # STEP 4: LoRA-Only Fine-tuning (Standard practice)
    # =========================================================================
    print('\n[4] LoRA-Only Fine-tuning (Standard practice)...')
    model = TransformerWithLoRA(config).to(device)
    model.load_state_dict(base_state)
    model.enable_lora()
    model.freeze_base()
    model.unfreeze_lora()
    
    opt = torch.optim.Adam(model.get_lora_params(), lr=1e-3)
    
    for step in range(1000):
        bx, by = gen_kv_data(32, 12)
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        logits = model(bx)
        F.cross_entropy(logits[:, -1, :], by).backward()
        opt.step()
        
        if (step + 1) % 200 == 0:
            acc = eval_model(model, 12, n=30)
            print(f'  Step {step+1}: {100*acc:.1f}%')
    
    results['lora_only'] = eval_model(model, 12)
    print(f'  >>> LoRA-Only: {100*results["lora_only"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('\n' + '='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    print(f'  Pre-trained Base:      {100*base_acc:.1f}%')
    print(f'  Joint Fine-tuning:     {100*results["joint"]:.1f}%')
    print(f'  LoRA-Only (Standard):  {100*results["lora_only"]:.1f}%')
    print(f'  Two-Phase Fine-tuning: {100*results["two_phase"]:.1f}%')
    print()
    
    if results['two_phase'] > results['joint']:
        print(f'  Two-Phase beats Joint by: +{100*(results["two_phase"] - results["joint"]):.1f}%')
    if results['two_phase'] > results['lora_only']:
        print(f'  Two-Phase beats LoRA-Only by: +{100*(results["two_phase"] - results["lora_only"]):.1f}%')
    
    print('\n' + '='*70)
    print('KEY INSIGHT')
    print('='*70)
    if results['two_phase'] > max(results['joint'], results['lora_only']):
        print('Two-Phase Training is superior for modular fine-tuning!')
    elif results['lora_only'] > results['joint']:
        print('Standard LoRA-only fine-tuning is optimal.')
        print('Two-Phase provides no benefit when base is already pre-trained.')
    else:
        print('Joint fine-tuning works well for this architecture.')

if __name__ == '__main__':
    main()
