#!/usr/bin/env python
"""
Cross-Architecture Validation: Does Two-Phase Training Generalize?

Tests the gradient interference hypothesis on:
1. Transformer + LoRA adapters
2. Simple MoE (Mixture of Experts)
3. Multi-head attention with learnable gating

Goal: Prove this is a universal principle, not ANA-specific.
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

# ============================================================================
# EXPERIMENT 1: Transformer + LoRA Adapters
# ============================================================================

@dataclass
class LoRAConfig:
    vocab_size: int = 60
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    lora_rank: int = 8
    lora_alpha: float = 16.0

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

class TransformerBlock(nn.Module):
    def __init__(self, config: LoRAConfig, use_lora=False):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        if use_lora:
            self.q_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
            self.k_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
            self.v_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
            self.o_proj = LoRALinear(config.d_model, config.d_model, config.lora_rank, config.lora_alpha)
            self.ff_up = LoRALinear(config.d_model, config.d_model * 4, config.lora_rank, config.lora_alpha)
            self.ff_down = LoRALinear(config.d_model * 4, config.d_model, config.lora_rank, config.lora_alpha)
        else:
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.ff_up = nn.Linear(config.d_model, config.d_model * 4, bias=False)
            self.ff_down = nn.Linear(config.d_model * 4, config.d_model, bias=False)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
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
    def __init__(self, config: LoRAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.base_layers = nn.ModuleList([
            TransformerBlock(config, use_lora=False) for _ in range(config.n_layers)
        ])
        self.lora_layers = nn.ModuleList([
            TransformerBlock(config, use_lora=True) for _ in range(config.n_layers)
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
    
    def freeze_lora(self):
        for layer in self.lora_layers:
            for p in layer.parameters():
                p.requires_grad = False
    
    def unfreeze_lora(self):
        for layer in self.lora_layers:
            for p in layer.parameters():
                p.requires_grad = True
    
    def freeze_base(self):
        for layer in self.base_layers:
            for p in layer.parameters():
                p.requires_grad = False
    
    def unfreeze_base(self):
        for layer in self.base_layers:
            for p in layer.parameters():
                p.requires_grad = True
    
    def get_lora_params(self):
        return [p for layer in self.lora_layers for p in layer.parameters()]
    
    def get_base_params(self):
        params = [p for layer in self.base_layers for p in layer.parameters()]
        params.extend(self.embedding.parameters())
        params.extend(self.pos_emb.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.head.parameters())
        return params

def test_transformer_lora():
    print('\n' + '='*70)
    print('EXPERIMENT 1: Transformer + LoRA Adapters')
    print('='*70)
    
    config = LoRAConfig()
    results = {}
    
    curriculum = [(1, 400), (2, 400), (4, 400), (6, 400), (8, 400), (10, 400), (12, 500)]
    
    # Test 1: Base only (no LoRA adapters)
    print('\n[1/3] Base Transformer Only (LoRA adapters disabled)...')
    model = TransformerWithLoRA(config).to(device)
    for layer in model.lora_layers:
        for p in layer.parameters():
            p.requires_grad = False
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    results['base_only'] = eval_model(model, 12)
    print(f'  Base Only: {100*results["base_only"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # Test 2: Joint Training (Base + LoRA together)
    print('\n[2/3] Joint Training (Base + LoRA together)...')
    model = TransformerWithLoRA(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    results['joint'] = eval_model(model, 12)
    print(f'  Joint Training: {100*results["joint"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # Test 3: Two-Phase Training
    print('\n[3/3] Two-Phase Training...')
    model = TransformerWithLoRA(config).to(device)
    
    # Phase 1: Train base, freeze LoRA
    print('  Phase 1: Training Base (LoRA frozen)...')
    model.freeze_lora()
    opt = torch.optim.Adam(model.get_base_params(), lr=1e-3)
    
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    phase1_acc = eval_model(model, 12)
    print(f'  Phase 1 (Base): {100*phase1_acc:.1f}%')
    
    # Phase 2: Freeze base, train LoRA
    print('  Phase 2: Training LoRA (Base frozen)...')
    model.freeze_base()
    model.unfreeze_lora()
    opt = torch.optim.Adam(model.get_lora_params(), lr=1e-4)
    
    for _ in range(500):
        bx, by = gen_kv_data(32, 12)
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        logits = model(bx)
        F.cross_entropy(logits[:, -1, :], by).backward()
        opt.step()
    
    results['two_phase'] = eval_model(model, 12)
    print(f'  Phase 2 (+LoRA): {100*results["two_phase"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    return results

# ============================================================================
# EXPERIMENT 2: Mixture of Experts (MoE)
# ============================================================================

@dataclass
class MoEConfig:
    vocab_size: int = 60
    d_model: int = 64
    d_ff: int = 256
    n_experts: int = 4
    top_k: int = 2
    n_layers: int = 2
    max_seq_len: int = 128

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class Router(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts)
    
    def forward(self, x):
        return self.proj(x)

class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, config.d_ff) for _ in range(config.n_experts)
        ])
        self.router = Router(config.d_model, config.n_experts)
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        B, S, D = x.shape
        h = self.norm(x)
        
        router_logits = self.router(h)
        top_k_vals, top_k_idx = torch.topk(router_logits, self.config.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_vals, dim=-1)
        
        out = torch.zeros_like(x)
        for i in range(self.config.top_k):
            expert_idx = top_k_idx[..., i]
            weight = top_k_weights[..., i:i+1]
            
            for e in range(self.config.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    selected = h[mask]
                    expert_out = self.experts[e](selected)
                    out[mask] += weight[mask] * expert_out
        
        return x + out
    
    def get_expert_params(self):
        return [p for e in self.experts for p in e.parameters()]
    
    def get_router_params(self):
        return list(self.router.parameters())

class MoEModel(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([MoELayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_emb(torch.arange(S, device=input_ids.device))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.head(x)
    
    def freeze_experts(self):
        for layer in self.layers:
            for p in layer.get_expert_params():
                p.requires_grad = False
    
    def unfreeze_experts(self):
        for layer in self.layers:
            for p in layer.get_expert_params():
                p.requires_grad = True
    
    def freeze_router(self):
        for layer in self.layers:
            for p in layer.get_router_params():
                p.requires_grad = False
    
    def unfreeze_router(self):
        for layer in self.layers:
            for p in layer.get_router_params():
                p.requires_grad = True
    
    def get_expert_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_expert_params())
        return params
    
    def get_router_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_router_params())
        return params
    
    def get_other_params(self):
        expert_params = set(self.get_expert_params())
        router_params = set(self.get_router_params())
        return [p for p in self.parameters() if p not in expert_params and p not in router_params]

def test_moe():
    print('\n' + '='*70)
    print('EXPERIMENT 2: Mixture of Experts (MoE)')
    print('='*70)
    
    config = MoEConfig()
    results = {}
    
    curriculum = [(1, 400), (2, 400), (4, 400), (6, 400), (8, 400), (10, 400), (12, 500)]
    
    # Test 1: Experts only (random router)
    print('\n[1/3] Experts Only (Router disabled)...')
    model = MoEModel(config).to(device)
    model.freeze_router()
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    results['experts_only'] = eval_model(model, 12)
    print(f'  Experts Only: {100*results["experts_only"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # Test 2: Joint Training
    print('\n[2/3] Joint Training (Experts + Router together)...')
    model = MoEModel(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    results['joint'] = eval_model(model, 12)
    print(f'  Joint Training: {100*results["joint"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # Test 3: Two-Phase Training
    print('\n[3/3] Two-Phase Training...')
    model = MoEModel(config).to(device)
    
    # Phase 1: Train experts, freeze router
    print('  Phase 1: Training Experts (Router frozen)...')
    model.freeze_router()
    params = list(model.get_expert_params()) + list(model.get_other_params())
    opt = torch.optim.Adam(params, lr=1e-3)
    
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    phase1_acc = eval_model(model, 12)
    print(f'  Phase 1 (Experts): {100*phase1_acc:.1f}%')
    
    # Phase 2: Freeze experts, train router
    print('  Phase 2: Training Router (Experts frozen)...')
    model.freeze_experts()
    model.unfreeze_router()
    opt = torch.optim.Adam(model.get_router_params(), lr=1e-4)
    
    for _ in range(500):
        bx, by = gen_kv_data(32, 12)
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        logits = model(bx)
        F.cross_entropy(logits[:, -1, :], by).backward()
        opt.step()
    
    results['two_phase'] = eval_model(model, 12)
    print(f'  Phase 2 (+Router): {100*results["two_phase"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    return results

# ============================================================================
# EXPERIMENT 3: Multi-Head Attention with Learnable Gating
# ============================================================================

@dataclass
class GatedMHAConfig:
    vocab_size: int = 60
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128

class GatedMHA(nn.Module):
    def __init__(self, config: GatedMHAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.head_gates = nn.Parameter(torch.zeros(config.n_heads))
        
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        B, S, D = x.shape
        h = self.norm(x)
        
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        gates = torch.sigmoid(self.head_gates).view(1, -1, 1, 1)
        attn = gates * attn
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        out = self.o_proj(out)
        return x + out
    
    def get_attention_params(self):
        return [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.o_proj.weight, self.norm.weight, self.norm.bias]
    
    def get_gate_params(self):
        return [self.head_gates]

class GatedMHAModel(nn.Module):
    def __init__(self, config: GatedMHAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([GatedMHA(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_emb(torch.arange(S, device=input_ids.device))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.head(x)
    
    def freeze_gates(self):
        for layer in self.layers:
            for p in layer.get_gate_params():
                p.requires_grad = False
    
    def unfreeze_gates(self):
        for layer in self.layers:
            for p in layer.get_gate_params():
                p.requires_grad = True
    
    def freeze_attention(self):
        for layer in self.layers:
            for p in layer.get_attention_params():
                p.requires_grad = False
    
    def unfreeze_attention(self):
        for layer in self.layers:
            for p in layer.get_attention_params():
                p.requires_grad = True
    
    def get_attention_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_attention_params())
        params.extend(self.embedding.parameters())
        params.extend(self.pos_emb.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.head.parameters())
        return params
    
    def get_gate_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_gate_params())
        return params

def test_gated_mha():
    print('\n' + '='*70)
    print('EXPERIMENT 3: Multi-Head Attention with Learnable Gating')
    print('='*70)
    
    config = GatedMHAConfig()
    results = {}
    
    curriculum = [(1, 400), (2, 400), (4, 400), (6, 400), (8, 400), (10, 400), (12, 500)]
    
    # Test 1: Attention only (no gating)
    print('\n[1/3] Attention Only (Gates disabled)...')
    model = GatedMHAModel(config).to(device)
    model.freeze_gates()
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    results['attn_only'] = eval_model(model, 12)
    print(f'  Attention Only: {100*results["attn_only"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # Test 2: Joint Training
    print('\n[2/3] Joint Training (Attention + Gates together)...')
    model = GatedMHAModel(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    results['joint'] = eval_model(model, 12)
    print(f'  Joint Training: {100*results["joint"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    # Test 3: Two-Phase Training
    print('\n[3/3] Two-Phase Training...')
    model = GatedMHAModel(config).to(device)
    
    # Phase 1: Train attention, freeze gates
    print('  Phase 1: Training Attention (Gates frozen)...')
    model.freeze_gates()
    opt = torch.optim.Adam(model.get_attention_params(), lr=1e-3)
    
    for pairs, steps in curriculum:
        for _ in range(steps):
            bx, by = gen_kv_data(32, pairs)
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            F.cross_entropy(logits[:, -1, :], by).backward()
            opt.step()
    
    phase1_acc = eval_model(model, 12)
    print(f'  Phase 1 (Attention): {100*phase1_acc:.1f}%')
    
    # Phase 2: Freeze attention, train gates
    print('  Phase 2: Training Gates (Attention frozen)...')
    model.freeze_attention()
    model.unfreeze_gates()
    opt = torch.optim.Adam(model.get_gate_params(), lr=1e-4)
    
    for _ in range(500):
        bx, by = gen_kv_data(32, 12)
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        logits = model(bx)
        F.cross_entropy(logits[:, -1, :], by).backward()
        opt.step()
    
    results['two_phase'] = eval_model(model, 12)
    print(f'  Phase 2 (+Gates): {100*results["two_phase"]:.1f}%')
    del model
    torch.cuda.empty_cache()
    
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('\n' + '='*70)
    print('CROSS-ARCHITECTURE VALIDATION')
    print('Testing: Does Two-Phase Training Generalize?')
    print('='*70)
    
    all_results = {}
    
    all_results['transformer_lora'] = test_transformer_lora()
    all_results['moe'] = test_moe()
    all_results['gated_mha'] = test_gated_mha()
    
    print('\n' + '='*70)
    print('FINAL RESULTS SUMMARY')
    print('='*70)
    
    improvements = []
    
    for name, results in all_results.items():
        print(f'\n{name.upper().replace("_", " ")}:')
        
        keys = list(results.keys())
        if 'base_only' in keys:
            baseline = results['base_only']
            joint = results['joint']
            two_phase = results['two_phase']
            print(f'  Base/Experts Only: {100*baseline:.1f}%')
        elif 'experts_only' in keys:
            baseline = results['experts_only']
            joint = results['joint']
            two_phase = results['two_phase']
            print(f'  Base/Experts Only: {100*baseline:.1f}%')
        else:
            baseline = results['attn_only']
            joint = results['joint']
            two_phase = results['two_phase']
            print(f'  Attention Only: {100*baseline:.1f}%')
        
        print(f'  Joint Training:    {100*joint:.1f}%')
        print(f'  Two-Phase:         {100*two_phase:.1f}%')
        
        if two_phase > joint:
            improvement = two_phase - joint
            improvements.append(improvement)
            print(f'  Two-Phase Improvement: +{100*improvement:.1f}%')
        else:
            print(f'  Two-Phase Difference: {100*(two_phase - joint):.1f}%')
    
    print('\n' + '='*70)
    print('CONCLUSION')
    print('='*70)
    
    if sum(1 for r in all_results.values() if r['two_phase'] > r['joint']) >= 2:
        print('SUCCESS: Two-Phase Training generalizes across architectures!')
        print('This confirms it is a universal principle, not ANA-specific.')
    else:
        print('MIXED: Results vary by architecture.')
        print('Further investigation needed.')
