#!/usr/bin/env python3
"""
Phase A: Scaling Validation with Architecture Improvements
Test if layer norm and better residual connections fix scaling issues
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import random
import math

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig

class ImprovedLRU(nn.Module):
    """LRU with pre-norm and better initialization"""
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model, state_dim)
        self.output_proj = nn.Linear(state_dim, d_model)
        self.alpha_logit = nn.Parameter(torch.zeros(state_dim))
        self.beta_logit = nn.Parameter(torch.zeros(state_dim))
        nn.init.normal_(self.input_proj.weight, std=0.01)
        nn.init.normal_(self.output_proj.weight, std=0.01)
    
    def forward(self, x, h_prev=None):
        x_norm = self.norm(x)
        u = self.input_proj(x_norm)
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.state_dim, device=x.device)
        
        if x.dim() == 3:
            h_seq = []
            h = h_prev
            for t in range(x.size(1)):
                h = alpha * h + beta * u[:, t]
                h_seq.append(h)
            h_seq = torch.stack(h_seq, dim=1)
            y = self.output_proj(h_seq)
            return y, h_seq[:, -1]
        else:
            h = alpha * h_prev + beta * u
            y = self.output_proj(h)
            return y, h

class ImprovedController(nn.Module):
    def __init__(self, d_model, hidden_dim, num_tracks):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, num_tracks * 2 + 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.num_tracks = num_tracks
    
    def forward(self, x):
        h = self.norm(x)
        h = self.net(h)
        out = self.head(h)
        gates = out[..., :self.num_tracks*2].view(*out.shape[:-1], self.num_tracks, 2)
        ret_gate = out[..., -1:]
        return gates, ret_gate

class ImprovedHoloLink(nn.Module):
    def __init__(self, d_model, key_dim):
        super().__init__()
        self.d_model = d_model
        self.key_dim = key_dim
        self.norm = nn.LayerNorm(d_model)
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.decay = nn.Parameter(torch.zeros(1))
        nn.init.orthogonal_(self.k_proj.weight, gain=0.1)
        nn.init.orthogonal_(self.q_proj.weight, gain=0.1)
    
    def forward(self, x, h, M=None):
        x_norm = self.norm(x)
        k = F.normalize(self.k_proj(h), p=2, dim=-1)
        v = self.v_proj(h)
        decay = torch.sigmoid(self.decay) * 0.9 + 0.1
        
        is_3d = x.dim() == 3
        if is_3d:
            batch, seq_len, _ = x.shape
            k = k.view(batch * seq_len, -1)
            v = v.view(batch * seq_len, -1)
            x_flat = x_norm.view(batch * seq_len, -1)
        
        if M is None:
            if is_3d:
                M = torch.zeros(batch * seq_len, self.key_dim, self.d_model, device=x.device)
            else:
                M = torch.zeros(x.size(0), self.key_dim, self.d_model, device=x.device)
        
        update = torch.bmm(k.unsqueeze(-1), v.unsqueeze(-2))
        M = decay * M + update
        
        q = F.normalize(self.q_proj(x_flat if is_3d else x_norm), p=2, dim=-1)
        retrieved = torch.bmm(q.unsqueeze(1), M).squeeze(1)
        out = self.out_proj(retrieved)
        
        if is_3d:
            out = out.view(batch, seq_len, -1)
        return out, M

class ImprovedANA(nn.Module):
    """ANA with pre-norm architecture for better scaling"""
    def __init__(self, vocab_size, d_model, state_dim, num_layers, num_tracks, use_controller=True, use_hololink=True):
        super().__init__()
        self.use_controller = use_controller
        self.use_hololink = use_hololink
        self.num_tracks = num_tracks
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'tracks': nn.ModuleList([ImprovedLRU(d_model, state_dim) for _ in range(num_tracks)]),
                'track_norm': nn.LayerNorm(d_model),
            })
            if use_controller:
                layer['controller'] = ImprovedController(d_model, d_model, num_tracks)
            if use_hololink:
                layer['holo'] = ImprovedHoloLink(d_model, d_model)
            self.layers.append(layer)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        h = self.embed_norm(self.embedding(x))
        
        for layer in self.layers:
            residual = h
            h_norm = layer['track_norm'](h)
            
            track_outs = []
            for track in layer['tracks']:
                out, _ = track(h_norm)
                track_outs.append(out)
            
            track_combined = sum(track_outs) / len(track_outs)
            
            if self.use_controller:
                gates, ret_gate = layer['controller'](h)
                mix_weights = F.softmax(gates[..., 0], dim=-1)
                track_combined = sum(w.unsqueeze(-1) * o for w, o in zip(mix_weights.unbind(-1), track_outs))
            
            if self.use_hololink:
                holo_out, _ = layer['holo'](h, track_combined)
                if self.use_controller:
                    ret = torch.sigmoid(ret_gate)
                    track_combined = track_combined + ret * holo_out
                else:
                    track_combined = track_combined + holo_out
            
            h = residual + track_combined
        
        return self.output(self.final_norm(h)), {}

class QuickMultiKV(data.Dataset):
    def __init__(self, size=400, num_kv=8, min_noise=3, max_noise=10):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, 30))
        for _ in range(size):
            kvs = [(random.choice(content), random.choice(content)) for _ in range(num_kv)]
            seq = []
            for k, v in kvs:
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            seq.extend([random.choice(content) for _ in range(random.randint(min_noise, max_noise))])
            ti = random.randint(0, num_kv-1)
            seq.extend([TOK_QUERY, kvs[ti][0], kvs[ti][1]])
            x = torch.tensor(seq[:-1])
            y = torch.tensor(seq[1:])
            m = torch.ones_like(y, dtype=torch.float) * 0.01
            m[-1] = 1.0
            self.data.append((x, y, m))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    xs, ys, ms = zip(*batch)
    ml = max(x.size(0) for x in xs)
    return (torch.stack([F.pad(x, (0, ml-x.size(0))) for x in xs]),
            torch.stack([F.pad(y, (0, ml-y.size(0))) for y in ys]),
            torch.stack([F.pad(m, (0, ml-m.size(0))) for m in ms]))

def train_eval(model, num_kv, epochs=20, lr=2e-4):
    ds = QuickMultiKV(size=600, num_kv=num_kv)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    for epoch in range(epochs):
        model.train()
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = (crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size()) * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, _ = model(x)
            for i in range(x.size(0)):
                pos = (m[i] > 0.5).nonzero(as_tuple=True)[0][0]
                if logits[i, pos].argmax().item() == y[i, pos].item():
                    correct += 1
                total += 1
    return correct / total if total else 0

SCALING_CONFIGS = [
    {'name': 'small', 'd_model': 64, 'num_layers': 2, 'state_dim': 64},
    {'name': 'medium', 'd_model': 128, 'num_layers': 3, 'state_dim': 128},
    {'name': 'large', 'd_model': 256, 'num_layers': 4, 'state_dim': 256},
]

CONFIGS = {
    'baseline': {'use_controller': False, 'use_hololink': False},
    'controller': {'use_controller': True, 'use_hololink': False},
    'hololink': {'use_controller': False, 'use_hololink': True},
    'full': {'use_controller': True, 'use_hololink': True},
}

print("="*70)
print("PHASE A: IMPROVED SCALING VALIDATION")
print("="*70)

results = {}

for scale_cfg in SCALING_CONFIGS:
    name = scale_cfg['name']
    print(f"\n{'='*70}")
    print(f"Scale: {name} (d_model={scale_cfg['d_model']}, layers={scale_cfg['num_layers']})")
    print(f"{'='*70}")
    
    results[name] = {'config': scale_cfg, 'ablations': {}}
    
    for ablation_name, flags in CONFIGS.items():
        print(f"\n  Testing {ablation_name}...")
        accs = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            random.seed(seed)
            
            model = ImprovedANA(
                vocab_size=30,
                d_model=scale_cfg['d_model'],
                state_dim=scale_cfg['state_dim'],
                num_layers=scale_cfg['num_layers'],
                num_tracks=2,
                **flags
            ).to(device)
            
            params = sum(p.numel() for p in model.parameters())
            acc = train_eval(model, num_kv=8, epochs=25)
            accs.append(acc)
        
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
        results[name]['ablations'][ablation_name] = {
            'mean': mean_acc,
            'std': std_acc,
            'runs': accs,
            'params': params
        }
        print(f"    {ablation_name}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}% ({params:,} params)")
    
    single_best = max(results[name]['ablations'][c]['mean'] for c in ['controller', 'hololink'])
    full_mean = results[name]['ablations']['full']['mean']
    synergy = full_mean - single_best
    results[name]['synergy'] = synergy
    print(f"\n  Synergy: +{synergy*100:.1f}%")

with open('archive/experiments/phaseA_scaling_improved.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("IMPROVED SCALING SUMMARY")
print("="*70)
print(f"{'Scale':<10} {'Full ANA':>12} {'Single Best':>12} {'Synergy':>10}")
print("-"*50)
for name, data in results.items():
    full = data['ablations']['full']['mean']
    best = max(data['ablations'][c]['mean'] for c in ['controller', 'hololink'])
    print(f"{name:<10} {full*100:>11.1f}% {best*100:>11.1f}% {data['synergy']*100:>+9.1f}%")

print(f"\nResults saved to: archive/experiments/phaseA_scaling_improved.json")
