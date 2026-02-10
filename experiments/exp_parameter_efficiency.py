#!/usr/bin/env python3
"""
Experiment: ANA vs Transformer Parameter Efficiency
Match parameters at different scales
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import random

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel

class ParameterEfficientTransformer(nn.Module):
    """Transformer designed to match ANA parameters closely"""
    def __init__(self, vocab_size, d_model, num_layers, n_heads):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(2048, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*2,  # Smaller FFN
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        h = self.embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = h + self.pos_embedding(positions)
        h = self.transformer(h)
        return self.output(h), {}

class MultiKVDataset(data.Dataset):
    def __init__(self, size=500, num_kv=8, min_noise=3, max_noise=10, vocab_size=50):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, vocab_size))
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

def train_eval(model, num_kv, epochs, lr):
    ds = MultiKVDataset(size=500, num_kv=num_kv, vocab_size=50)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    for _ in range(epochs):
        model.train()
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = (crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size()) * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
    
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

SCALES = [
    {'name': '50K', 'ana_config': {'d_model': 48, 'num_layers': 2, 'state_dim': 48}},
    {'name': '100K', 'ana_config': {'d_model': 64, 'num_layers': 2, 'state_dim': 64}},
    {'name': '200K', 'ana_config': {'d_model': 96, 'num_layers': 3, 'state_dim': 96}},
]

KV_COUNTS = [4, 8, 12]

print("="*70)
print("ANA vs TRANSFORMER PARAMETER EFFICIENCY")
print("="*70)

results = {}

for scale in SCALES:
    name = scale['name']
    ana_cfg = scale['ana_config']
    print(f"\n{'='*70}")
    print(f"Scale: {name}")
    print(f"{'='*70}")
    
    results[name] = {}
    
    for kv in KV_COUNTS:
        print(f"\n  {kv} KV pairs:")
        results[name][kv] = {}
        
        # Train ANA
        torch.manual_seed(42)
        random.seed(42)
        ana = ANAModel(ANAConfig(
            vocab_size=50,
            max_seq_len=2048,
            use_hololink=True,
            use_controller=True,
            **ana_cfg
        )).to(device)
        ana_params = sum(p.numel() for p in ana.parameters())
        ana_acc = train_eval(ana, kv, epochs=25, lr=1e-3)
        
        # Build matching Transformer
        xf_cfg = {
            'vocab_size': 50,
            'd_model': ana_cfg['d_model'],
            'num_layers': ana_cfg['num_layers'],
            'n_heads': max(2, ana_cfg['d_model'] // 32)
        }
        xformer = ParameterEfficientTransformer(**xf_cfg).to(device)
        xf_params = sum(p.numel() for p in xformer.parameters())
        xf_acc = train_eval(xformer, kv, epochs=25, lr=1e-3)
        
        results[name][kv] = {
            'ana_acc': ana_acc,
            'ana_params': ana_params,
            'xf_acc': xf_acc,
            'xf_params': xf_params,
            'advantage': ana_acc - xf_acc
        }
        
        print(f"    ANA:      {ana_acc*100:.1f}% ({ana_params:,} params)")
        print(f"    XF:       {xf_acc*100:.1f}% ({xf_params:,} params)")
        print(f"    Advantage: +{(ana_acc - xf_acc)*100:.1f}%")

with open('archive/experiments/parameter_efficiency.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("PARAMETER EFFICIENCY SUMMARY")
print("="*70)
print(f"{'Scale':>6} | {'KV':>3} | {'ANA':>6} | {'ANA Params':>10} | {'XF':>6} | {'XF Params':>10} | {'Adv':>6}")
print("-"*75)
for scale in SCALES:
    name = scale['name']
    for kv in KV_COUNTS:
        r = results[name][kv]
        print(f"{name:>6} | {kv:>3} | {r['ana_acc']*100:>5.1f}% | {r['ana_params']:>10,} | {r['xf_acc']*100:>5.1f}% | {r['xf_params']:>10,} | {r['advantage']*100:>+5.1f}%")

print(f"\nResults saved to: archive/experiments/parameter_efficiency.json")
