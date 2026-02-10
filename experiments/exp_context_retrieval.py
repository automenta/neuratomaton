#!/usr/bin/env python3
"""
Experiment: Real-world Style Task - Context Window Memory
Test ANA on a task that mimics real information retrieval from context
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import random
import re

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel

class ContextRetrievalDataset(data.Dataset):
    """
    Simulates a real-world task: retrieving facts from a document context
    Structure: [KEY] entity [VAL] fact ... noise ... [QUERY] entity -> predict fact
    """
    def __init__(self, size=500, num_facts=6):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        
        # Simulated entities and facts
        entities = list(range(5, 25))  # 20 entities
        facts = list(range(25, 50))     # 25 possible facts
        
        for _ in range(size):
            entity_facts = [(random.choice(entities), random.choice(facts)) for _ in range(num_facts)]
            
            # Build context
            seq = []
            for entity, fact in entity_facts:
                seq.extend([TOK_KEY, entity, TOK_VAL, fact])
            
            # Add noise (distractor content)
            noise_len = random.randint(5, 15)
            seq.extend([random.choice(list(range(5, 50))) for _ in range(noise_len)])
            
            # Query for a random entity
            query_entity = random.choice(entity_facts)[0]
            query_fact = next(f for e, f in entity_facts if e == query_entity)
            seq.extend([TOK_QUERY, query_entity])
            
            x = torch.tensor(seq[:-1])
            y = torch.tensor(seq[1:])
            
            # Mask: only care about the final prediction
            m = torch.zeros_like(y, dtype=torch.float)
            m[-1] = 1.0
            
            self.data.append((x, y, m))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    xs, ys, ms = zip(*batch)
    ml = max(x.size(0) for x in xs)
    return (torch.stack([F.pad(x, (0, ml-x.size(0)), value=0) for x in xs]),
            torch.stack([F.pad(y, (0, ml-y.size(0)), value=0) for y in ys]),
            torch.stack([F.pad(m, (0, ml-m.size(0)), value=0) for m in ms]))

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        h = self.embedding(x)
        h = h + self.pos_embedding(torch.arange(seq_len, device=x.device).unsqueeze(0))
        return self.transformer(h), {}

def train_eval(model, num_facts, epochs=25, lr=1e-3):
    ds = ContextRetrievalDataset(size=400, num_facts=num_facts)
    loader = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    for _ in range(epochs):
        model.train()
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = (crit(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size()) * m).sum() / m.sum().clamp(min=1)
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
                positions = (m[i] > 0.5).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    pos = positions[0]
                    if pos < logits.size(1):
                        pred = logits[i, pos].argmax().item()
                        target = y[i, pos].item()
                        if pred == target and target != 0:
                            correct += 1
                        total += 1
    return correct / total if total > 0 else 0

print("="*70)
print("REAL-WORLD STYLE TASK: Context Window Memory Retrieval")
print("="*70)

FACT_COUNTS = [4, 6, 8, 10]

results = {}

for num_facts in FACT_COUNTS:
    print(f"\n{'='*70}")
    print(f"Retrieving from {num_facts} facts in context")
    print(f"{'='*70}")
    
    results[num_facts] = {}
    
    # ANA (ultra-efficient)
    print(f"\n  ANA (small):")
    torch.manual_seed(42)
    random.seed(42)
    ana_small = ANAModel(ANAConfig(
        d_model=48, num_layers=1, state_dim=48,
        vocab_size=50, max_seq_len=512,
        use_hololink=True, use_controller=True
    )).to(device)
    ana_small_params = sum(p.numel() for p in ana_small.parameters())
    ana_small_acc = train_eval(ana_small, num_facts, epochs=30)
    print(f"    {ana_small_acc*100:.1f}% ({ana_small_params:,} params)")
    results[num_facts]['ana_small'] = {'acc': ana_small_acc, 'params': ana_small_params}
    
    # Transformer (matched params)
    xf_cfg = {'vocab_size': 50, 'd_model': 24, 'num_layers': 1, 'n_heads': 2}
    xf_tiny = TinyTransformer(**xf_cfg).to(device)
    xf_tiny_params = sum(p.numel() for p in xf_tiny.parameters())
    xf_tiny_acc = train_eval(xf_tiny, num_facts, epochs=30)
    print(f"  Transformer (tiny): {xf_tiny_acc*100:.1f}% ({xf_tiny_params:,} params)")
    results[num_facts]['xf_tiny'] = {'acc': xf_tiny_acc, 'params': xf_tiny_params}
    
    advantage = ana_small_acc - xf_tiny_acc
    print(f"  Advantage: +{advantage*100:.1f}%")

with open('archive/experiments/context_retrieval.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("CONTEXT RETRIEVAL SUMMARY")
print("="*70)
print(f"{'Facts':>6} | {'ANA':>8} | {'ANA P':>8} | {'XF':>8} | {'XF P':>8} | {'Adv':>8}")
print("-"*58)
for num_facts in FACT_COUNTS:
    ana = results[num_facts]['ana_small']
    xf = results[num_facts]['xf_tiny']
    print(f"{num_facts:>6} | {ana['acc']*100:>7.1f}% | {ana['params']:>8,} | {xf['acc']*100:>7.1f}% | {xf['params']:>8,} | {(ana['acc']-xf['acc'])*100:>7.1f}%")

print(f"\nResults saved to: archive/experiments/context_retrieval.json")
