"""
Hybrid Architecture Experiments

Tests the hypothesis that a hybrid ANA-Transformer with learned routing
achieves better performance than either pure architecture alone.

Key Experiments:
1. Mixed associative + pattern tasks
2. Routing analysis and entropy
3. Error analysis by route
4. Comparison to pure ANA and pure Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ana.hybrid import (
    HybridANATransformer, HybridWithSpecialization, 
    MultiRouterHybrid, LearnableRouter
)
from ana.model_v3 import ANAv2Model
from ana.models_v3 import SpecializedTracks
from ana.config_v2 import ANAv2Config


class MixedAssociativePatternDataset(Dataset):
    def __init__(self, num_samples=1000, vocab_size=50, seq_len=128, 
                 associative_ratio=0.5, kv_pairs=4):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.associative_ratio = associative_ratio
        self.kv_pairs = kv_pairs
        
        self.data = self._generate_data()
    
    def _generate_associative_sample(self):
        keys = torch.randint(1, self.vocab_size // 2, (self.kv_pairs,))
        values = torch.randint(self.vocab_size // 2, self.vocab_size, (self.kv_pairs,))
        
        kv_sequence = []
        for k, v in zip(keys, values):
            kv_sequence.extend([k.item(), v.item()])
        
        noise_length = np.random.randint(5, 20)
        noise = torch.randint(1, self.vocab_size, (noise_length,)).tolist()
        
        query_key = keys[np.random.randint(self.kv_pairs)].item()
        target_value = values[keys.tolist().index(query_key)].item()
        
        sequence = kv_sequence + noise + [query_key]
        sequence = sequence[:self.seq_len - 1] + [target_value]
        
        input_ids = torch.tensor(sequence[:self.seq_len], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:self.seq_len + 1], dtype=torch.long)
        
        return input_ids, target_ids, 'associative'
    
    def _generate_pattern_sample(self):
        # Repeating pattern task
        pattern_length = np.random.randint(3, 8)
        pattern = torch.randint(1, self.vocab_size // 2, (pattern_length,)).tolist()
        
        repeats = np.random.randint(2, 5)
        sequence = pattern * repeats
        
        # Add some noise
        noise_positions = np.random.choice(len(sequence), min(3, len(sequence)), replace=False)
        for pos in noise_positions:
            sequence[pos] = np.random.randint(1, self.vocab_size)
        
        # Predict next in pattern
        next_token = pattern[np.random.randint(len(pattern))]
        sequence.append(next_token)
        
        input_ids = torch.tensor(sequence[:self.seq_len], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:self.seq_len + 1], dtype=torch.long)
        
        return input_ids, target_ids, 'pattern'
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            if np.random.random() < self.associative_ratio:
                data.append(self._generate_associative_sample())
            else:
                data.append(self._generate_pattern_sample())
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx][:2]


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=50, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer(x)
        return self.output(x)


class SimpleANA(nn.Module):
    def __init__(self, vocab_size=50, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simple tracks
        self.track1 = nn.Linear(d_model, d_model)
        self.track2 = nn.Linear(d_model, d_model)
        self.track3 = nn.Linear(d_model, d_model)
        
        self.mixer = nn.Linear(d_model * 3, d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        
        outputs = []
        for t in range(x.size(1)):
            t1 = torch.sigmoid(self.track1(h)) * h + torch.sigmoid(self.track2(x[:, t])) * x[:, t]
            t2 = torch.sigmoid(self.track3(x[:, t])) * x[:, t]
            combined = torch.cat([t1, t2, h], dim=-1)
            h = self.mixer(combined)
            outputs.append(h)
        
        out = torch.stack(outputs, dim=1)
        return self.output(out)


def collate_fn(batch):
    max_len = max(item[0].size(0) for item in batch)
    padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (inp, tgt) in enumerate(batch):
        padded_inputs[i, :inp.size(0)] = inp
        padded_targets[i, :tgt.size(0)] = tgt
    
    return padded_inputs, padded_targets


def train_model(model, train_loader, val_loader, epochs=20, device='cuda', model_name='Model'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=0)
                val_loss += loss.item()
                
                predictions = outputs.argmax(-1)
                mask = targets != 0
                correct += (predictions[mask] == targets[mask]).sum().item()
                total += mask.sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"{model_name} Epoch {epoch+1}/{epochs}: "
                  f"Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={val_acc:.2%}")
        
        scheduler.step()
    
    return history


def analyze_routing(hybrid_model, val_loader, device='cuda'):
    hybrid_model.eval()
    
    route_usage = defaultdict(list)
    error_by_route = defaultdict(list)
    position_usage = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, route_weights = hybrid_model(inputs, return_routing=True)
            
            predictions = logits.argmax(-1)
            errors = (predictions != targets).float()
            
            # Track route usage
            avg_route = route_weights.mean(dim=(0, 1)).cpu().numpy()
            route_usage['route_0'].append(avg_route[0])
            route_usage['route_1'].append(avg_route[1])
            
            # Track errors per route
            for r in range(route_weights.size(-1)):
                route_errors = (errors * route_weights[:, :, r]).sum() / (route_weights[:, :, r].sum() + 1e-10)
                error_by_route[f'route_{r}'].append(route_errors.item())
            
            # Track position-based routing
            pos_usage = route_weights.mean(dim=0).cpu().numpy()
            position_usage.append(pos_usage)
    
    results = {
        'route_0_usage': np.mean(route_usage['route_0']),
        'route_1_usage': np.mean(route_usage['route_1']),
        'route_0_error': np.mean(error_by_route['route_0']),
        'route_1_error': np.mean(error_by_route['route_1']),
        'position_usage': np.mean(position_usage, axis=0).tolist()
    }
    
    return results


def run_hybrid_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets
    train_dataset = MixedAssociativePatternDataset(num_samples=2000, vocab_size=50, seq_len=128)
    val_dataset = MixedAssociativePatternDataset(num_samples=500, vocab_size=50, seq_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    print("=" * 60)
    print("Hybrid ANA-Transformer Experiment")
    print("=" * 60)
    
    results = {}
    
    # 1. Pure Transformer
    print("\n1. Training Pure Transformer...")
    xf_model = SimpleTransformer(vocab_size=50, d_model=128, nhead=4, num_layers=2)
    xf_history = train_model(xf_model, train_loader, val_loader, epochs=20, device=device, model_name='Transformer')
    results['transformer'] = xf_history
    
    # 2. Pure ANA
    print("\n2. Training Pure ANA...")
    ana_model = SimpleANA(vocab_size=50, d_model=128)
    ana_history = train_model(ana_model, train_loader, val_loader, epochs=20, device=device, model_name='ANA')
    results['ana'] = ana_history
    
    # 3. Hybrid with Standard Router
    print("\n3. Training Hybrid (Standard Router)...")
    config = ANAv2Config(
        vocab_size=50,
        d_model=128,
        max_seq_len=128,
        use_position_encoding=True
    )
    hybrid_model = HybridANATransformer(config, num_layers=2, nhead=4)
    hybrid_history = train_model(hybrid_model, train_loader, val_loader, epochs=20, device=device, model_name='Hybrid')
    results['hybrid'] = hybrid_history
    
    # 4. Analyze routing
    print("\n4. Analyzing Routing Decisions...")
    routing_analysis = analyze_routing(hybrid_model, val_loader, device)
    results['routing'] = routing_analysis
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for name, history in results.items():
        if name == 'routing':
            continue
        final_acc = history['val_acc'][-1]
        print(f"\n{name.upper()}:")
        print(f"  Final Accuracy: {final_acc:.2%}")
        print(f"  Best Accuracy: {max(history['val_acc']):.2%}")
    
    print(f"\nROUTING ANALYSIS:")
    print(f"  Route 0 (ANA) Usage: {routing_analysis['route_0_usage']:.1%}")
    print(f"  Route 1 (Transformer) Usage: {routing_analysis['route_1_usage']:.1%}")
    print(f"  Route 0 Error Rate: {routing_analysis['route_0_error']:.2%}")
    print(f"  Route 1 Error Rate: {routing_analysis['route_1_error']:.2%}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'experiments' / 'hybrid'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['transformer']['val_acc'], label='Transformer')
    plt.plot(results['ana']['val_acc'], label='ANA')
    plt.plot(results['hybrid']['val_acc'], label='Hybrid')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    positions = np.array(routing_analysis['position_usage'])
    plt.plot(positions[:, 0], label='ANA Route')
    plt.plot(positions[:, 1], label='Transformer Route')
    plt.xlabel('Position')
    plt.ylabel('Route Weight')
    plt.title('Routing by Position')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_analysis.png', dpi=150)
    
    print(f"\n✓ Results saved to: experiments/hybrid/")
    
    return results


if __name__ == '__main__':
    results = run_hybrid_experiment()
    print("\n✓ Hybrid experiment complete!")
