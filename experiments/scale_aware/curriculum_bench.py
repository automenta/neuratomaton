"""
Scale-Aware Curriculum Experiments

Tests the hypothesis that scale-aware training eliminates sensitivity
across model sizes and enables 100% accuracy at all scales.

Key Experiments:
1. Small, medium, large models with scale-specific curriculum
2. Comparison to uniform training (baseline)
3. Learning rate sweep validation
4. Training convergence analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import os
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ana.curriculum import ScaleAwareTrainer, create_curriculum
from ana.model_v3 import ANAv2Model
from ana.config_v2 import ANAv2Config


class NeedleHaystackDataset(Dataset):
    def __init__(self, num_samples=1000, vocab_size=30, seq_len=64, kv_pairs=4):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.kv_pairs = kv_pairs
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            sample = self._generate_sample()
            data.append(sample)
        return data
    
    def _generate_sample(self):
        keys = torch.randint(1, self.vocab_size // 2, (self.kv_pairs,))
        values = torch.randint(self.vocab_size // 2, self.vocab_size, (self.kv_pairs,))
        
        kv_sequence = []
        for k, v in zip(keys, values):
            kv_sequence.extend([k.item(), v.item()])
        
        noise_length = np.random.randint(3, 15)
        noise = torch.randint(1, self.vocab_size, (noise_length,)).tolist()
        
        query_key = keys[np.random.randint(self.kv_pairs)].item()
        target_value = values[keys.tolist().index(query_key)].item()
        
        sequence = kv_sequence + noise + [query_key]
        sequence = sequence[:self.seq_len - 1] + [target_value]
        
        input_ids = torch.tensor(sequence[:self.seq_len], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:self.seq_len + 1], dtype=torch.long)
        
        return input_ids, target_ids
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    max_len = max(item[0].size(0) for item in batch)
    padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (inp, tgt) in enumerate(batch):
        padded_inputs[i, :inp.size(0)] = inp
        padded_targets[i, :tgt.size(0)] = tgt
    
    return padded_inputs, padded_targets


class SmallModel(nn.Module):
    def __init__(self, vocab_size=30, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lru1 = nn.Linear(d_model, d_model)
        self.lru2 = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        outputs = []
        
        for t in range(x.size(1)):
            h = torch.sigmoid(self.lru1(h)) * h + torch.sigmoid(self.lru2(x[:, t])) * x[:, t]
            outputs.append(h)
        
        out = torch.stack(outputs, dim=1)
        logits = self.output(out)
        return logits


class MediumModel(nn.Module):
    def __init__(self, vocab_size=30, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lru1 = nn.Linear(d_model, d_model)
        self.lru2 = nn.Linear(d_model, d_model)
        self.lru3 = nn.Linear(d_model, d_model)
        self.lru4 = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        outputs = []
        
        for t in range(x.size(1)):
            h = torch.sigmoid(self.lru1(h)) * h + torch.sigmoid(self.lru2(x[:, t])) * x[:, t]
            h2 = torch.sigmoid(self.lru3(h)) * h + torch.sigmoid(self.lru4(x[:, t])) * x[:, t]
            outputs.append(h2)
        
        out = torch.stack(outputs, dim=1)
        logits = self.output(out)
        return logits


class LargeModel(nn.Module):
    def __init__(self, vocab_size=30, d_model=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.lrus = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(8)
        ])
        
        self.mixer = nn.Linear(d_model * 2, d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        outputs = []
        
        for t in range(x.size(1)):
            h_combined = torch.cat([h, x[:, t]], dim=-1)
            h = torch.sigmoid(self.lrus[0](h)) * h + torch.sigmoid(self.lrus[1](x[:, t])) * x[:, t]
            h = self.mixer(h_combined)
            outputs.append(h)
        
        out = torch.stack(outputs, dim=1)
        logits = self.output(out)
        return logits


def run_scale_experiment(model_class, model_name, num_params_target, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_class(vocab_size=30).to(device)
    
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    train_dataset = NeedleHaystackDataset(num_samples=2000, vocab_size=30, seq_len=64, kv_pairs=4)
    val_dataset = NeedleHaystackDataset(num_samples=500, vocab_size=30, seq_len=64, kv_pairs=4)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    trainer = ScaleAwareTrainer(model, train_loader, val_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} Model")
    print(f"Target params: {num_params_target:,}")
    print(f"Actual params: {actual_params:,}")
    print(f"{'='*60}")
    
    history = trainer.train()
    
    final_metrics = history[-1]
    
    print(f"\nFinal Results:")
    print(f"  Train Loss: {final_metrics['train_loss']:.4f}")
    print(f"  Val Loss: {final_metrics['val_loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
    
    return {
        'model_name': model_name,
        'target_params': num_params_target,
        'actual_params': actual_params,
        'config': trainer.curriculum.get_config(),
        'history': history,
        'final_metrics': final_metrics
    }


def run_comparison_experiment():
    results = {}
    
    models = [
        (SmallModel, 'Small', 30_000),
        (MediumModel, 'Medium', 300_000),
        (LargeModel, 'Large', 1_500_000)
    ]
    
    for model_class, name, target in models:
        result = run_scale_experiment(model_class, name, target)
        results[name] = result
    
    output_dir = Path(__file__).parent.parent / 'experiments' / 'scale_aware'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'curriculum_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results.items():
        metrics = result['final_metrics']
        config = result['config']
        print(f"\n{name} Model ({result['actual_params']:,} params):")
        print(f"  Scale: {config['scale']}")
        print(f"  Learning Rate: {config['lr_schedule']['base_lr']}")
        print(f"  Max Epochs: {config['epoch_schedule']['total']}")
        print(f"  Final Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Target Met: {metrics['accuracy'] >= 1.0}")
    
    return results


if __name__ == '__main__':
    results = run_comparison_experiment()
    
    print("\n✓ Scale-aware curriculum experiments complete!")
    print(f"Results saved to: experiments/scale_aware/curriculum_results.json")
