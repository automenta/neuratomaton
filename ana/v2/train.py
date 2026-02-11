#!/usr/bin/env python3
"""
ANA v2: Simple, brutal training loop.

No bloat, no curriculum frameworks, no fancy schedulers.
Just gradient descent on the task.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from pathlib import Path
from typing import Dict, Tuple

from .core import ANAConfig, ANAModel


class SimpleDataset(Dataset):
    """Simple sequence dataset for training."""
    
    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class Trainer:
    """
    Brutally simple trainer.
    
    Features:
    - Adam optimizer
    - Cosine learning rate schedule
    - Gradient clipping
    - Checkpointing
    - That's it.
    """
    
    def __init__(self, 
                 config: ANAConfig,
                 output_dir: str = "ana/v2/checkpoints",
                 lr: float = 3e-4,
                 warmup_steps: int = 100):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = ANAModel(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.best_loss = float('inf')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, 
              train_loader: DataLoader,
              num_epochs: int = 100,
              val_loader: DataLoader = None,
              eval_every: int = 10) -> Dict:
        """
        Train the model.
        
        Returns training history.
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (x, targets) in enumerate(train_loader):
                x = x.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                logits = self.model(x)
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    ignore_index=0
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                self.step_count += 1
                lr = self._get_lr()
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
            
            avg_train_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(lr)
            
            print(f"Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f}")
            
            if val_loader is not None and (epoch + 1) % eval_every == 0:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(f'best.pt')
            
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, targets in dataloader:
                x = x.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(x)
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    ignore_index=0
                )
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _get_lr(self) -> float:
        """Cosine learning rate with warmup."""
        if self.step_count < self.warmup_steps:
            return self.step_count / self.warmup_steps * 3e-4
        
        progress = (self.step_count - self.warmup_steps) / max(1, 10000 - self.warmup_steps)
        return 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.best_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint from {path}")


def train_on_dataset(sequences: torch.Tensor, 
                     targets: torch.Tensor,
                     vocab_size: int,
                     epochs: int = 100,
                     batch_size: int = 32) -> Trainer:
    """
    Train ANA on a given dataset.
    
    Simple entry point for training.
    """
    config = ANAConfig(
        d_model=128,
        vocab_size=vocab_size,
        track_dims=(32, 64, 32),
        stack_depth=5,
        stack_dim=64,
        num_layers=2
    )
    
    dataset = SimpleDataset(sequences, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    trainer = Trainer(config)
    history = trainer.train(loader, num_epochs=epochs)
    
    return trainer
