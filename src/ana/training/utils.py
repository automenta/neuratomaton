"""
Training utilities for ANA models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from tqdm import tqdm
import logging


class Trainer:
    """
    Generic trainer for ANA and baseline models
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: float = 1.0,
        log_interval: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        """
        Perform a single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            logits, _ = self.model(batch_x)
        else:
            logits = self.model(batch_x)
        
        # Compute loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> Tuple[float, float]:
        """
        Evaluate model on validation set
        Returns (average_loss, perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        batch_count = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            if hasattr(self.model, 'forward'):
                logits, _ = self.model(batch_x)
            else:
                logits = self.model(batch_x)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += batch_x.numel()
            
            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = float(torch.exp(torch.tensor(avg_loss)))
        
        return avg_loss, perplexity
    
    def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> List[float]:
        """
        Train for one epoch
        """
        losses = []
        step_count = 0
        
        for batch_x, batch_y in tqdm(dataloader, desc="Training"):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            loss = self.train_step(batch_x, batch_y)
            losses.append(loss)
            
            if step_count % self.log_interval == 0:
                self.logger.info(f"Step {step_count}, Loss: {loss:.4f}")
            
            step_count += 1
            if max_steps and step_count >= max_steps:
                break
        
        return losses


class TwoPhaseTrainer:
    """
    Specialized trainer for two-phase training methodology
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        phase1_lr: float = 1e-3,
        phase2_lr: float = 1e-4,
        grad_clip: float = 1.0
    ):
        self.model = model
        self.device = device
        self.phase1_lr = phase1_lr
        self.phase2_lr = phase2_lr
        self.grad_clip = grad_clip
        
        # Separate optimizers for each phase
        self.phase1_optimizer = None
        self.phase2_optimizer = None
        
    def setup_phase1(self):
        """
        Setup for Phase 1: Train HoloLink only (freeze controller)
        """
        for name, param in self.model.named_parameters():
            if 'controller' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Create optimizer for trainable parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.phase1_optimizer = torch.optim.Adam(trainable_params, lr=self.phase1_lr)
        
        print(f"Phase 1: Training {sum(p.numel() for p in trainable_params):,} parameters")
    
    def setup_phase2(self):
        """
        Setup for Phase 2: Fine-tune controller (freeze HoloLink)
        """
        for name, param in self.model.named_parameters():
            if 'holo' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Create optimizer for trainable parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.phase2_optimizer = torch.optim.Adam(trainable_params, lr=self.phase2_lr)
        
        print(f"Phase 2: Training {sum(p.numel() for p in trainable_params):,} parameters")
    
    def train_phase1(self, dataloader: DataLoader, steps: int) -> List[float]:
        """
        Train Phase 1
        """
        self.setup_phase1()
        trainer = Trainer(self.model, self.phase1_optimizer, self.device, self.grad_clip)
        
        losses = []
        step_count = 0
        
        for batch_item in dataloader:
            if step_count >= steps:
                break
                
            # Handle both 2-tuple (x, y) and 3-tuple (x, y, mask) datasets
            if len(batch_item) == 2:
                batch_x, batch_y = batch_item
            elif len(batch_item) == 3:
                batch_x, batch_y, _ = batch_item  # Ignore mask for standard training
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_item)} elements")
                
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            loss = trainer.train_step(batch_x, batch_y)
            losses.append(loss)
            
            step_count += 1
        
        return losses
    
    def train_phase2(self, dataloader: DataLoader, steps: int) -> List[float]:
        """
        Train Phase 2
        """
        self.setup_phase2()
        trainer = Trainer(self.model, self.phase2_optimizer, self.device, self.grad_clip)
        
        losses = []
        step_count = 0
        
        for batch_item in dataloader:
            if step_count >= steps:
                break
                
            # Handle both 2-tuple (x, y) and 3-tuple (x, y, mask) datasets
            if len(batch_item) == 2:
                batch_x, batch_y = batch_item
            elif len(batch_item) == 3:
                batch_x, batch_y, _ = batch_item  # Ignore mask for standard training
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_item)} elements")
                
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            loss = trainer.train_step(batch_x, batch_y)
            losses.append(loss)
            
            step_count += 1
        
        return losses
    
    def train_both_phases(self, dataloader: DataLoader, phase1_steps: int, phase2_steps: int) -> Dict:
        """
        Train both phases and return results
        """
        print("Starting Phase 1: Training HoloLink only")
        phase1_losses = self.train_phase1(dataloader, phase1_steps)
        
        print("Starting Phase 2: Fine-tuning controller")
        phase2_losses = self.train_phase2(dataloader, phase2_steps)
        
        return {
            'phase1_losses': phase1_losses,
            'phase2_losses': phase2_losses,
            'total_steps': phase1_steps + phase2_steps
        }


def create_masked_dataloader(dataloader: DataLoader, mask_func: Callable) -> DataLoader:
    """
    Create a dataloader that applies masking to targets
    Useful for tasks like associative recall where only certain positions matter
    """
    class MaskedDataset:
        def __init__(self, base_dataset, mask_func):
            self.base_dataset = base_dataset
            self.mask_func = mask_func
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            x, y = self.base_dataset[idx]
            mask = self.mask_func(y)
            return x, y, mask
    
    masked_dataset = MaskedDataset(dataloader.dataset, mask_func)
    return DataLoader(masked_dataset, batch_size=dataloader.batch_size, shuffle=dataloader.shuffle)


def compute_masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute loss only on masked positions
    """
    # Only compute loss where mask is 1
    active_positions = mask.bool()
    if not active_positions.any():
        return torch.tensor(0.0, device=logits.device)
    
    active_logits = logits[active_positions]
    active_targets = targets[active_positions]
    
    return F.cross_entropy(active_logits, active_targets)