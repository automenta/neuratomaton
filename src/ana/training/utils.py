"""
Training utilities for ANA models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import datetime

class Trainer:
    """
    Generic trainer for ANA and baseline models with checkpointing and metrics logging.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: float = 1.0,
        log_interval: int = 100,
        checkpoint_dir: str = "checkpoints",
        result_manager: Optional[Any] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.result_manager = result_manager

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """
        Perform a single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            # Handle models that return (logits, info) or just logits
            output = self.model(batch_x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        else:
            logits = self.model(batch_x)
        
        # Compute loss
        if mask is not None:
            # Mask is usually [Batch, Seq] or [Batch, Seq, 1]
            active_pos = mask.view(-1).bool()
            if active_pos.any():
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = batch_y.reshape(-1)

                # Filter
                logits_active = logits_flat[active_pos]
                targets_active = targets_flat[active_pos]

                loss = F.cross_entropy(logits_active, targets_active)
            else:
                # No active tokens, zero loss
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
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
                output = self.model(batch_x)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
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
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}", leave=False)
        for batch_item in progress_bar:
            mask = None
            if len(batch_item) == 2:
                batch_x, batch_y = batch_item
            elif len(batch_item) == 3:
                batch_x, batch_y, mask = batch_item
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_item)} elements")

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            
            loss = self.train_step(batch_x, batch_y, mask=mask)
            losses.append(loss)
            
            if step_count % self.log_interval == 0:
                progress_bar.set_postfix({'loss': f"{loss:.4f}"})
                if self.result_manager:
                     self.result_manager.log(f"Epoch {self.current_epoch} Step {step_count}: Loss {loss:.4f}")

            step_count += 1
            if max_steps and step_count >= max_steps:
                break
        
        return losses

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)
            self.logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint not found: {path}")
            return

        # Use weights_only=False to allow loading of python primitives/numpy scalars stored in checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        self.logger.info(f"Loaded checkpoint from {path} (Epoch {self.current_epoch})")

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10, val_every: int = 1):
        """
        Full training loop.
        """
        self.logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            self.current_epoch = epoch
            train_losses = self.train_epoch(train_loader)
            avg_train_loss = float(np.mean(train_losses))
            self.train_losses.append(avg_train_loss)

            log_msg = f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}"

            if val_loader and epoch % val_every == 0:
                val_loss, val_ppl = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                log_msg += f", Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}"

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", is_best=True)
                else:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", is_best=False)
            else:
                 self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", is_best=False)

            self.logger.info(log_msg)
            if self.result_manager:
                self.result_manager.log(log_msg)

        self.current_epoch += epochs


class TwoPhaseTrainer:
    """
    Specialized trainer for two-phase training methodology with checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        phase1_lr: float = 1e-3,
        phase2_lr: float = 1e-4,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        result_manager: Optional[Any] = None
    ):
        self.model = model
        self.device = device
        self.phase1_lr = phase1_lr
        self.phase2_lr = phase2_lr
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.result_manager = result_manager
        
        # Separate optimizers for each phase
        self.phase1_optimizer = None
        self.phase2_optimizer = None
        
        self.logger = logging.getLogger(__name__)
        os.makedirs(checkpoint_dir, exist_ok=True)

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
        if not self.phase1_optimizer:
             self.phase1_optimizer = torch.optim.Adam(trainable_params, lr=self.phase1_lr)
        
        self.logger.info(f"Phase 1: Training {sum(p.numel() for p in trainable_params):,} parameters")
    
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
        if not self.phase2_optimizer:
             self.phase2_optimizer = torch.optim.Adam(trainable_params, lr=self.phase2_lr)
        
        self.logger.info(f"Phase 2: Training {sum(p.numel() for p in trainable_params):,} parameters")
    
    def train_phase1(self, dataloader: DataLoader, steps: int) -> List[float]:
        """
        Train Phase 1
        """
        self.setup_phase1()
        # Use basic trainer for the loop, but manage optimizer ourselves
        trainer = Trainer(self.model, self.phase1_optimizer, self.device, self.grad_clip, result_manager=self.result_manager)
        
        losses = []
        step_count = 0
        
        progress_bar = tqdm(dataloader, desc="Phase 1", total=steps)
        # We might need to cycle dataloader if it's smaller than steps
        data_iter = iter(dataloader)

        while step_count < steps:
            try:
                batch_item = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_item = next(data_iter)
                
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
            
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f"{loss:.4f}"})

            step_count += 1
        
        progress_bar.close()
        self.save_checkpoint("phase1_checkpoint.pt")
        return losses
    
    def train_phase2(self, dataloader: DataLoader, steps: int) -> List[float]:
        """
        Train Phase 2
        """
        self.setup_phase2()
        trainer = Trainer(self.model, self.phase2_optimizer, self.device, self.grad_clip, result_manager=self.result_manager)
        
        losses = []
        step_count = 0
        
        progress_bar = tqdm(dataloader, desc="Phase 2", total=steps)
        data_iter = iter(dataloader)

        while step_count < steps:
            try:
                batch_item = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_item = next(data_iter)
                
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
            
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f"{loss:.4f}"})

            step_count += 1

        progress_bar.close()
        self.save_checkpoint("phase2_checkpoint.pt")
        return losses
    
    def train_both_phases(self, dataloader: DataLoader, phase1_steps: int, phase2_steps: int) -> Dict:
        """
        Train both phases and return results
        """
        self.logger.info("Starting Phase 1: Training HoloLink only")
        phase1_losses = self.train_phase1(dataloader, phase1_steps)
        
        self.logger.info("Starting Phase 2: Fine-tuning controller")
        phase2_losses = self.train_phase2(dataloader, phase2_steps)
        
        return {
            'phase1_losses': phase1_losses,
            'phase2_losses': phase2_losses,
            'total_steps': phase1_steps + phase2_steps
        }

    def save_checkpoint(self, filename: str):
        """
        Save checkpoint with both optimizers.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            'model_state_dict': self.model.state_dict(),
            'phase1_optimizer': self.phase1_optimizer.state_dict() if self.phase1_optimizer else None,
            'phase2_optimizer': self.phase2_optimizer.state_dict() if self.phase2_optimizer else None
        }
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str):
        """
        Load checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint not found: {path}")
            return

        # Use weights_only=False to allow loading of python primitives/numpy scalars stored in checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint.get('phase1_optimizer') and self.phase1_optimizer:
            self.phase1_optimizer.load_state_dict(checkpoint['phase1_optimizer'])

        if checkpoint.get('phase2_optimizer') and self.phase2_optimizer:
            self.phase2_optimizer.load_state_dict(checkpoint['phase2_optimizer'])

        self.logger.info(f"Checkpoint loaded from {path}")


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
