"""
Advanced training utilities for ANA models
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
import json
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.config import ANAConfig
from ..models.core import ANAModel


class AdvancedTrainer:
    """
    Advanced trainer with more sophisticated features
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: float = 1.0,
        log_interval: int = 100,
        save_checkpoint_every: Optional[int] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_checkpoint_every = save_checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch': []
        }
        
    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None) -> float:
        """
        Perform a single training step with optional masking
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            logits, info = self.model(batch_x)
        else:
            logits = self.model(batch_x)
            info = []
        
        # Compute loss (with optional masking)
        if mask is not None:
            # Only compute loss where mask is 1
            active_positions = mask.bool()
            if active_positions.any():
                active_logits = logits[active_positions]
                active_targets = batch_y[active_positions]
                loss = F.cross_entropy(active_logits, active_targets)
            else:
                # If no active positions, return 0 loss
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update parameters
        self.optimizer.step()
        
        # Record learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = None, 
                 mask_func: Optional[Callable] = None) -> Tuple[float, float, Dict]:
        """
        Evaluate model on validation set with additional metrics
        Returns (average_loss, perplexity, additional_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_predictions = 0
        
        batch_count = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            if hasattr(self.model, 'forward'):
                logits, _ = self.model(batch_x)
            else:
                logits = self.model(batch_x)
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += batch_x.numel()
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == batch_y).sum().item()
            total_correct += correct
            total_predictions += batch_y.numel()
            
            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = float(torch.exp(torch.tensor(avg_loss))) if avg_loss != float('inf') else float('inf')
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        additional_metrics = {
            'accuracy': accuracy,
            'total_correct': total_correct,
            'total_predictions': total_predictions
        }
        
        return avg_loss, perplexity, additional_metrics
    
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
            self.train_losses.append(loss)
            
            if step_count % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Step {step_count}, Loss: {loss:.4f}, LR: {current_lr:.2e}")
            
            step_count += 1
            if max_steps and step_count >= max_steps:
                break
        
        return losses
    
    def save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        """
        Save model checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.history = checkpoint.get('history', {})
        
        epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
        return epoch


class CurriculumLearningScheduler:
    """
    Curriculum learning scheduler that gradually increases task difficulty
    """
    def __init__(self, initial_difficulty: float = 0.1, max_difficulty: float = 1.0, 
                 schedule_type: str = "linear", milestones: Optional[List[int]] = None):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.schedule_type = schedule_type
        self.milestones = milestones or [100, 200, 300]  # Default milestones
        self.current_difficulty = initial_difficulty
        
    def get_difficulty(self, step: int) -> float:
        """
        Get current difficulty level based on training step
        """
        if self.schedule_type == "linear":
            progress = min(step / max(self.milestones), 1.0)
            self.current_difficulty = self.initial_difficulty + \
                                    (self.max_difficulty - self.initial_difficulty) * progress
        elif self.schedule_type == "step":
            milestone_idx = 0
            for milestone in self.milestones:
                if step >= milestone:
                    milestone_idx += 1
            fraction = min(milestone_idx / len(self.milestones), 1.0)
            self.current_difficulty = self.initial_difficulty + \
                                    (self.max_difficulty - self.initial_difficulty) * fraction
        elif self.schedule_type == "exponential":
            # Exponential increase in difficulty
            progress = min(step / max(self.milestones), 1.0)
            self.current_difficulty = self.initial_difficulty + \
                                    (self.max_difficulty - self.initial_difficulty) * (progress ** 2)
        
        return min(self.current_difficulty, self.max_difficulty)


class ModelAnalyzer:
    """
    Advanced model analysis tools
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def analyze_gradients(self) -> Dict[str, Any]:
        """
        Analyze gradient flow in the model
        """
        grad_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_abs_mean = param.grad.abs().mean().item()
                grad_abs_max = param.grad.abs().max().item()
                
                grad_stats[name] = {
                    'grad_norm': grad_norm,
                    'grad_abs_mean': grad_abs_mean,
                    'grad_abs_max': grad_abs_max
                }
        
        return grad_stats
    
    def analyze_activations(self, input_batch: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze activations in the model
        """
        activation_stats = {}
        
        # Register hooks to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    activations[name] = input[0].detach() if len(input) > 0 else output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                handle = module.register_forward_hook(get_activation(name))
                handles.append(handle)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                _ = self.model(input_batch)
            else:
                _ = self.model(input_batch)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Calculate statistics
        for name, activation in activations.items():
            act_norm = activation.norm().item()
            act_mean = activation.mean().item()
            act_std = activation.std().item()
            act_abs_mean = activation.abs().mean().item()
            
            activation_stats[name] = {
                'activation_norm': act_norm,
                'activation_mean': act_mean,
                'activation_std': act_std,
                'activation_abs_mean': act_abs_mean
            }
        
        return activation_stats
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analyze memory usage of the model
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        buffer_count = sum(b.numel() for b in self.model.buffers())
        total_params = param_count + buffer_count
        
        param_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        buffer_size_mb = sum(b.numel() * b.element_size() for b in self.model.buffers()) / (1024**2)
        total_size_mb = param_size_mb + buffer_size_mb
        
        return {
            'parameter_count': param_count,
            'buffer_count': buffer_count,
            'total_elements': total_params,
            'parameter_size_mb': param_size_mb,
            'buffer_size_mb': buffer_size_mb,
            'total_size_mb': total_size_mb
        }
    
    def analyze_parameter_distribution(self) -> Dict[str, Any]:
        """
        Analyze parameter distributions
        """
        param_stats = {}
        
        for name, param in self.model.named_parameters():
            param_norm = param.norm().item()
            param_mean = param.mean().item()
            param_std = param.std().item()
            param_abs_mean = param.abs().mean().item()
            
            param_stats[name] = {
                'param_norm': param_norm,
                'param_mean': param_mean,
                'param_std': param_std,
                'param_abs_mean': param_abs_mean
            }
        
        return param_stats


class VisualizationTools:
    """
    Tools for visualizing model behavior and training progress
    """
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_training_curves(self, train_losses: List[float], val_losses: Optional[List[float]] = None, 
                           title: str = "Training Curves", save_path: Optional[str] = None):
        """
        Plot training curves
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(train_losses, label='Training Loss', alpha=0.7)
        if val_losses is not None:
            ax.plot(val_losses, label='Validation Loss', alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_attention_patterns(self, attention_weights: torch.Tensor, 
                              title: str = "Attention Patterns", save_path: Optional[str] = None):
        """
        Plot attention patterns (for models that have attention weights)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attention_weights.cpu().numpy(), cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Positions')
        ax.set_ylabel('Query Positions')
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_parameter_heatmap(self, param_dict: Dict[str, Dict], 
                              title: str = "Parameter Statistics Heatmap", save_path: Optional[str] = None):
        """
        Plot heatmap of parameter statistics
        """
        # Prepare data
        param_names = list(param_dict.keys())
        stat_types = list(param_dict[param_names[0]].keys()) if param_names else []
        
        if not stat_types:
            print("No parameter statistics to plot")
            return
        
        data_matrix = []
        for name in param_names:
            row = [param_dict[name][stat] for stat in stat_types]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(param_names) * 0.3)))
        
        im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', vmin=data_matrix.min(), vmax=data_matrix.max())
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(stat_types)))
        ax.set_yticks(np.arange(len(param_names)))
        ax.set_xticklabels(stat_types)
        ax.set_yticklabels([name.split('.')[-1][:20] for name in param_names])  # Shorten names
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")
        
        # Annotate cells
        for i in range(len(param_names)):
            for j in range(len(stat_types)):
                text = ax.text(j, i, f"{data_matrix[i, j]:.3f}",
                              ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class ModelProfiler:
    """
    Performance profiling tools
    """
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
    
    def profile_throughput(self, input_shapes: List[Tuple], num_runs: int = 100) -> Dict[str, Any]:
        """
        Profile model throughput for different input shapes
        """
        results = {}
        
        for shape in input_shapes:
            # Create dummy input
            dummy_input = torch.randn(shape).to(self.device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    if hasattr(self.model, 'forward'):
                        _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)
            
            # Timing runs
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_event = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
            end_event = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
            
            if start_event:
                start_event.record()
            
            start_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else torch.tensor([torch.cuda.Event(enable_timing=True)]) if self.device == 'cuda' else torch.tensor([torch.tensor(torch.cuda.Event(enable_timing=True))]) if self.device == 'cuda' else torch.tensor([torch.tensor(0.0)])
            start_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else torch.tensor([torch.tensor(0.0)])
            
            if self.device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    if hasattr(self.model, 'forward'):
                        _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)
            
            if self.device == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            else:
                import time
                start_time = time.time()
                for _ in range(num_runs):
                    with torch.no_grad():
                        if hasattr(self.model, 'forward'):
                            _ = self.model(dummy_input)
                        else:
                            _ = self.model(dummy_input)
                elapsed_time = time.time() - start_time
            
            avg_time = elapsed_time / num_runs
            tokens_per_second = (dummy_input.numel() / dummy_input.size(-1)) / avg_time if avg_time > 0 else 0
            
            results[str(shape)] = {
                'avg_time_per_run': avg_time,
                'tokens_per_second': tokens_per_second,
                'shape': shape
            }
        
        return results
    
    def profile_memory(self, input_shapes: List[Tuple], num_runs: int = 10) -> Dict[str, Any]:
        """
        Profile model memory usage for different input shapes
        """
        results = {}
        
        for shape in input_shapes:
            # Create dummy input
            dummy_input = torch.randn(shape).to(self.device)
            
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                
                # Forward pass
                for _ in range(num_runs):
                    with torch.no_grad():
                        if hasattr(self.model, 'forward'):
                            _ = self.model(dummy_input)
                        else:
                            _ = self.model(dummy_input)
                
                torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            else:
                # For CPU, we'll just return 0 for now
                peak_memory = 0.0
            
            results[str(shape)] = {
                'peak_memory_mb': peak_memory,
                'shape': shape
            }
        
        return results


def create_adaptive_optimizer(model: nn.Module, base_lr: float = 1e-3, 
                             optimizer_type: str = "adam") -> torch.optim.Optimizer:
    """
    Create adaptive optimizer with different learning rates for different components
    """
    # Separate parameters by module type
    hololink_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'holo' in name.lower():
            hololink_params.append(param)
        else:
            other_params.append(param)
    
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam([
            {'params': hololink_params, 'lr': base_lr * 0.5},  # Lower LR for HoloLink
            {'params': other_params, 'lr': base_lr}  # Normal LR for other components
        ])
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD([
            {'params': hololink_params, 'lr': base_lr * 0.5},
            {'params': other_params, 'lr': base_lr}
        ], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer