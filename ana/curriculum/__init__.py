"""
Scale-Aware Training Curriculum for Bio-ANA Models

This module implements automated hyperparameter selection based on model size.
Addresses the finding that different scales require different learning rates.

Key Results from Research:
- Small models (< 50K params): lr=1e-3, 20 epochs
- Medium models (50K-500K params): lr=3e-4, 30 epochs  
- Large models (> 500K params): lr=1e-4, 40 epochs
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import math


class ScaleAwareCurriculum:
    def __init__(self, model, target_accuracy=1.0, device='auto'):
        self.model = model
        self.device = self._resolve_device(device)
        self.num_params = self._count_params()
        self.target_accuracy = target_accuracy
        
        self.lr_schedule = self._compute_lr_schedule()
        self.epoch_schedule = self._compute_epoch_schedule()
        self.batch_schedule = self._compute_batch_schedule()
        
    def _resolve_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _count_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _compute_lr_schedule(self):
        num_params = self.num_params
        
        if num_params < 50_000:
            base_lr = 1e-3
            max_lr = 3e-3
        elif num_params < 500_000:
            base_lr = 3e-4
            max_lr = 1e-3
        else:
            base_lr = 1e-4
            max_lr = 3e-4
        
        return {'base_lr': base_lr, 'max_lr': max_lr}
    
    def _compute_epoch_schedule(self):
        num_params = self.num_params
        
        if num_params < 50_000:
            total_epochs = 20
            warmup_epochs = 3
        elif num_params < 500_000:
            total_epochs = 30
            warmup_epochs = 5
        else:
            total_epochs = 40
            warmup_epochs = 8
        
        return {'total': total_epochs, 'warmup': warmup_epochs}
    
    def _compute_batch_schedule(self):
        num_params = self.num_params
        
        if num_params < 50_000:
            batch_size = 32
        elif num_params < 500_000:
            batch_size = 16
        else:
            batch_size = 8
        
        return batch_size
    
    def get_optimizer(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr_schedule['base_lr'],
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-8
        )
        return optimizer
    
    def get_scheduler(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr_schedule['max_lr'],
            total_steps=self.epoch_schedule['total'] * 100,
            pct_start=self.epoch_schedule['warmup'] / self.epoch_schedule['total'],
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        return scheduler
    
    def get_batch_size(self):
        return self.batch_schedule
    
    def get_total_epochs(self):
        return self.epoch_schedule['total']
    
    def should_stop(self, metrics):
        accuracy = metrics.get('accuracy', 0.0)
        epochs = metrics.get('epoch', 0)
        
        if accuracy >= self.target_accuracy:
            return True, f"Target accuracy {self.target_accuracy} reached"
        
        if epochs >= self.epoch_schedule['total']:
            return True, f"Max epochs {self.epoch_schedule['total']} reached"
        
        return False, None
    
    def get_config(self):
        return {
            'num_params': self.num_params,
            'scale': self._get_scale_name(),
            'lr_schedule': self.lr_schedule,
            'epoch_schedule': self.epoch_schedule,
            'batch_size': self.batch_schedule
        }
    
    def _get_scale_name(self):
        if self.num_params < 50_000:
            return 'small'
        elif self.num_params < 500_000:
            return 'medium'
        else:
            return 'large'


class AdaptiveTrackCurriculum:
    def __init__(self, model, initial_weights={'syntax': 1.0, 'semantic': 1.0, 'logic': 1.0}):
        self.model = model
        self.track_weights = initial_weights.copy()
        self.track_history = {'syntax': [], 'semantic': [], 'logic': []}
    
    def update_weights(self, track_metrics):
        for track_name, metrics in track_metrics.items():
            accuracy = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', float('inf'))
            
            self.track_history[track_name].append({
                'accuracy': accuracy,
                'loss': loss,
                'weight': self.track_weights[track_name]
            })
            
            if len(self.track_history[track_name]) > 10:
                recent = self.track_history[track_name][-5:]
                avg_accuracy = sum(h['accuracy'] for h in recent) / len(recent)
                
                if avg_accuracy < 0.8:
                    self.track_weights[track_name] *= 1.1
                elif avg_accuracy > 0.95:
                    self.track_weights[track_name] *= 0.9
                
                self.track_weights[track_name] = max(0.5, min(2.0, self.track_weights[track_name]))
    
    def get_loss_weights(self):
        return self.track_weights.copy()


class ConvergenceMonitor:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.loss_history = []
    
    def update(self, loss):
        self.loss_history.append(loss)
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def is_converged(self):
        if len(self.loss_history) < 10:
            return False
        
        recent = self.loss_history[-5:]
        variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
        return variance < self.min_delta ** 2


class ScaleAwareTrainer:
    def __init__(self, model, train_loader, val_loader, device='auto'):
        self.model = model.to(self._resolve_device(device))
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.curriculum = ScaleAwareCurriculum(self.model)
        self.track_curriculum = AdaptiveTrackCurriculum(self.model)
        self.convergence_monitor = ConvergenceMonitor()
        
        self.optimizer = self.curriculum.get_optimizer()
        self.scheduler = self.curriculum.get_scheduler(self.optimizer)
        
        self.metrics_history = []
    
    def _resolve_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        track_metrics = {'syntax': {'accuracy': 0.0, 'loss': 0.0},
                        'semantic': {'accuracy': 0.0, 'loss': 0.0},
                        'logic': {'accuracy': 0.0, 'loss': 0.0}}
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
            targets = targets.to(self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss_dict = self.model.compute_loss(outputs, targets)
            
            loss_weights = self.track_curriculum.get_loss_weights()
            weighted_loss = (loss_weights.get('syntax', 1.0) * loss_dict.get('syntax_loss', 0) +
                           loss_weights.get('semantic', 1.0) * loss_dict.get('semantic_loss', 0) +
                           loss_weights.get('logic', 1.0) * loss_dict.get('logic_loss', 0) +
                           loss_dict.get('total', loss_dict['ce']))
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += weighted_loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        
        should_stop = self.convergence_monitor.update(avg_loss)
        
        metrics = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics, should_stop
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
                targets = targets.to(self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
                
                outputs = self.model(inputs)
                loss_dict = self.model.compute_loss(outputs, targets)
                total_loss += loss_dict['total'].item()
                
                predictions = outputs.argmax(-1)
                correct += (predictions == targets).sum().item()
                total += targets.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'epoch': epoch,
            'val_loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self):
        print(f"Scale-Aware Training: {self.curriculum._get_scale_name()} model ({self.curriculum.num_params:,} params)")
        print(f"Config: {self.curriculum.get_config()}")
        
        for epoch in range(self.curriculum.get_total_epochs()):
            train_metrics, should_stop = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(metrics)
            
            should_stop_final, reason = self.curriculum.should_stop(metrics)
            
            if (epoch + 1) % 5 == 0 or should_stop_final or should_stop:
                print(f"Epoch {epoch+1}/{self.curriculum.get_total_epochs()}: "
                      f"Train Loss={train_metrics['train_loss']:.4f}, "
                      f"Val Loss={val_metrics['val_loss']:.4f}, "
                      f"Acc={val_metrics['accuracy']:.2%}")
            
            if should_stop_final:
                print(f"Early stopping: {reason}")
                break
        
        return self.metrics_history


def create_curriculum(model, **kwargs):
    return ScaleAwareCurriculum(model, **kwargs)


def create_trainer(model, train_loader, val_loader, **kwargs):
    return ScaleAwareTrainer(model, train_loader, val_loader, **kwargs)
