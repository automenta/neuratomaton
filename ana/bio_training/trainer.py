import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

import sys
from pathlib import Path as P
sys.path.insert(0, str(P(__file__).parent.parent.parent / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config, BioANAModel, BioANAConfig


@dataclass
class TrainingMetrics:
    step: int
    epoch: int
    stage: str
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    avg_iterations: float = 7.0
    early_stop_rate: float = 0.0
    time_per_step_ms: float = 0.0
    tokens_per_sec: float = 0.0
    memory_mb: float = 0.0


class BioANATrainer:
    def __init__(
        self,
        config: BioANAConfig,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_adaptive_relaxation: bool = True,
        use_early_stopping: bool = True,
        convergence_threshold: float = 0.01,
        relaxation_schedule: Optional[List[int]] = None,
    ):
        self.config = config
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_adaptive_relaxation = use_adaptive_relaxation
        self.use_early_stopping = use_early_stopping
        self.convergence_threshold = convergence_threshold
        self.relaxation_schedule = relaxation_schedule or [12, 7, 3, 2]
        
        self.model = create_bio_ana(
            variant=config.variant,
            relaxation_iterations=config.relaxation_iterations,
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        
        self.max_grad_norm = max_grad_norm
        self.global_step = 0
        self.current_epoch = 0
        self.current_stage = '0'
        
        self.metrics_history: List[TrainingMetrics] = []
        self.stage_accuracy_buffer: Dict[str, List[float]] = {'0': [], '1': [], '2': []}
    
    def compute_adaptive_iterations(self, token_idx: int, total_tokens: int) -> int:
        if not self.use_adaptive_relaxation:
            return self.config.relaxation_iterations
        
        progress = token_idx / max(total_tokens, 1)
        
        if progress < 0.25:
            return self.relaxation_schedule[0]
        elif progress < 0.5:
            return self.relaxation_schedule[1]
        elif progress < 0.75:
            return self.relaxation_schedule[2]
        else:
            return self.relaxation_schedule[3]
    
    def forward_with_optimizations(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len = input_ids.shape
        
        x = self.model.embedding(input_ids)
        x = self.model._add_position_encoding(x)
        
        outputs = []
        track_states = {'syntax': None, 'semantic': None, 'logic': None}
        
        iterations_used = []
        early_stops = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            iters = self.compute_adaptive_iterations(t, seq_len)
            
            h_prev = None
            early_stop = False
            
            for i in range(iters):
                track_out, track_states = self.model.tracks(
                    xt,
                    h_syntax=track_states['syntax'],
                    h_semantic=track_states['semantic'],
                    h_logic=track_states['logic'],
                    steps=1,
                )
                
                if self.use_early_stopping and h_prev is not None:
                    max_diff = max(
                        torch.abs(track_states['syntax'] - h_prev['syntax']).max().item(),
                        torch.abs(track_states['semantic'] - h_prev['semantic']).max().item(),
                        torch.abs(track_states['logic'] - h_prev['logic']).max().item(),
                    )
                    if max_diff < self.convergence_threshold:
                        early_stop = True
                        break
                
                h_prev = {k: v.clone() for k, v in track_states.items()}
            
            iterations_used.append(i + 1)
            early_stops.append(early_stop)
            
            if self.model.hololink:
                track_out, _ = self.model.hololink(track_out, write_mode=self.model.training)
            
            mixed = self.model.mixer(track_out)
            out = self.model.norm(xt + mixed)
            outputs.append(out)
        
        output_seq = torch.stack(outputs, dim=1)
        logits = self.model.output_head(output_seq)
        
        info = {
            'iterations_used': iterations_used,
            'early_stops': early_stops,
            'avg_iterations': sum(iterations_used) / len(iterations_used),
            'early_stop_rate': sum(early_stops) / len(early_stops),
        }
        
        return logits, info
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        start_time = time.perf_counter()
        
        logits, info = self.forward_with_optimizations(input_ids)
        
        loss = self.model.compute_loss(logits, target_ids)
        
        loss['total'].backward()
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        elapsed = time.perf_counter() - start_time
        
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            mask = target_ids != 0
            correct = (predictions == target_ids) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        
        metrics = {
            'loss': loss['total'].item(),
            'ce_loss': loss['ce'].item(),
            'accuracy': accuracy.item(),
            'avg_iterations': info['avg_iterations'],
            'early_stop_rate': info['early_stop_rate'],
            'time_ms': elapsed * 1000,
            'tokens_per_sec': input_ids.numel() / elapsed,
        }
        
        if self.device.type == 'cuda':
            metrics['memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        self.global_step += 1
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_iterations = 0.0
        total_early_stops = 0.0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, info = self.forward_with_optimizations(input_ids)
            loss = self.model.compute_loss(logits, target_ids)
            
            predictions = logits.argmax(dim=-1)
            mask = target_ids != 0
            correct = (predictions == target_ids) & mask
            
            total_loss += loss['total'].item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            total_iterations += info['avg_iterations']
            total_early_stops += info['early_stop_rate']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_tokens,
            'avg_iterations': total_iterations / num_batches,
            'early_stop_rate': total_early_stops / num_batches,
        }
    
    def check_stage_advancement(self, accuracy: float, patience: int = 3) -> bool:
        self.stage_accuracy_buffer[self.current_stage].append(accuracy)
        
        buffer = self.stage_accuracy_buffer[self.current_stage]
        
        if len(buffer) >= patience:
            recent = buffer[-patience:]
            if all(acc >= self._get_stage_threshold() for acc in recent):
                return True
        
        return False
    
    def _get_stage_threshold(self) -> float:
        thresholds = {
            '0': 0.98,
            '1': 0.90,
            '2': 0.85,
        }
        return thresholds.get(self.current_stage, 0.98)
    
    def advance_stage(self) -> bool:
        stage_order = ['0', '1', '2']
        current_idx = stage_order.index(self.current_stage)
        
        if current_idx < len(stage_order) - 1:
            self.current_stage = stage_order[current_idx + 1]
            self.stage_accuracy_buffer[self.current_stage] = []
            return True
        
        return False
    
    def save_checkpoint(self, path: Path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'current_stage': self.current_stage,
            'config': asdict(self.config),
            'metrics_history': [asdict(m) for m in self.metrics_history],
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.current_stage = checkpoint['current_stage']
        
        self.metrics_history = [
            TrainingMetrics(**m) for m in checkpoint.get('metrics_history', [])
        ]
    
    def get_model_stats(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': str(self.device),
            'current_stage': self.current_stage,
            'global_step': self.global_step,
        }
