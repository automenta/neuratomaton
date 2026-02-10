import sys
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


class OptimizedBioANATrainer:
    def __init__(
        self,
        config_name: str = 'nano',
        device: Optional[str] = None,
        use_amp: bool = True,
        use_compile: bool = False,
        adaptive_relaxation: bool = True,
    ):
        self.config_name = config_name
        self.config = get_bio_config(config_name)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.use_compile = use_compile
        self.adaptive_relaxation = adaptive_relaxation
        
        self.model = create_bio_ana(config_name).to(self.device)
        
        if self.use_compile:
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.stats = {
            'iterations_per_token': [],
            'convergence_rates': [],
            'speedup_factors': [],
        }
    
    def compute_adaptive_iterations(
        self,
        token_idx: int,
        total_tokens: int,
        base_iters: int,
    ) -> int:
        if not self.adaptive_relaxation:
            return base_iters
        
        progress = token_idx / total_tokens
        
        if progress < 0.3:
            return base_iters
        elif progress < 0.6:
            return max(base_iters // 2, 5)
        elif progress < 0.8:
            return max(base_iters // 4, 3)
        else:
            return max(base_iters // 6, 2)
    
    def forward_with_early_stopping(
        self,
        input_ids: torch.Tensor,
        convergence_threshold: float = 0.01,
        max_iters: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        base_iters = max_iters or self.config.relaxation_iterations
        batch_size, seq_len = input_ids.shape
        
        x = self.model.embedding(input_ids)
        x = self.model._add_position_encoding(x)
        
        outputs = []
        track_states = {'syntax': None, 'semantic': None, 'logic': None}
        info = {
            'iterations_used': [],
            'converged_early': [],
        }
        
        for t in range(seq_len):
            xt = x[:, t, :]
            iters = self.compute_adaptive_iterations(t, seq_len, base_iters)
            
            h_prev = None
            converged = False
            
            for i in range(iters):
                track_out, h_new = self.model.tracks(
                    xt,
                    h_syntax=track_states['syntax'],
                    h_semantic=track_states['semantic'],
                    h_logic=track_states['logic'],
                    steps=1,
                )
                
                if h_prev is not None:
                    diff = sum(
                        torch.abs(h_new[k] - h_prev[k]).max().item()
                        for k in ['syntax', 'semantic', 'logic']
                    )
                    if diff < convergence_threshold:
                        converged = True
                        break
                
                h_prev = {k: v.clone() for k, v in h_new.items()}
                track_states = h_new
            
            actual_iters = i + 1
            info['iterations_used'].append(actual_iters)
            info['converged_early'].append(converged)
            
            if self.model.hololink:
                track_out, _ = self.model.hololink(track_out, write_mode=self.model.training)
            
            mixed = self.model.mixer(track_out)
            out = self.model.norm(xt + mixed)
            outputs.append(out)
        
        output_seq = torch.stack(outputs, dim=1)
        logits = self.model.output_head(output_seq)
        
        return logits, info
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits, info = self.forward_with_early_stopping(input_ids)
                loss = self.model.compute_loss(logits, targets)
            
            self.scaler.scale(loss['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, info = self.forward_with_early_stopping(input_ids)
            loss = self.model.compute_loss(logits, targets)
            loss['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.stats['iterations_per_token'].append(info['iterations_used'])
        self.stats['convergence_rates'].append(
            sum(info['converged_early']) / len(info['converged_early'])
        )
        
        return {
            'total_loss': loss['total'].item(),
            'ce_loss': loss['ce'].item(),
            'avg_iterations': sum(info['iterations_used']) / len(info['iterations_used']),
            'early_stop_rate': sum(info['converged_early']) / len(info['converged_early']),
        }
    
    @torch.no_grad()
    def evaluate(self, input_ids: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits, info = self.forward_with_early_stopping(input_ids)
                loss = self.model.compute_loss(logits, targets)
        else:
            logits, info = self.forward_with_early_stopping(input_ids)
            loss = self.model.compute_loss(logits, targets)
        
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()
        
        return {
            'total_loss': loss['total'].item(),
            'ce_loss': loss['ce'].item(),
            'accuracy': accuracy,
            'avg_iterations': sum(info['iterations_used']) / len(info['iterations_used']),
        }
    
    def benchmark(
        self,
        batch_size: int = 4,
        seq_len: int = 64,
        num_steps: int = 10,
    ) -> Dict[str, float]:
        self.model.eval()
        
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(num_steps):
            self.train_step(input_ids, targets)
        elapsed = time.perf_counter() - start
        
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2 if self.device.type == 'cuda' else 0
        
        steps_per_sec = num_steps / elapsed
        tokens_per_sec = num_steps * batch_size * seq_len / elapsed
        
        return {
            'steps_per_sec': steps_per_sec,
            'tokens_per_sec': tokens_per_sec,
            'time_per_step_ms': elapsed / num_steps * 1000,
            'memory_mb': memory_mb,
        }


def run_optimization_validation():
    output_dir = Path("results/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    configs = [
        {'name': 'baseline', 'use_amp': False, 'use_compile': False, 'adaptive_relaxation': False},
        {'name': 'amp_only', 'use_amp': True, 'use_compile': False, 'adaptive_relaxation': False},
        {'name': 'adaptive_only', 'use_amp': False, 'use_compile': False, 'adaptive_relaxation': True},
        {'name': 'amp_adaptive', 'use_amp': True, 'use_compile': False, 'adaptive_relaxation': True},
    ]
    
    print("="*60)
    print("OPTIMIZATION VALIDATION")
    print("="*60)
    
    for cfg in configs:
        print(f"\nTesting: {cfg['name']}")
        print("-" * 60)
        
        trainer = OptimizedBioANATrainer(
            config_name='nano',
            **{k: v for k, v in cfg.items() if k != 'name'}
        )
        
        benchmark = trainer.benchmark(batch_size=4, seq_len=64, num_steps=20)
        
        print(f"  Steps/sec: {benchmark['steps_per_sec']:.2f}")
        print(f"  Tokens/sec: {benchmark['tokens_per_sec']:.0f}")
        print(f"  Time/step: {benchmark['time_per_step_ms']:.2f}ms")
        print(f"  Memory: {benchmark['memory_mb']:.1f} MB")
        
        results[cfg['name']] = benchmark
        results[cfg['name']]['config'] = cfg
    
    baseline = results['baseline']
    print("\n" + "="*60)
    print("SPEEDUP COMPARISON")
    print("="*60)
    
    for name, res in results.items():
        if name != 'baseline':
            speedup = res['tokens_per_sec'] / baseline['tokens_per_sec']
            print(f"{name}: {speedup:.2f}x speedup")
    
    results_file = output_dir / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_optimization_validation()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    best = max(results.items(), key=lambda x: x[1]['tokens_per_sec'])
    print(f"\nBest configuration: {best[0]}")
    print(f"Tokens/sec: {best[1]['tokens_per_sec']:.0f}")
    print(f"Memory: {best[1]['memory_mb']:.1f} MB")
