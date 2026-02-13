"""
Experiment runner for ANA models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import logging

from ..models.config import ANAConfig
from ..models.core import ANAModel, BaselineSSM
from ..training.utils import Trainer, TwoPhaseTrainer
from ..utils.datasets import AssociativeRecallDataset, TextDataset


class ExperimentRunner:
    """
    Main class to run comparative experiments between ANA and baseline models
    """
    def __init__(self, exp_name: str = "ana_experiment"):
        self.exp_name = exp_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"ana.experiments.{exp_name}")
        self.logger.setLevel(logging.INFO)
        
        # Create results directory
        self.results_dir = os.path.join("results", exp_name)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_text_generation_comparison(
        self,
        ana_config: ANAConfig,
        baseline_config: ANAConfig,
        train_dataset,
        val_dataset,
        max_steps: int = 1000,
        batch_size: int = 8,
        learning_rate: float = 1e-3
    ) -> Dict:
        """
        Run text generation comparison between ANA and baseline
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create models
        ana_model = ANAModel(ana_config).to(device)
        baseline_model = BaselineSSM(baseline_config).to(device)
        
        # Count parameters
        ana_params = sum(p.numel() for p in ana_model.parameters())
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        
        print(f"ANA Parameters: {ana_params:,}")
        print(f"Baseline Parameters: {baseline_params:,}")
        param_diff = abs(ana_params - baseline_params)
        param_ratio = max(ana_params, baseline_params) / min(ana_params, baseline_params)
        param_diff_percent = param_diff / min(ana_params, baseline_params) * 100
        print(f"Parameter Difference: {param_diff:,} ({param_diff_percent:.2f}%)")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create optimizers
        ana_optimizer = torch.optim.Adam(ana_model.parameters(), lr=learning_rate)
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        # Create trainers
        ana_trainer = Trainer(ana_model, ana_optimizer, device)
        baseline_trainer = Trainer(baseline_model, baseline_optimizer, device)
        
        # Training loop
        print("Training models...")
        for step in range(max_steps):
            try:
                batch_x, batch_y = next(iter(train_loader))
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_y = next(train_iter)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Train both models
            ana_loss = ana_trainer.train_step(batch_x, batch_y)
            baseline_loss = baseline_trainer.train_step(batch_x, batch_y)
            
            if step % 200 == 0:
                print(f"Step {step}: ANA Loss: {ana_loss:.4f}, Baseline Loss: {baseline_loss:.4f}")
        
        # Final evaluation
        ana_val_loss, ana_ppl = ana_trainer.evaluate(val_loader)
        baseline_val_loss, baseline_ppl = baseline_trainer.evaluate(val_loader)
        
        # Calculate improvements
        loss_improvement = ((baseline_val_loss - ana_val_loss) / baseline_val_loss) * 100
        ppl_improvement = ((baseline_ppl - ana_ppl) / baseline_ppl) * 100
        
        print(f"\\nFinal Results:")
        print(f"ANA - Loss: {ana_val_loss:.4f}, Perplexity: {ana_ppl:.2f}")
        print(f"Baseline - Loss: {baseline_val_loss:.4f}, Perplexity: {baseline_ppl:.2f}")
        print(f"ANA Loss Improvement: {loss_improvement:.2f}%")
        print(f"ANA Perplexity Improvement: {ppl_improvement:.2f}%")
        
        results = {
            'ana_final_loss': ana_val_loss,
            'baseline_final_loss': baseline_val_loss,
            'ana_final_perplexity': ana_ppl,
            'baseline_final_perplexity': baseline_ppl,
            'loss_improvement_pct': loss_improvement,
            'perplexity_improvement_pct': ppl_improvement,
            'ana_params': ana_params,
            'baseline_params': baseline_params,
            'param_difference': param_diff,
            'param_difference_pct': param_diff_percent,
            'param_ratio': param_ratio
        }
        
        return results
    
    def run_associative_recall_comparison(
        self,
        ana_config: ANAConfig,
        baseline_config: ANAConfig,
        num_samples: int = 1000,
        vocab_size: int = 40,
        num_pairs: int = 4,
        noise_len: int = 8,
        max_steps: int = 500,
        batch_size: int = 8,
        learning_rate: float = 1e-3
    ) -> Dict:
        """
        Run associative recall comparison between ANA and baseline
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create datasets
        train_dataset = AssociativeRecallDataset(
            num_samples=num_samples,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            noise_len=noise_len
        )
        
        # Create models
        ana_config.vocab_size = vocab_size
        baseline_config.vocab_size = vocab_size
        
        ana_model = ANAModel(ana_config).to(device)
        baseline_model = BaselineSSM(baseline_config).to(device)
        
        # Count parameters
        ana_params = sum(p.numel() for p in ana_model.parameters())
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        
        print(f"ANA Parameters: {ana_params:,}")
        print(f"Baseline Parameters: {baseline_params:,}")
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create optimizers
        ana_optimizer = torch.optim.Adam(ana_model.parameters(), lr=learning_rate)
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        # Training loop for masked data
        print("Training models on associative recall...")
        
        for step in range(max_steps):
            try:
                batch_x, batch_y, batch_mask = next(iter(train_loader))
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_y, batch_mask = next(train_iter)
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            
            # Train ANA model
            ana_model.train()
            ana_optimizer.zero_grad()
            ana_logits, _ = ana_model(batch_x)
            
            # Only compute loss where mask is 1
            active_positions = batch_mask.bool()
            if active_positions.any():
                active_logits = ana_logits[active_positions]
                active_targets = batch_y[active_positions]
                if active_targets.numel() > 0:
                    ana_loss = F.cross_entropy(active_logits, active_targets)
                    ana_loss.backward()
                    torch.nn.utils.clip_grad_norm_(ana_model.parameters(), 1.0)
                    ana_optimizer.step()
            
            # Train baseline model
            baseline_model.train()
            baseline_optimizer.zero_grad()
            baseline_logits, _ = baseline_model(batch_x)
            
            # Only compute loss where mask is 1
            if active_positions.any():
                active_logits = baseline_logits[active_positions]
                active_targets = batch_y[active_positions]
                if active_targets.numel() > 0:
                    baseline_loss = F.cross_entropy(active_logits, active_targets)
                    baseline_loss.backward()
                    torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
                    baseline_optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step}: Processed batch with {active_positions.sum()} active positions")
        
        # Evaluation function for masked data
        @torch.no_grad()
        def eval_model_masked(model, dataloader, max_batches=20):
            model.eval()
            total_loss = 0
            total_active = 0
            
            batch_count = 0
            for batch_x, batch_y, batch_mask in dataloader:
                if batch_count >= max_batches:
                    break
                    
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                
                logits, _ = model(batch_x)
                
                # Only compute loss where mask is 1
                active_positions = batch_mask.bool()
                if active_positions.any():
                    active_logits = logits[active_positions]
                    active_targets = batch_y[active_positions]
                    if active_targets.numel() > 0:
                        loss = F.cross_entropy(active_logits, active_targets)
                        total_loss += loss.item() * active_targets.numel()
                        total_active += active_targets.numel()
                
                batch_count += 1
            
            if total_active > 0:
                avg_loss = total_loss / total_active
                perplexity = float(torch.exp(torch.tensor(avg_loss)))
                return avg_loss, perplexity
            else:
                return float('inf'), float('inf')
        
        # Final evaluation
        ana_val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        ana_loss, ana_ppl = eval_model_masked(ana_model, ana_val_loader)
        baseline_loss, baseline_ppl = eval_model_masked(baseline_model, ana_val_loader)
        
        # Calculate improvements
        if baseline_loss != float('inf') and ana_loss != float('inf'):
            loss_improvement = ((baseline_loss - ana_loss) / baseline_loss) * 100
        else:
            loss_improvement = 0
        
        if baseline_ppl != float('inf') and ana_ppl != float('inf'):
            ppl_improvement = ((baseline_ppl - ana_ppl) / baseline_ppl) * 100
        else:
            ppl_improvement = 0
        
        print(f"\\nFinal Results (Associative Recall):")
        print(f"ANA - Loss: {ana_loss:.4f}, Perplexity: {ana_ppl:.2f}")
        print(f"Baseline - Loss: {baseline_loss:.4f}, Perplexity: {baseline_ppl:.2f}")
        print(f"ANA Loss Improvement: {loss_improvement:.2f}%")
        print(f"ANA Perplexity Improvement: {ppl_improvement:.2f}%")
        
        results = {
            'ana_final_loss': ana_loss,
            'baseline_final_loss': baseline_loss,
            'ana_final_perplexity': ana_ppl,
            'baseline_final_perplexity': baseline_ppl,
            'loss_improvement_pct': loss_improvement,
            'perplexity_improvement_pct': ppl_improvement,
            'ana_params': ana_params,
            'baseline_params': baseline_params
        }
        
        return results
    
    def run_two_phase_training(
        self,
        config: ANAConfig,
        num_samples: int = 1000,
        vocab_size: int = 40,
        num_pairs: int = 4,
        noise_len: int = 8,
        phase1_steps: int = 300,
        phase2_steps: int = 200,
        batch_size: int = 8,
        phase1_lr: float = 1e-3,
        phase2_lr: float = 1e-4
    ) -> Dict:
        """
        Run two-phase training experiment
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create dataset
        dataset = AssociativeRecallDataset(
            num_samples=num_samples,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            noise_len=noise_len
        )
        
        # Update config
        config.vocab_size = vocab_size
        
        # Create model
        model = ANAModel(config).to(device)
        
        print(f"Two-phase training model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create two-phase trainer
        trainer = TwoPhaseTrainer(
            model=model,
            device=device,
            phase1_lr=phase1_lr,
            phase2_lr=phase2_lr
        )
        
        # Run both phases
        results = trainer.train_both_phases(dataloader, phase1_steps, phase2_steps)
        
        # Evaluate final model
        @torch.no_grad()
        def eval_model_masked(model, dataloader, max_batches=20):
            model.eval()
            total_loss = 0
            total_active = 0
            
            batch_count = 0
            for batch_x, batch_y, batch_mask in dataloader:
                if batch_count >= max_batches:
                    break
                    
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                
                logits, _ = model(batch_x)
                
                # Only compute loss where mask is 1
                active_positions = batch_mask.bool()
                if active_positions.any():
                    active_logits = logits[active_positions]
                    active_targets = batch_y[active_positions]
                    if active_targets.numel() > 0:
                        loss = F.cross_entropy(active_logits, active_targets)
                        total_loss += loss.item() * active_targets.numel()
                        total_active += active_targets.numel()
                
                batch_count += 1
            
            if total_active > 0:
                avg_loss = total_loss / total_active
                perplexity = float(torch.exp(torch.tensor(avg_loss)))
                return avg_loss, perplexity
            else:
                return float('inf'), float('inf')
        
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        final_loss, final_ppl = eval_model_masked(model, eval_loader)
        
        print(f"\\nTwo-Phase Training Results:")
        print(f"Phase 1 Average Loss: {np.mean(results['phase1_losses']):.4f}")
        print(f"Phase 2 Average Loss: {np.mean(results['phase2_losses']):.4f}")
        print(f"Final Model Loss: {final_loss:.4f}, PPL: {final_ppl:.2f}")
        
        results['final_loss'] = final_loss
        results['final_perplexity'] = final_ppl
        
        return results
    
    def save_results(self, results: Dict, filename: Optional[str] = None):
        """
        Save experiment results to JSON file
        """
        if filename is None:
            filename = f"{self.exp_name}_results_{self.timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert any tensors to floats/ints for JSON serialization
        def convert_tensors(obj):
            if torch.is_tensor(obj):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_tensors(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath


def run_comprehensive_comparison():
    """
    Run a comprehensive comparison experiment
    """
    print("Running Comprehensive ANA vs Baseline Comparison...")
    print("="*60)
    
    # Create experiment runner
    exp_runner = ExperimentRunner("comprehensive_comparison")
    
    # Define configurations
    ana_config = ANAConfig(
        d_model=64,
        state_dim=64,
        key_dim=32,
        num_layers=2,
        use_hololink=True,
        use_controller=False,
        use_parallel_scan=True
    )
    
    baseline_config = ANAConfig(
        d_model=80,
        state_dim=80,
        num_layers=2,
        use_hololink=False,
        use_controller=False
    )
    
    # Create simple text dataset
    text = "the quick brown fox jumps over the lazy dog " * 500
    train_dataset = TextDataset(text, seq_len=32, vocab_size=40)
    val_dataset = TextDataset(text, seq_len=32, vocab_size=40)
    
    # Run text generation comparison
    tg_results = exp_runner.run_text_generation_comparison(
        ana_config=ana_config,
        baseline_config=baseline_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_steps=500,
        batch_size=8
    )
    
    # Run associative recall comparison
    ar_results = exp_runner.run_associative_recall_comparison(
        ana_config=ana_config,
        baseline_config=baseline_config,
        num_samples=500,
        vocab_size=40,
        num_pairs=4,
        noise_len=6,
        max_steps=400,
        batch_size=8
    )
    
    # Run two-phase training
    tp_config = ANAConfig(
        d_model=64,
        state_dim=64,
        key_dim=32,
        num_layers=2,
        use_hololink=True,
        use_controller=True,  # Include controller for two-phase training
        use_parallel_scan=True
    )
    
    tp_results = exp_runner.run_two_phase_training(
        config=tp_config,
        num_samples=500,
        vocab_size=40,
        num_pairs=4,
        noise_len=6,
        phase1_steps=300,
        phase2_steps=200,
        batch_size=8
    )
    
    # Compile all results
    all_results = {
        'timestamp': exp_runner.timestamp,
        'text_generation': tg_results,
        'associative_recall': ar_results,
        'two_phase_training': tp_results
    }
    
    # Save results
    exp_runner.save_results(all_results)
    
    # Print summary
    print("\\n" + "="*60)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\\nText Generation:")
    print(f"  ANA Perplexity: {tg_results['ana_final_perplexity']:.2f}")
    print(f"  Baseline Perplexity: {tg_results['baseline_final_perplexity']:.2f}")
    print(f"  ANA Improvement: {tg_results['perplexity_improvement_pct']:.2f}%")
    
    print(f"\\nAssociative Recall:")
    print(f"  ANA Perplexity: {ar_results['ana_final_perplexity']:.2f}")
    print(f"  Baseline Perplexity: {ar_results['baseline_final_perplexity']:.2f}")
    print(f"  ANA Improvement: {ar_results['perplexity_improvement_pct']:.2f}%")
    
    print(f"\\nTwo-Phase Training:")
    print(f"  Final Perplexity: {tp_results['final_perplexity']:.2f}")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_comparison()