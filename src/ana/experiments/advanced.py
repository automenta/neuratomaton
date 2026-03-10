"""
Advanced experiment runner for ANA models
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
import matplotlib.pyplot as plt

from ..models.config import ANAConfig
from ..models.core import ANAModel, BaselineSSM
from ..training.advanced_utils import (
    AdvancedTrainer, 
    CurriculumLearningScheduler, 
    ModelAnalyzer, 
    VisualizationTools,
    ModelProfiler,
    create_adaptive_optimizer
)
from ..utils.datasets import AssociativeRecallDataset, TextDataset
from ..utils.analysis import ModelVisualizer, PerformanceAnalyzer


class AdvancedExperimentRunner:
    """
    Advanced experiment runner with comprehensive analysis
    """
    def __init__(self, exp_name: str = "advanced_ana_experiment"):
        self.exp_name = exp_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"ana.advanced_experiments.{exp_name}")
        self.logger.setLevel(logging.INFO)
        
        # Create results directory
        self.results_dir = os.path.join("results", exp_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create figures directory
        self.figures_dir = os.path.join(self.results_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Create checkpoints directory
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def run_advanced_text_generation_comparison(
        self,
        ana_config: ANAConfig,
        baseline_config: ANAConfig,
        train_dataset,
        val_dataset,
        max_steps: int = 1000,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        use_curriculum: bool = False,
        analyze_model: bool = True
    ) -> Dict:
        """
        Run advanced text generation comparison with comprehensive analysis
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
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
        ana_optimizer = create_adaptive_optimizer(ana_model, base_lr=learning_rate, optimizer_type="adam")
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        # Create advanced trainers
        ana_trainer = AdvancedTrainer(
            ana_model, 
            ana_optimizer, 
            device=device,
            save_checkpoint_every=200,
            checkpoint_dir=self.checkpoints_dir
        )
        
        baseline_trainer = AdvancedTrainer(
            baseline_model, 
            baseline_optimizer, 
            device=device,
            save_checkpoint_every=200,
            checkpoint_dir=self.checkpoints_dir
        )
        
        # Curriculum learning if enabled
        curriculum_scheduler = None
        if use_curriculum:
            curriculum_scheduler = CurriculumLearningScheduler(
                initial_difficulty=0.1,
                max_difficulty=1.0,
                schedule_type="linear",
                milestones=[max_steps//4, max_steps//2, 3*max_steps//4]
            )
        
        # Training loop
        print("Training models...")
        ana_train_losses = []
        baseline_train_losses = []
        
        train_iter = iter(train_loader)
        
        for step in range(max_steps):
            try:
                batch_x, batch_y = next(train_iter)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_y = next(train_iter)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Train both models
            ana_loss = ana_trainer.train_step(batch_x, batch_y)
            baseline_loss = baseline_trainer.train_step(batch_x, batch_y)
            
            ana_train_losses.append(ana_loss)
            baseline_train_losses.append(baseline_loss)
            
            if step % 200 == 0:
                print(f"Step {step}: ANA Loss: {ana_loss:.4f}, Baseline Loss: {baseline_loss:.4f}")
        
        # Final evaluation
        ana_val_loss, ana_ppl, ana_metrics = ana_trainer.evaluate(val_loader)
        baseline_val_loss, baseline_ppl, baseline_metrics = baseline_trainer.evaluate(val_loader)
        
        # Calculate improvements
        loss_improvement = ((baseline_val_loss - ana_val_loss) / baseline_val_loss) * 100
        ppl_improvement = ((baseline_ppl - ana_ppl) / baseline_ppl) * 100
        acc_improvement = ((ana_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy']) * 100 if baseline_metrics['accuracy'] > 0 else 0
        
        print(f"\\nFinal Results:")
        print(f"ANA - Loss: {ana_val_loss:.4f}, Perplexity: {ana_ppl:.2f}, Accuracy: {ana_metrics['accuracy']:.4f}")
        print(f"Baseline - Loss: {baseline_val_loss:.4f}, Perplexity: {baseline_ppl:.2f}, Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"ANA Loss Improvement: {loss_improvement:.2f}%")
        print(f"ANA Perplexity Improvement: {ppl_improvement:.2f}%")
        print(f"ANA Accuracy Improvement: {acc_improvement:.2f}%")
        
        # Model analysis if requested
        analysis_results = {}
        if analyze_model:
            print("\\nPerforming model analysis...")
            
            # Analyze ANA model
            ana_analyzer = ModelAnalyzer(ana_model)
            ana_grad_stats = ana_analyzer.analyze_gradients()
            ana_param_stats = ana_analyzer.analyze_parameter_distribution()
            ana_holo_stats = ana_analyzer.analyze_hololink_memory()
            
            # Analyze Baseline model
            baseline_analyzer = ModelAnalyzer(baseline_model)
            baseline_grad_stats = baseline_analyzer.analyze_gradients()
            baseline_param_stats = baseline_analyzer.analyze_parameter_distribution()
            
            analysis_results = {
                'ana_gradients': ana_grad_stats,
                'ana_parameters': ana_param_stats,
                'ana_hololink': ana_holo_stats,
                'baseline_gradients': baseline_grad_stats,
                'baseline_parameters': baseline_param_stats
            }
        
        # Visualization
        viz_tools = VisualizationTools()
        viz = ModelVisualizer()
        
        # Plot training curves
        train_fig_path = os.path.join(self.figures_dir, f"training_curves_{self.timestamp}.png")
        viz.plot_training_curves(
            train_losses=ana_train_losses,
            val_losses=[ana_val_loss]*len(ana_train_losses),  # Just for visualization
            title=f"ANA Training Curve - {self.exp_name}",
            save_path=train_fig_path
        )
        
        # Save results
        results = {
            'ana_final_loss': ana_val_loss,
            'baseline_final_loss': baseline_val_loss,
            'ana_final_perplexity': ana_ppl,
            'baseline_final_perplexity': baseline_ppl,
            'ana_final_accuracy': ana_metrics['accuracy'],
            'baseline_final_accuracy': baseline_metrics['accuracy'],
            'loss_improvement_pct': loss_improvement,
            'perplexity_improvement_pct': ppl_improvement,
            'accuracy_improvement_pct': acc_improvement,
            'ana_params': ana_params,
            'baseline_params': baseline_params,
            'param_difference': param_diff,
            'param_difference_pct': param_diff_percent,
            'param_ratio': param_ratio,
            'training_history': {
                'ana_train_losses': ana_train_losses,
                'baseline_train_losses': baseline_train_losses
            },
            'analysis': analysis_results,
            'visualization_paths': {
                'training_curve': train_fig_path
            }
        }
        
        return results
    
    def run_advanced_associative_recall(
        self,
        ana_config: ANAConfig,
        baseline_config: ANAConfig,
        num_samples: int = 1000,
        vocab_size: int = 40,
        num_pairs: int = 4,
        noise_len: int = 8,
        max_steps: int = 500,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        analyze_model: bool = True
    ) -> Dict:
        """
        Run advanced associative recall experiment with detailed analysis
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Advanced Associative Recall Experiment on {device}")
        
        # Create datasets
        train_dataset = AssociativeRecallDataset(
            num_samples=num_samples,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            noise_len=noise_len
        )
        
        # Update configs
        ana_config.vocab_size = vocab_size
        baseline_config.vocab_size = vocab_size
        
        # Create models
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
        ana_optimizer = create_adaptive_optimizer(ana_model, base_lr=learning_rate, optimizer_type="adam")
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        # Training loop for masked data
        print("Training models on associative recall...")
        
        ana_train_losses = []
        baseline_train_losses = []
        
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
                    ana_train_losses.append(ana_loss.item())
            
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
                    baseline_train_losses.append(baseline_loss.item())
            
            if step % 100 == 0:
                print(f"Step {step}: ANA Loss: {ana_loss.item() if active_positions.any() and active_targets.numel() > 0 else 'N/A'}, "
                      f"Baseline Loss: {baseline_loss.item() if active_positions.any() and active_targets.numel() > 0 else 'N/A'}")
        
        # Evaluation function for masked data
        @torch.no_grad()
        def eval_model_masked(model, dataloader, max_batches=20):
            model.eval()
            total_loss = 0
            total_active = 0
            total_correct = 0
            total_predictions = 0
            
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
                        
                        # Calculate accuracy
                        predictions = torch.argmax(active_logits, dim=-1)
                        correct = (predictions == active_targets).sum().item()
                        total_correct += correct
                        total_predictions += active_targets.numel()
                
                batch_count += 1
            
            if total_active > 0:
                avg_loss = total_loss / total_active
                perplexity = float(torch.exp(torch.tensor(avg_loss)))
                accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
                return avg_loss, perplexity, accuracy
            else:
                return float('inf'), float('inf'), 0.0
        
        # Final evaluation
        ana_val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        ana_loss, ana_ppl, ana_acc = eval_model_masked(ana_model, ana_val_loader)
        baseline_loss, baseline_ppl, baseline_acc = eval_model_masked(baseline_model, ana_val_loader)
        
        # Calculate improvements
        if baseline_loss != float('inf') and ana_loss != float('inf'):
            loss_improvement = ((baseline_loss - ana_loss) / baseline_loss) * 100
        else:
            loss_improvement = 0
        
        if baseline_ppl != float('inf') and ana_ppl != float('inf'):
            ppl_improvement = ((baseline_ppl - ana_ppl) / baseline_ppl) * 100
        else:
            ppl_improvement = 0
            
        acc_improvement = ((ana_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        print(f"\\nFinal Results (Advanced Associative Recall):")
        print(f"ANA - Loss: {ana_loss:.4f}, Perplexity: {ana_ppl:.2f}, Accuracy: {ana_acc:.4f}")
        print(f"Baseline - Loss: {baseline_loss:.4f}, Perplexity: {baseline_ppl:.2f}, Accuracy: {baseline_acc:.4f}")
        print(f"ANA Loss Improvement: {loss_improvement:.2f}%")
        print(f"ANA Perplexity Improvement: {ppl_improvement:.2f}%")
        print(f"ANA Accuracy Improvement: {acc_improvement:.2f}%")
        
        # Model analysis
        analysis_results = {}
        if analyze_model:
            print("\\nPerforming model analysis...")
            
            # Analyze ANA model
            ana_analyzer = ModelAnalyzer(ana_model)
            ana_grad_stats = ana_analyzer.analyze_gradients()
            ana_param_stats = ana_analyzer.analyze_parameter_distribution()
            ana_holo_stats = ana_analyzer.analyze_hololink_memory()
            
            # Analyze Baseline model
            baseline_analyzer = ModelAnalyzer(baseline_model)
            baseline_grad_stats = baseline_analyzer.analyze_gradients()
            baseline_param_stats = baseline_analyzer.analyze_parameter_distribution()
            
            analysis_results = {
                'ana_gradients': ana_grad_stats,
                'ana_parameters': ana_param_stats,
                'ana_hololink': ana_holo_stats,
                'baseline_gradients': baseline_grad_stats,
                'baseline_parameters': baseline_param_stats
            }
        
        # Visualization
        viz = ModelVisualizer()
        
        # Plot training curves
        train_fig_path = os.path.join(self.figures_dir, f"assoc_recall_training_{self.timestamp}.png")
        viz.plot_training_curves(
            train_losses=ana_train_losses[:len(baseline_train_losses)],
            val_losses=None,
            title=f"Associative Recall Training - {self.exp_name}",
            save_path=train_fig_path
        )
        
        results = {
            'ana_final_loss': ana_loss,
            'baseline_final_loss': baseline_loss,
            'ana_final_perplexity': ana_ppl,
            'baseline_final_perplexity': baseline_ppl,
            'ana_final_accuracy': ana_acc,
            'baseline_final_accuracy': baseline_acc,
            'loss_improvement_pct': loss_improvement,
            'perplexity_improvement_pct': ppl_improvement,
            'accuracy_improvement_pct': acc_improvement,
            'ana_params': ana_params,
            'baseline_params': baseline_params,
            'training_history': {
                'ana_train_losses': ana_train_losses,
                'baseline_train_losses': baseline_train_losses
            },
            'analysis': analysis_results,
            'visualization_paths': {
                'training_curve': train_fig_path
            }
        }
        
        return results
    
    def run_scaling_analysis(
        self,
        model_sizes: List[Tuple[int, int, int]],  # List of (d_model, state_dim, num_layers)
        vocab_size: int = 40,
        num_samples: int = 500,
        batch_size: int = 8,
        max_steps: int = 300
    ) -> Dict:
        """
        Run scaling analysis with different model sizes
        """
        print(f"Running scaling analysis with {len(model_sizes)} different model sizes...")
        
        results = []
        
        for i, (d_model, state_dim, num_layers) in enumerate(model_sizes):
            print(f"\\nTraining model {i+1}/{len(model_sizes)}: d_model={d_model}, state_dim={state_dim}, layers={num_layers}")
            
            # Create config
            config = ANAConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                state_dim=state_dim,
                key_dim=max(8, d_model // 2),  # Proportional key dimension
                num_layers=num_layers,
                use_hololink=True,
                use_controller=False,
                use_parallel_scan=True
            )
            
            # Create model
            model = ANAModel(config)
            params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {params:,}")
            
            # Create simple dataset
            text = "the quick brown fox jumps over the lazy dog " * (num_samples // 10)
            dataset = TextDataset(text, seq_len=32, vocab_size=vocab_size)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Move to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Training
            train_losses = []
            for step in range(max_steps):
                try:
                    batch_x, batch_y = next(iter(dataloader))
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                except StopIteration:
                    dl_iter = iter(dataloader)
                    batch_x, batch_y = next(dl_iter)
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                model.train()
                optimizer.zero_grad()
                logits, _ = model(batch_x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                
                if step % 100 == 0:
                    print(f"  Step {step}: Loss = {loss.item():.4f}")
            
            # Evaluate
            model.eval()
            eval_loss = 0
            eval_count = 0
            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    if eval_count >= 10:  # Limit evaluation
                        break
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits, _ = model(batch_x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                    eval_loss += loss.item()
                    eval_count += 1
            
            avg_eval_loss = eval_loss / eval_count if eval_count > 0 else float('inf')
            perplexity = float(torch.exp(torch.tensor(avg_eval_loss)))
            
            results.append({
                'model_index': i,
                'd_model': d_model,
                'state_dim': state_dim,
                'num_layers': num_layers,
                'parameters': params,
                'train_loss': np.mean(train_losses[-50:]),  # Last 50 steps average
                'eval_loss': avg_eval_loss,
                'perplexity': perplexity,
                'train_time_steps': len(train_losses)
            })
            
            print(f"  Final eval loss: {avg_eval_loss:.4f}, perplexity: {perplexity:.2f}")
        
        # Create scaling visualization
        scaling_df = {
            'parameters': [r['parameters'] for r in results],
            'perplexity': [r['perplexity'] for r in results],
            'eval_loss': [r['eval_loss'] for r in results],
            'd_model': [r['d_model'] for r in results],
            'num_layers': [r['num_layers'] for r in results]
        }
        
        # Plot scaling results
        viz = ModelVisualizer()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.scatter(scaling_df['parameters'], scaling_df['perplexity'], alpha=0.7)
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity vs Parameters')
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(scaling_df['parameters'], scaling_df['eval_loss'], alpha=0.7)
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Evaluation Loss')
        ax2.set_title('Evaluation Loss vs Parameters')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        scaling_fig_path = os.path.join(self.figures_dir, f"scaling_analysis_{self.timestamp}.png")
        plt.savefig(scaling_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        scaling_results = {
            'scaling_data': results,
            'visualization_path': scaling_fig_path
        }
        
        return scaling_results
    
    def save_results(self, results: Dict, filename: Optional[str] = None):
        """
        Save experiment results to JSON file
        """
        if filename is None:
            filename = f"{self.exp_name}_advanced_results_{self.timestamp}.json"
        
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


def run_advanced_comprehensive_experiment():
    """
    Run a comprehensive advanced experiment
    """
    print("Running Advanced ANA vs Baseline Comprehensive Experiment...")
    print("="*70)
    
    # Create experiment runner
    exp_runner = AdvancedExperimentRunner("advanced_comprehensive")
    
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
        d_model=80,  # Slightly larger to compensate for missing HoloLink
        state_dim=80,
        num_layers=2,
        use_hololink=False,
        use_controller=False
    )
    
    # Create simple text dataset
    text = "the quick brown fox jumps over the lazy dog " * 500
    train_dataset = TextDataset(text, seq_len=32, vocab_size=40)
    val_dataset = TextDataset(text, seq_len=32, vocab_size=40)
    
    # Run advanced text generation comparison
    print("\\n1. Running Advanced Text Generation Comparison...")
    tg_results = exp_runner.run_advanced_text_generation_comparison(
        ana_config=ana_config,
        baseline_config=baseline_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_steps=500,
        batch_size=8
    )
    
    # Run advanced associative recall
    print("\\n2. Running Advanced Associative Recall...")
    ar_results = exp_runner.run_advanced_associative_recall(
        ana_config=ana_config,
        baseline_config=baseline_config,
        num_samples=500,
        vocab_size=40,
        num_pairs=4,
        noise_len=6,
        max_steps=400,
        batch_size=8
    )
    
    # Run scaling analysis
    print("\\n3. Running Scaling Analysis...")
    scaling_configs = [
        (32, 32, 1),
        (48, 48, 1), 
        (64, 64, 1),
        (32, 32, 2),
        (48, 48, 2),
        (64, 64, 2)
    ]
    scaling_results = exp_runner.run_scaling_analysis(
        model_sizes=scaling_configs,
        vocab_size=40,
        num_samples=300,
        batch_size=8,
        max_steps=200
    )
    
    # Compile all results
    all_results = {
        'timestamp': exp_runner.timestamp,
        'experiment_name': exp_runner.exp_name,
        'text_generation': tg_results,
        'associative_recall': ar_results,
        'scaling_analysis': scaling_results
    }
    
    # Save results
    exp_runner.save_results(all_results)
    
    # Print summary
    print("\\n" + "="*70)
    print("ADVANCED COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*70)
    
    print(f"\\nText Generation:")
    print(f"  ANA Perplexity: {tg_results['ana_final_perplexity']:.2f}")
    print(f"  Baseline Perplexity: {tg_results['baseline_final_perplexity']:.2f}")
    print(f"  ANA Improvement: {tg_results['perplexity_improvement_pct']:.2f}%")
    print(f"  ANA Accuracy: {tg_results['ana_final_accuracy']:.4f}")
    print(f"  Baseline Accuracy: {tg_results['baseline_final_accuracy']:.4f}")
    
    print(f"\\nAssociative Recall:")
    print(f"  ANA Perplexity: {ar_results['ana_final_perplexity']:.2f}")
    print(f"  Baseline Perplexity: {ar_results['baseline_final_perplexity']:.2f}")
    print(f"  ANA Improvement: {ar_results['perplexity_improvement_pct']:.2f}%")
    print(f"  ANA Accuracy: {ar_results['ana_final_accuracy']:.4f}")
    print(f"  Baseline Accuracy: {ar_results['baseline_final_accuracy']:.4f}")
    
    print(f"\\nScaling Analysis:")
    print(f"  Tested {len(scaling_results['scaling_data'])} model configurations")
    best_config = min(scaling_results['scaling_data'], key=lambda x: x['perplexity'])
    print(f"  Best config: {best_config['d_model']}d, {best_config['num_layers']} layers")
    print(f"  Best perplexity: {best_config['perplexity']:.2f}")
    print(f"  Parameters: {best_config['parameters']:,}")
    
    print(f"\\nResults saved to: {exp_runner.results_dir}")
    print(f"Figures saved to: {exp_runner.figures_dir}")
    print(f"Checkpoints saved to: {exp_runner.checkpoints_dir}")
    
    return all_results


if __name__ == "__main__":
    results = run_advanced_comprehensive_experiment()