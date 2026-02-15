"""
Comprehensive Experiment Framework for ANA (Adaptive Neural Automaton)

This framework facilitates systematic comparisons between ANA and baseline models,
enabling discovery of performance characteristics and validation of the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    vocab_size: int = 40
    d_model: int = 64
    state_dim: int = 64
    key_dim: int = 32
    num_layers: int = 2
    use_hololink: bool = True
    use_controller: bool = False
    use_parallel_scan: bool = True
    max_position: int = 512
    track_count: int = 1


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters"""
    dataset_name: str = "simple_text"
    task_type: str = "text_generation"  # Options: text_generation, associative_recall, copy_task
    max_epochs: int = 10
    batch_size: int = 8
    seq_len: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class TextDataset:
    """Simple text dataset for language modeling"""
    def __init__(self, text: str, seq_len: int = 32, vocab_size: Optional[int] = None):
        chars = sorted(list(set(text)))
        if vocab_size and len(chars) < vocab_size:
            # Add padding characters if needed
            extra_chars = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in chars]
            chars.extend(extra_chars[:vocab_size - len(chars)])
        
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, (len(self.data) - self.seq_len) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        if end > len(self.data):
            # Pad if needed
            seq = self.data[start:end]
            if len(seq) < self.seq_len + 1:
                padded = torch.full((self.seq_len + 1,), 0, dtype=torch.long)
                padded[:len(seq)] = seq
                seq = padded
        else:
            seq = self.data[start:end]
        return seq[:-1], seq[1:]


class AssociativeRecallDataset:
    """Dataset for associative recall tasks"""
    def __init__(self, num_samples: int = 1000, vocab_size: int = 40, num_pairs: int = 4, noise_len: int = 8):
        self.samples = []
        self.TOK_KEY, self.TOK_VAL, self.TOK_QUERY = 1, 2, 3
        content_range = list(range(4, vocab_size))
        
        for _ in range(num_samples):
            # Select unique keys and values
            keys = np.random.choice(content_range, size=num_pairs, replace=False)
            vals = np.random.choice([x for x in content_range if x not in keys], size=num_pairs, replace=False)
            
            # Create KV pairs
            kv_seq = []
            for k, v in zip(keys, vals):
                kv_seq.extend([self.TOK_KEY, k, self.TOK_VAL, v])
            
            # Add noise
            noise = np.random.choice(content_range, size=noise_len)
            kv_seq.extend(noise)
            
            # Add query
            query_idx = np.random.randint(0, num_pairs)
            query_key = keys[query_idx]
            target_val = vals[query_idx]
            
            kv_seq.extend([self.TOK_QUERY, query_key, target_val])
            
            # Convert to tensor
            x = torch.tensor(kv_seq[:-1], dtype=torch.long)
            y = torch.tensor(kv_seq[1:], dtype=torch.long)
            
            # Mask - only care about predicting the final value
            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0
            
            self.samples.append((x, y, mask))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class ModelFactory:
    """Factory for creating ANA and baseline models with consistent parameter counting"""
    
    @staticmethod
    def create_ana_model(config: ModelConfig):
        from ana import ANAModel
        return ANAModel(config)
    
    @staticmethod
    def create_baseline_model(config: ModelConfig):
        from ana import BaselineSSM
        return BaselineSSM(config)
    
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_parameter_efficient_configs(target_params: int = 100000) -> Tuple[ModelConfig, ModelConfig]:
        """Generate config pairs with similar parameter counts"""
        # Try different combinations to get close to target parameters
        configs = []
        
        # ANA configs
        ana_configs = [
            ModelConfig(d_model=32, state_dim=32, key_dim=16, num_layers=1, use_hololink=True),
            ModelConfig(d_model=48, state_dim=48, key_dim=24, num_layers=1, use_hololink=True),
            ModelConfig(d_model=24, state_dim=24, key_dim=12, num_layers=2, use_hololink=True),
            ModelConfig(d_model=64, state_dim=64, key_dim=32, num_layers=1, use_hololink=True),
        ]
        
        # Baseline configs
        baseline_configs = [
            ModelConfig(d_model=48, state_dim=48, num_layers=2, use_hololink=False),
            ModelConfig(d_model=64, state_dim=64, num_layers=2, use_hololink=False),
            ModelConfig(d_model=32, state_dim=32, num_layers=4, use_hololink=False),
            ModelConfig(d_model=80, state_dim=80, num_layers=1, use_hololink=False),
        ]
        
        best_pair = None
        min_diff = float('inf')
        
        for ana_cfg in ana_configs:
            ana_model = ModelFactory.create_ana_model(ana_cfg)
            ana_params = ModelFactory.count_parameters(ana_model)
            
            for base_cfg in baseline_configs:
                base_model = ModelFactory.create_baseline_model(base_cfg)
                base_params = ModelFactory.count_parameters(base_model)
                
                diff = abs(ana_params - base_params)
                if diff < min_diff and abs(ana_params - target_params) < target_params * 0.5:
                    min_diff = diff
                    best_pair = (ana_cfg, base_cfg)
        
        if best_pair is None:
            # Fallback: use default configs
            ana_cfg = ModelConfig(d_model=32, state_dim=32, key_dim=16, num_layers=1, use_hololink=True)
            base_cfg = ModelConfig(d_model=64, state_dim=64, num_layers=1, use_hololink=False)
            best_pair = (ana_cfg, base_cfg)
        
        return best_pair


class Trainer:
    """Generic trainer for both ANA and baseline models"""
    
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, batch_x, batch_y):
        self.model.train()
        self.optimizer.zero_grad()
        
        if hasattr(self.model, 'forward'):  # ANA/Baseline models
            logits, _ = self.model(batch_x)
        else:
            logits = self.model(batch_x)
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, dataloader, max_batches=10):
        self.model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                if hasattr(self.model, 'forward'):
                    logits, _ = self.model(batch_x)
                else:
                    logits = self.model(batch_x)
                
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else float('inf')


class ExperimentRunner:
    """Main class to run comparative experiments"""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.exp_config = exp_config
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set random seeds
        torch.manual_seed(exp_config.seed)
        np.random.seed(exp_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(exp_config.seed)
    
    def create_datasets(self):
        """Create datasets based on experiment config"""
        if self.exp_config.task_type == "text_generation":
            text = "the quick brown fox jumps over the lazy dog " * 500
            dataset = TextDataset(text, seq_len=self.exp_config.seq_len)
            return dataset, dataset  # train and val are same for this simple case
        
        elif self.exp_config.task_type == "associative_recall":
            dataset = AssociativeRecallDataset(num_samples=1000, vocab_size=40, num_pairs=4, noise_len=8)
            # For now, return the same dataset for train/val
            return dataset, dataset
        
        else:
            raise ValueError(f"Unknown task type: {self.exp_config.task_type}")
    
    def create_dataloaders(self, train_dataset, val_dataset):
        """Create data loaders"""
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.exp_config.batch_size, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.exp_config.batch_size, 
            shuffle=False
        )
        return train_loader, val_loader
    
    def run_comparison(self, ana_config: ModelConfig, baseline_config: ModelConfig, 
                      train_loader, val_loader, max_steps: int = 1000):
        """Run comparison between ANA and baseline models"""
        
        device = self.exp_config.device
        
        # Create models
        ana_model = ModelFactory.create_ana_model(ana_config)
        baseline_model = ModelFactory.create_baseline_model(baseline_config)
        
        # Move to device
        ana_model = ana_model.to(device)
        baseline_model = baseline_model.to(device)
        
        # Count parameters
        ana_params = ModelFactory.count_parameters(ana_model)
        baseline_params = ModelFactory.count_parameters(baseline_model)
        
        print(f"ANA Parameters: {ana_params:,}")
        print(f"Baseline Parameters: {baseline_params:,}")
        print(f"Parameter Difference: {abs(ana_params - baseline_params):,} "
              f"({abs(ana_params - baseline_params)/min(ana_params, baseline_params)*100:.2f}%)")
        
        # Create optimizers
        ana_optimizer = torch.optim.Adam(ana_model.parameters(), 
                                         lr=self.exp_config.learning_rate,
                                         weight_decay=self.exp_config.weight_decay)
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(),
                                              lr=self.exp_config.learning_rate,
                                              weight_decay=self.exp_config.weight_decay)
        
        # Create trainers
        ana_trainer = Trainer(ana_model, ana_optimizer, device)
        baseline_trainer = Trainer(baseline_model, baseline_optimizer, device)
        
        # Training loop
        results = {
            'ana_train_losses': [],
            'baseline_train_losses': [],
            'ana_val_losses': [],
            'baseline_val_losses': [],
            'ana_perplexities': [],
            'baseline_perplexities': [],
            'steps': []
        }
        
        print("Starting training...")
        for step in range(max_steps):
            # Get batch
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
            
            # Log training losses periodically
            if step % 100 == 0:
                # Evaluate on validation set
                ana_val_loss = ana_trainer.eval_step(val_loader)
                baseline_val_loss = baseline_trainer.eval_step(val_loader)
                
                ana_perplexity = float(torch.exp(torch.tensor(ana_val_loss)))
                baseline_perplexity = float(torch.exp(torch.tensor(baseline_val_loss)))
                
                results['ana_train_losses'].append(ana_loss)
                results['baseline_train_losses'].append(baseline_loss)
                results['ana_val_losses'].append(ana_val_loss)
                results['baseline_val_losses'].append(baseline_val_loss)
                results['ana_perplexities'].append(ana_perplexity)
                results['baseline_perplexities'].append(baseline_perplexity)
                results['steps'].append(step)
                
                print(f"Step {step}:")
                print(f"  ANA - Train: {ana_loss:.4f}, Val: {ana_val_loss:.4f}, PPL: {ana_perplexity:.2f}")
                print(f"  Baseline - Train: {baseline_loss:.4f}, Val: {baseline_val_loss:.4f}, PPL: {baseline_perplexity:.2f}")
        
        # Final evaluation
        final_ana_val_loss = ana_trainer.eval_step(val_loader)
        final_baseline_val_loss = baseline_trainer.eval_step(val_loader)
        
        final_ana_perplexity = float(torch.exp(torch.tensor(final_ana_val_loss)))
        final_baseline_perplexity = float(torch.exp(torch.tensor(final_baseline_val_loss)))
        
        # Calculate improvements
        loss_improvement = ((final_baseline_val_loss - final_ana_val_loss) / final_baseline_val_loss) * 100
        ppl_improvement = ((final_baseline_perplexity - final_ana_perplexity) / final_baseline_perplexity) * 100
        
        print(f"\nFinal Results:")
        print(f"ANA Final Loss: {final_ana_val_loss:.4f}, PPL: {final_ana_perplexity:.2f}")
        print(f"Baseline Final Loss: {final_baseline_val_loss:.4f}, PPL: {final_baseline_perplexity:.2f}")
        print(f"ANA Loss Improvement: {loss_improvement:.2f}%")
        print(f"ANA Perplexity Improvement: {ppl_improvement:.2f}%")
        
        # Store results
        comparison_results = {
            'ana_final_loss': final_ana_val_loss,
            'baseline_final_loss': final_baseline_val_loss,
            'ana_final_perplexity': final_ana_perplexity,
            'baseline_final_perplexity': final_baseline_perplexity,
            'loss_improvement_pct': loss_improvement,
            'perplexity_improvement_pct': ppl_improvement,
            'ana_params': ana_params,
            'baseline_params': baseline_params,
            'param_difference': abs(ana_params - baseline_params),
            'param_difference_pct': abs(ana_params - baseline_params)/min(ana_params, baseline_params)*100,
            'training_history': results
        }
        
        return comparison_results
    
    def run_parameter_sweep(self, param_ranges: Dict[str, List], num_trials: int = 5):
        """Run parameter sweep to find optimal configurations"""
        print("Running parameter sweep...")
        
        results = []
        
        # Generate parameter combinations
        import itertools
        
        keys, values = zip(*param_ranges.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for i, params in enumerate(combinations[:num_trials]):  # Limit trials for demo
            print(f"\nTrial {i+1}/{min(num_trials, len(combinations))}: {params}")
            
            # Create configs with these parameters
            ana_config = ModelConfig(**params, use_hololink=True)
            baseline_config = ModelFactory.get_parameter_efficient_configs(
                target_params=ModelFactory.count_parameters(ModelFactory.create_ana_model(ana_config))
            )[1]
            
            # Update baseline config with compatible parameters
            baseline_config.d_model = params.get('d_model', 32)
            baseline_config.state_dim = params.get('state_dim', 32)
            baseline_config.num_layers = params.get('num_layers', 2)
            baseline_config.vocab_size = params.get('vocab_size', 40)
            
            # Create datasets
            train_dataset, val_dataset = self.create_datasets()
            train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
            
            # Run comparison
            trial_result = self.run_comparison(ana_config, baseline_config, 
                                             train_loader, val_loader, max_steps=200)
            trial_result['trial_params'] = params
            results.append(trial_result)
        
        return results
    
    def save_results(self, results, filename=None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"experiment_results_{self.timestamp}.json"
        
        filepath = os.path.join("results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
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
    
    def plot_results(self, results):
        """Plot comparison results"""
        if 'training_history' in results:
            history = results['training_history']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot training losses
            axes[0, 0].plot(history['steps'], history['ana_train_losses'], label='ANA Train', marker='o')
            axes[0, 0].plot(history['steps'], history['baseline_train_losses'], label='Baseline Train', marker='s')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot validation losses
            axes[0, 1].plot(history['steps'], history['ana_val_losses'], label='ANA Val', marker='o')
            axes[0, 1].plot(history['baseline_val_losses'], label='Baseline Val', marker='s')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot perplexities
            axes[1, 0].plot(history['steps'], history['ana_perplexities'], label='ANA PPL', marker='o')
            axes[1, 0].plot(history['baseline_perplexities'], label='Baseline PPL', marker='s')
            axes[1, 0].set_title('Perplexity')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Perplexity')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Parameter comparison
            labels = ['ANA', 'Baseline']
            params = [results['ana_params'], results['baseline_params']]
            axes[1, 1].bar(labels, params)
            axes[1, 1].set_title('Parameter Count')
            axes[1, 1].set_ylabel('Number of Parameters')
            
            plt.tight_layout()
            plt.savefig(f"results/comparison_plot_{self.timestamp}.png")
            plt.show()


def run_basic_comparison():
    """Run a basic comparison experiment"""
    print("Running basic ANA vs Baseline comparison...")
    
    exp_config = ExperimentConfig(
        task_type="text_generation",
        max_epochs=1,
        batch_size=8,
        seq_len=32,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    runner = ExperimentRunner(exp_config)
    
    # Get parameter-efficient configs
    ana_config, baseline_config = ModelFactory.get_parameter_efficient_configs(target_params=50000)
    
    # Override vocab size to match dataset
    ana_config.vocab_size = 40
    baseline_config.vocab_size = 40
    
    # Create datasets
    train_dataset, val_dataset = runner.create_datasets()
    train_loader, val_loader = runner.create_dataloaders(train_dataset, val_dataset)
    
    # Run comparison
    results = runner.run_comparison(ana_config, baseline_config, train_loader, val_loader, max_steps=500)
    
    # Save results
    runner.save_results(results)
    
    # Plot results
    runner.plot_results(results)
    
    return results


def run_associative_recall_experiment():
    """Run associative recall experiment"""
    print("Running associative recall experiment...")
    
    exp_config = ExperimentConfig(
        task_type="associative_recall",
        max_epochs=1,
        batch_size=8,
        seq_len=32,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    runner = ExperimentRunner(exp_config)
    
    # Use configs optimized for associative tasks
    ana_config = ModelConfig(
        vocab_size=40,
        d_model=48,
        state_dim=48,
        key_dim=24,
        num_layers=1,
        use_hololink=True,
        use_controller=False,
        use_parallel_scan=True
    )
    
    baseline_config = ModelConfig(
        vocab_size=40,
        d_model=64,
        state_dim=64,
        num_layers=2,
        use_hololink=False
    )
    
    # Create datasets
    train_dataset, val_dataset = runner.create_datasets()
    train_loader, val_loader = runner.create_dataloaders(train_dataset, val_dataset)
    
    # Run comparison
    results = runner.run_comparison(ana_config, baseline_config, train_loader, val_loader, max_steps=500)
    
    # Save results
    runner.save_results(results, f"assoc_recall_results_{runner.timestamp}.json")
    
    return results


def run_two_phase_training_experiment():
    """Run two-phase training experiment as described in the research"""
    print("Running two-phase training experiment...")
    
    from ana import ANAModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with both hololink and controller
    config = ModelConfig(
        vocab_size=40,
        d_model=64,
        state_dim=64,
        key_dim=32,
        num_layers=2,
        use_hololink=True,
        use_controller=True,
        use_parallel_scan=True
    )
    
    model = ANAModel(config).to(device)
    
    # Create dataset
    text = "the quick brown fox jumps over the lazy dog " * 500
    dataset = TextDataset(text, seq_len=32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Phase 1: Train HoloLink only (freeze controller)
    print("Phase 1: Training HoloLink only...")
    for name, param in model.named_parameters():
        if 'controller' in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    optimizer_phase1 = torch.optim.Adam([
        p for n, p in model.named_parameters() if p.requires_grad
    ], lr=1e-3)
    
    trainer_phase1 = Trainer(model, optimizer_phase1, device)
    
    # Train for phase 1
    for step in range(300):
        try:
            batch_x, batch_y = next(iter(dataloader))
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch_x, batch_y = next(dl_iter)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        loss = trainer_phase1.train_step(batch_x, batch_y)
        
        if step % 100 == 0:
            print(f"Phase 1 Step {step}: Loss = {loss:.4f}")
    
    phase1_loss = trainer_phase1.eval_step(dataloader)
    print(f"Phase 1 Final Loss: {phase1_loss:.4f}")
    
    # Phase 2: Fine-tune controller (freeze HoloLink)
    print("Phase 2: Fine-tuning controller...")
    for name, param in model.named_parameters():
        if 'holo' in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    optimizer_phase2 = torch.optim.Adam([
        p for n, p in model.named_parameters() if p.requires_grad
    ], lr=1e-4)  # Lower LR for fine-tuning
    
    trainer_phase2 = Trainer(model, optimizer_phase2, device)
    
    # Train for phase 2
    for step in range(200):
        try:
            batch_x, batch_y = next(iter(dataloader))
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch_x, batch_y = next(dl_iter)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        loss = trainer_phase2.train_step(batch_x, batch_y)
        
        if step % 100 == 0:
            print(f"Phase 2 Step {step}: Loss = {loss:.4f}")
    
    phase2_loss = trainer_phase2.eval_step(dataloader)
    print(f"Phase 2 Final Loss: {phase2_loss:.4f}")
    
    print(f"Two-phase training complete. Loss improved from {phase1_loss:.4f} to {phase2_loss:.4f}")
    
    return phase1_loss, phase2_loss


if __name__ == "__main__":
    print("ANA Experiment Framework")
    print("="*50)
    
    # Run basic comparison
    basic_results = run_basic_comparison()
    
    # Run associative recall experiment
    assoc_results = run_associative_recall_experiment()
    
    # Run two-phase training experiment
    phase1_loss, phase2_loss = run_two_phase_training_experiment()
    
    print("\\nExperiment Summary:")
    print(f"Basic Comparison - ANA PPL: {basic_results['ana_final_perplexity']:.2f}, "
          f"Baseline PPL: {basic_results['baseline_final_perplexity']:.2f}")
    print(f"Associative Recall - ANA PPL: {assoc_results['ana_final_perplexity']:.2f}, "
          f"Baseline PPL: {assoc_results['baseline_final_perplexity']:.2f}")
    print(f"Two-Phase Training - Phase 1 Loss: {phase1_loss:.4f}, Phase 2 Loss: {phase2_loss:.4f}")