"""
Quick Experiment Runner for ANA (Adaptive Neural Automaton)

This script runs focused experiments to demonstrate ANA's capabilities
and compare against baseline models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ana import ANAConfig, ANAModel, BaselineSSM
import os
import json
from datetime import datetime


def create_text_dataset(text, seq_len=32, vocab_size=40):
    """Create a simple text dataset"""
    chars = sorted(list(set(text)))
    if len(chars) < vocab_size:
        # Add padding characters if needed
        extra_chars = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in chars]
        chars.extend(extra_chars[:vocab_size - len(chars)])
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    def get_batch(batch_size, seq_len):
        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
        return x, y
    
    return get_batch, len(chars)


def create_kv_dataset(num_samples=1000, vocab_size=40, num_pairs=4, noise_len=8):
    """Create associative recall dataset"""
    samples = []
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    content_range = list(range(4, vocab_size))
    
    for _ in range(num_samples):
        # Select unique keys and values
        keys = np.random.choice(content_range, size=num_pairs, replace=False)
        vals = np.random.choice([x for x in content_range if x not in keys], size=num_pairs, replace=False)
        
        # Create KV pairs
        kv_seq = []
        for k, v in zip(keys, vals):
            kv_seq.extend([TOK_KEY, k, TOK_VAL, v])
        
        # Add noise
        noise = np.random.choice(content_range, size=noise_len)
        kv_seq.extend(noise)
        
        # Add query
        query_idx = np.random.randint(0, num_pairs)
        query_key = keys[query_idx]
        target_val = vals[query_idx]
        
        kv_seq.extend([TOK_QUERY, query_key, target_val])
        
        # Convert to tensor
        x = torch.tensor(kv_seq[:-1], dtype=torch.long)
        y = torch.tensor(kv_seq[1:], dtype=torch.long)
        
        # Mask - only care about predicting the final value
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[-1] = 1.0
        
        samples.append((x, y, mask))
    
    def get_batch(batch_size):
        indices = np.random.choice(len(samples), size=batch_size)
        batch_x = torch.stack([samples[i][0] for i in indices])
        batch_y = torch.stack([samples[i][1] for i in indices])
        batch_mask = torch.stack([samples[i][2] for i in indices])
        return batch_x, batch_y, batch_mask
    
    return get_batch


class ModelTrainer:
    """Simple trainer for both ANA and baseline models"""
    
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train_step(self, batch_x, batch_y):
        self.model.train()
        self.optimizer.zero_grad()
        
        logits, _ = self.model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def eval_model(self, get_batch_func, num_batches=10):
        self.model.eval()
        total_loss = 0
        count = 0
        
        for _ in range(num_batches):
            batch_x, batch_y = get_batch_func(8, 32)  # Fixed batch size and seq len
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            logits, _ = self.model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count
        perplexity = float(torch.exp(torch.tensor(avg_loss)))
        return avg_loss, perplexity


def run_text_generation_comparison():
    """Run text generation comparison between ANA and baseline"""
    print("Running Text Generation Comparison...")
    print("-" * 50)
    
    # Create dataset
    text = "the quick brown fox jumps over the lazy dog " * 500
    get_batch, vocab_size = create_text_dataset(text, seq_len=32, vocab_size=40)
    
    # Create models with similar parameter counts
    ana_config = ANAConfig(
        vocab_size=vocab_size,
        d_model=48,
        state_dim=48,
        key_dim=24,
        num_layers=1,
        use_hololink=True,
        use_controller=False,
        use_parallel_scan=True
    )
    
    baseline_config = ANAConfig(
        vocab_size=vocab_size,
        d_model=64,
        state_dim=64,
        num_layers=2,
        use_hololink=False,
        use_controller=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    ana_model = ANAModel(ana_config).to(device)
    baseline_model = BaselineSSM(baseline_config).to(device)
    
    ana_params = sum(p.numel() for p in ana_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    print(f"ANA Parameters: {ana_params:,}")
    print(f"Baseline Parameters: {baseline_params:,}")
    print(f"Parameter Difference: {abs(ana_params - baseline_params):,}")
    
    # Create optimizers
    ana_optimizer = torch.optim.Adam(ana_model.parameters(), lr=1e-3)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    
    # Create trainers
    ana_trainer = ModelTrainer(ana_model, ana_optimizer, device)
    baseline_trainer = ModelTrainer(baseline_model, baseline_optimizer, device)
    
    # Training loop
    print("\\nTraining models...")
    for step in range(500):
        # Get batch
        batch_x, batch_y = get_batch(8, 32)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Train both models
        ana_loss = ana_trainer.train_step(batch_x, batch_y)
        baseline_loss = baseline_trainer.train_step(batch_x, batch_y)
        
        if step % 100 == 0:
            print(f"Step {step}: ANA Loss: {ana_loss:.4f}, Baseline Loss: {baseline_loss:.4f}")
    
    # Final evaluation
    ana_loss, ana_ppl = ana_trainer.eval_model(get_batch)
    baseline_loss, baseline_ppl = baseline_trainer.eval_model(get_batch)
    
    print(f"\\nFinal Results:")
    print(f"ANA - Loss: {ana_loss:.4f}, Perplexity: {ana_ppl:.2f}")
    print(f"Baseline - Loss: {baseline_loss:.4f}, Perplexity: {baseline_ppl:.2f}")
    
    # Calculate improvements
    loss_improvement = ((baseline_loss - ana_loss) / baseline_loss) * 100
    ppl_improvement = ((baseline_ppl - ana_ppl) / baseline_ppl) * 100
    
    print(f"ANA Loss Improvement: {loss_improvement:.2f}%")
    print(f"ANA Perplexity Improvement: {ppl_improvement:.2f}%")
    
    results = {
        'ana_loss': ana_loss,
        'baseline_loss': baseline_loss,
        'ana_perplexity': ana_ppl,
        'baseline_perplexity': baseline_ppl,
        'loss_improvement_pct': loss_improvement,
        'perplexity_improvement_pct': ppl_improvement,
        'ana_params': ana_params,
        'baseline_params': baseline_params
    }
    
    return results


def run_associative_recall_comparison():
    """Run associative recall comparison"""
    print("\\nRunning Associative Recall Comparison...")
    print("-" * 50)
    
    # Create dataset
    get_batch = create_kv_dataset(num_samples=1000, vocab_size=40, num_pairs=4, noise_len=8)
    
    # Create models
    ana_config = ANAConfig(
        vocab_size=40,
        d_model=48,
        state_dim=48,
        key_dim=24,
        num_layers=1,
        use_hololink=True,
        use_controller=False,
        use_parallel_scan=True
    )
    
    baseline_config = ANAConfig(
        vocab_size=40,
        d_model=64,
        state_dim=64,
        num_layers=2,
        use_hololink=False,
        use_controller=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    ana_model = ANAModel(ana_config).to(device)
    baseline_model = BaselineSSM(baseline_config).to(device)
    
    ana_params = sum(p.numel() for p in ana_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    print(f"ANA Parameters: {ana_params:,}")
    print(f"Baseline Parameters: {baseline_params:,}")
    
    # Create optimizers
    ana_optimizer = torch.optim.Adam(ana_model.parameters(), lr=1e-3)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    
    # Create trainers
    ana_trainer = ModelTrainer(ana_model, ana_optimizer, device)
    baseline_trainer = ModelTrainer(baseline_model, baseline_optimizer, device)
    
    # Training loop for associative recall
    print("\\nTraining models on associative recall...")
    
    for step in range(500):
        # Get batch (special handling for masked data)
        batch_x, batch_y, batch_mask = get_batch(8)
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
            print(f"Step {step}: Active positions found in batch")
    
    # Evaluation function for masked data
    @torch.no_grad()
    def eval_model_masked(model, get_batch_func, num_batches=10):
        model.eval()
        total_loss = 0
        count = 0
        
        for _ in range(num_batches):
            batch_x, batch_y, batch_mask = get_batch_func(8)
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            
            logits, _ = model(batch_x)
            
            # Only compute loss where mask is 1
            active_positions = batch_mask.bool()
            if active_positions.any():
                active_logits = logits[active_positions]
                active_targets = batch_y[active_positions]
                if active_targets.numel() > 0:
                    loss = F.cross_entropy(active_logits, active_targets)
                    total_loss += loss.item()
                    count += 1
        
        if count > 0:
            avg_loss = total_loss / count
            perplexity = float(torch.exp(torch.tensor(avg_loss)))
            return avg_loss, perplexity
        else:
            return float('inf'), float('inf')
    
    # Final evaluation
    ana_loss, ana_ppl = eval_model_masked(ana_model, get_batch)
    baseline_loss, baseline_ppl = eval_model_masked(baseline_model, get_batch)
    
    print(f"\\nFinal Results (Associative Recall):")
    print(f"ANA - Loss: {ana_loss:.4f}, Perplexity: {ana_ppl:.2f}")
    print(f"Baseline - Loss: {baseline_loss:.4f}, Perplexity: {baseline_ppl:.2f}")
    
    # Calculate improvements
    if baseline_loss != float('inf') and ana_loss != float('inf'):
        loss_improvement = ((baseline_loss - ana_loss) / baseline_loss) * 100
    else:
        loss_improvement = 0
    
    if baseline_ppl != float('inf') and ana_ppl != float('inf'):
        ppl_improvement = ((baseline_ppl - ana_ppl) / baseline_ppl) * 100
    else:
        ppl_improvement = 0
    
    print(f"ANA Loss Improvement: {loss_improvement:.2f}%")
    print(f"ANA Perplexity Improvement: {ppl_improvement:.2f}%")
    
    results = {
        'ana_loss': ana_loss,
        'baseline_loss': baseline_loss,
        'ana_perplexity': ana_ppl,
        'baseline_perplexity': baseline_ppl,
        'loss_improvement_pct': loss_improvement,
        'perplexity_improvement_pct': ppl_improvement,
        'ana_params': ana_params,
        'baseline_params': baseline_params
    }
    
    return results


def run_two_phase_training():
    """Run two-phase training demonstration"""
    print("\\nRunning Two-Phase Training Demonstration...")
    print("-" * 50)
    
    # Create text dataset
    text = "the quick brown fox jumps over the lazy dog " * 500
    get_batch, vocab_size = create_text_dataset(text, seq_len=32, vocab_size=40)
    
    # Create model with both hololink and controller
    config = ANAConfig(
        vocab_size=vocab_size,
        d_model=64,
        state_dim=64,
        key_dim=32,
        num_layers=2,
        use_hololink=True,
        use_controller=True,
        use_parallel_scan=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ANAModel(config).to(device)
    
    print(f"Two-phase training model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Phase 1: Train HoloLink only (freeze controller)
    print("\\nPhase 1: Training HoloLink only...")
    for name, param in model.named_parameters():
        if 'controller' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    optimizer_phase1 = torch.optim.Adam([
        p for n, p in model.named_parameters() if p.requires_grad
    ], lr=1e-3)
    
    trainer_phase1 = ModelTrainer(model, optimizer_phase1, device)
    
    for step in range(300):
        batch_x, batch_y = get_batch(8, 32)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        loss = trainer_phase1.train_step(batch_x, batch_y)
        
        if step % 100 == 0:
            print(f"Phase 1 Step {step}: Loss = {loss:.4f}")
    
    phase1_loss, phase1_ppl = trainer_phase1.eval_model(get_batch)
    print(f"Phase 1 Final - Loss: {phase1_loss:.4f}, PPL: {phase1_ppl:.2f}")
    
    # Phase 2: Fine-tune controller (freeze HoloLink)
    print("\\nPhase 2: Fine-tuning controller...")
    for name, param in model.named_parameters():
        if 'holo' in name:  # Freeze HoloLink
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    optimizer_phase2 = torch.optim.Adam([
        p for n, p in model.named_parameters() if p.requires_grad
    ], lr=1e-4)  # Lower LR for fine-tuning
    
    trainer_phase2 = ModelTrainer(model, optimizer_phase2, device)
    
    for step in range(200):
        batch_x, batch_y = get_batch(8, 32)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        loss = trainer_phase2.train_step(batch_x, batch_y)
        
        if step % 100 == 0:
            print(f"Phase 2 Step {step}: Loss = {loss:.4f}")
    
    phase2_loss, phase2_ppl = trainer_phase2.eval_model(get_batch)
    print(f"Phase 2 Final - Loss: {phase2_loss:.4f}, PPL: {phase2_ppl:.2f}")
    
    print(f"\\nTwo-Phase Training Results:")
    print(f"Phase 1 Final Loss: {phase1_loss:.4f}, PPL: {phase1_ppl:.2f}")
    print(f"Phase 2 Final Loss: {phase2_loss:.4f}, PPL: {phase2_ppl:.2f}")
    print(f"Improvement: {((phase1_loss - phase2_loss) / phase1_loss * 100):.2f}% loss reduction")
    
    return phase1_loss, phase1_ppl, phase2_loss, phase2_ppl


def main():
    print("ANA Quick Experiment Runner")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run text generation comparison
    tg_results = run_text_generation_comparison()
    
    # Run associative recall comparison
    ar_results = run_associative_recall_comparison()
    
    # Run two-phase training demonstration
    tp_results = run_two_phase_training()
    
    # Compile all results
    all_results = {
        'timestamp': timestamp,
        'text_generation': tg_results,
        'associative_recall': ar_results,
        'two_phase_training': {
            'phase1_loss': tp_results[0],
            'phase1_ppl': tp_results[1],
            'phase2_loss': tp_results[2],
            'phase2_ppl': tp_results[3],
            'improvement_pct': ((tp_results[0] - tp_results[2]) / tp_results[0] * 100)
        }
    }
    
    # Print summary
    print("\\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\\nText Generation:")
    print(f"  ANA Perplexity: {tg_results['ana_perplexity']:.2f}")
    print(f"  Baseline Perplexity: {tg_results['baseline_perplexity']:.2f}")
    print(f"  ANA Improvement: {tg_results['perplexity_improvement_pct']:.2f}%")
    
    print(f"\\nAssociative Recall:")
    print(f"  ANA Perplexity: {ar_results['ana_perplexity']:.2f}")
    print(f"  Baseline Perplexity: {ar_results['baseline_perplexity']:.2f}")
    print(f"  ANA Improvement: {ar_results['perplexity_improvement_pct']:.2f}%")
    
    print(f"\\nTwo-Phase Training:")
    print(f"  Phase 1 PPL: {tp_results[1]:.2f}")
    print(f"  Phase 2 PPL: {tp_results[3]:.2f}")
    print(f"  Improvement: {all_results['two_phase_training']['improvement_pct']:.2f}%")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/experiment_summary_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\\nResults saved to results/experiment_summary_{timestamp}.json")
    
    return all_results


if __name__ == "__main__":
    results = main()