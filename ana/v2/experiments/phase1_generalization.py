#!/usr/bin/env python3
"""
ANA v2: Generalization Experiment

Tests if ANA can learn algorithms from examples and generalize to longer sequences.

This is the core experiment that proves the thesis.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from pathlib import Path
from datetime import datetime

from ana.v2.core import ANAConfig, ANAModel
from ana.v2.train import Trainer, SimpleDataset
from ana.v2.tasks import (
    generate_copy_task,
    generate_reverse_task,
    generate_associative_recall_task
)


def train_and_evaluate(task, task_name, epochs=50):
    """Train on task and evaluate generalization."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    
    # Create model
    config = ANAConfig(
        d_model=64,
        vocab_size=task.vocab_size,
        track_dims=(16, 32, 16),
        stack_depth=4,
        stack_dim=32,
        num_layers=2
    )
    
    trainer = Trainer(config, lr=1e-3, output_dir=f"ana/v2/results/{task_name}")
    
    # Train
    dataset = SimpleDataset(task.train_seqs, task.train_targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Training {task_name} for {epochs} epochs...")
    history = trainer.train(loader, num_epochs=epochs)
    
    # Evaluate on different length ratios
    results = {
        'task': task_name,
        'train_len_range': str(task.__dict__.get('_train_len_range', 'unknown')),
        'test_len_range': str(task.__dict__.get('_test_len_range', 'unknown')),
        'epochs': epochs,
        'final_train_loss': history['train_loss'][-1],
        'generalization': {}
    }
    
    # Evaluate with increasing lengths
    trainer.model.eval()
    
    # Get test lengths
    test_lengths = []
    for seq in task.test_seqs:
        non_zero = (seq != 0).sum().item()
        if non_zero > 0:
            test_lengths.append(non_zero)
    
    min_test_len = min(test_lengths)
    max_test_len = max(test_lengths)
    
    # Calculate length ratios
    ratios = sorted(set([round(l / min_test_len, 1) for l in test_lengths]))
    
    for ratio in ratios:
        target_len = int(min_test_len * ratio)
        
        # Filter sequences of this length
        filtered_seqs = []
        filtered_targets = []
        for i, seq in enumerate(task.test_seqs):
            seq_len = (seq != 0).sum().item()
            if seq_len == target_len:
                filtered_seqs.append(seq)
                filtered_targets.append(task.test_targets[i])
        
        if len(filtered_seqs) > 0:
            filtered_seqs = torch.stack(filtered_seqs)
            filtered_targets = torch.stack(filtered_targets)
            
            with torch.no_grad():
                filtered_seqs = filtered_seqs.to(trainer.device)
                logits = trainer.model(filtered_seqs)
                preds = logits.argmax(dim=-1).cpu()
                
                # Exact match accuracy
                exact_matches = 0
                for i in range(len(filtered_seqs)):
                    if torch.equal(preds[i], filtered_targets[i]):
                        exact_matches += 1
                
                # Token accuracy
                correct_tokens = 0
                total_tokens = 0
                for i in range(len(filtered_seqs)):
                    mask = filtered_targets[i] != 0
                    correct_tokens += ((preds[i] == filtered_targets[i]) & mask).sum().item()
                    total_tokens += mask.sum().item()
                
                results['generalization'][f"ratio_{ratio:.1f}"] = {
                    'length': target_len,
                    'exact_accuracy': exact_matches / len(filtered_seqs),
                    'token_accuracy': correct_tokens / total_tokens if total_tokens > 0 else 0,
                    'num_samples': len(filtered_seqs)
                }
    
    return results, trainer


def main():
    """Run generalization experiments for all Phase 1 tasks."""
    print("=" * 60)
    print("ANA v2: Phase 1 Generalization Experiments")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("ana/v2/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Task 1: Copy
    copy_task = generate_copy_task(
        num_train=1000,
        num_test=200,
        train_len=(5, 10),
        test_len=(11, 30),
        vocab_size=10
    )
    copy_task._train_len_range = (5, 10)
    copy_task._test_len_range = (11, 30)
    
    copy_results, _ = train_and_evaluate(copy_task, "copy", epochs=50)
    all_results['copy'] = copy_results
    
    # Task 2: Reverse
    reverse_task = generate_reverse_task(
        num_train=1000,
        num_test=200,
        train_len=(3, 7),
        test_len=(8, 20),
        vocab_size=10
    )
    reverse_task._train_len_range = (3, 7)
    reverse_task._test_len_range = (8, 20)
    
    reverse_results, _ = train_and_evaluate(reverse_task, "reverse", epochs=50)
    all_results['reverse'] = reverse_results
    
    # Task 3: Associative Recall
    ar_task = generate_associative_recall_task(
        num_train=1000,
        num_test=200,
        train_pairs=(2, 4),
        test_pairs=(5, 10),
        vocab_size=20
    )
    ar_task._train_len_range = (2, 4)  # pairs
    ar_task._test_len_range = (5, 10)  # pairs
    
    ar_results, _ = train_and_evaluate(ar_task, "associative_recall", epochs=50)
    all_results['associative_recall'] = ar_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"phase1_generalization_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    # Print summary
    for task_name, results in all_results.items():
        print(f"\n{task_name.upper()}:")
        print(f"  Train loss: {results['final_train_loss']:.4f}")
        
        for ratio_key, metrics in results['generalization'].items():
            ratio = float(ratio_key.split('_')[1])
            exact = metrics['exact_accuracy']
            token = metrics['token_accuracy']
            length = metrics['length']
            print(f"  Ratio {ratio:.1f} (len={length}): Exact={exact:.2%}, Token={token:.2%}")
    
    # Check success criteria
    print(f"\n{'='*60}")
    print("Success Criteria Check")
    print(f"{'='*60}")
    
    success = True
    
    # Copy: >95% at ratio 2.0+
    copy_ratios = [float(k.split('_')[1]) for k in all_results['copy']['generalization'].keys()]
    copy_max_ratio = max(copy_ratios)
    copy_exact = all_results['copy']['generalization'][f'ratio_{copy_max_ratio:.1f}']['exact_accuracy']
    copy_pass = copy_exact > 0.95
    print(f"Copy (>95% at ratio {copy_max_ratio:.1f}): {'PASS' if copy_pass else 'FAIL'} ({copy_exact:.2%})")
    success = success and copy_pass
    
    # Reverse: >90% at ratio 2.0+
    reverse_ratios = [float(k.split('_')[1]) for k in all_results['reverse']['generalization'].keys()]
    reverse_max_ratio = max(reverse_ratios)
    reverse_exact = all_results['reverse']['generalization'][f'ratio_{reverse_max_ratio:.1f}']['exact_accuracy']
    reverse_pass = reverse_exact > 0.90
    print(f"Reverse (>90% at ratio {reverse_max_ratio:.1f}): {'PASS' if reverse_pass else 'FAIL'} ({reverse_exact:.2%})")
    success = success and reverse_pass
    
    # Associative Recall: >85% at ratio 2.0+
    ar_ratios = [float(k.split('_')[1]) for k in all_results['associative_recall']['generalization'].keys()]
    ar_max_ratio = max(ar_ratios)
    ar_exact = all_results['associative_recall']['generalization'][f'ratio_{ar_max_ratio:.1f}']['exact_accuracy']
    ar_pass = ar_exact > 0.85
    print(f"Associative Recall (>85% at ratio {ar_max_ratio:.1f}): {'PASS' if ar_pass else 'FAIL'} ({ar_exact:.2%})")
    success = success and ar_pass
    
    print(f"\n{'='*60}")
    if success:
        print("PHASE 1 COMPLETE: All success criteria met! 🎉")
    else:
        print("PHASE 1 INCOMPLETE: Some criteria not met. Need architecture improvements.")
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    main()
