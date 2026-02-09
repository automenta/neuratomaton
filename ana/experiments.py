import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ana.config import ANAConfig, TrainingConfig, DataConfig
from ana.models import ANAModel, BaselineSSM
from ana.train import run_training, evaluate, col_fn
from ana.data import AssociativeRecallDataset
from ana.benchmarks import BenchmarkEvaluator
import time

SCALING_CONFIGS = {
    'tiny': {
        'd_model': 32,
        'state_dim': 32,
        'num_layers': 1,
        'track_count': 2,
        'key_dim': 32,
        'epochs': 10,
        'batch_size': 32,
    },
    'small': {
        'd_model': 64,
        'state_dim': 64,
        'num_layers': 2,
        'track_count': 2,
        'key_dim': 64,
        'epochs': 15,
        'batch_size': 16,
    },
    'medium': {
        'd_model': 128,
        'state_dim': 128,
        'num_layers': 3,
        'track_count': 2,
        'key_dim': 128,
        'epochs': 20,
        'batch_size': 8,
    },
    'large': {
        'd_model': 256,
        'state_dim': 256,
        'num_layers': 4,
        'track_count': 3,
        'key_dim': 256,
        'epochs': 25,
        'batch_size': 4,
    },
}

ABLATION_CONFIGS = {
    'full': {
        'use_hololink': True,
        'use_controller': True,
        'max_thinking_steps': 0,
    },
    'no_hololink': {
        'use_hololink': False,
        'use_controller': True,
        'max_thinking_steps': 0,
    },
    'no_controller': {
        'use_hololink': True,
        'use_controller': False,
        'max_thinking_steps': 0,
    },
    'static_only': {
        'use_hololink': False,
        'use_controller': False,
        'max_thinking_steps': 0,
    },
    'with_thinking': {
        'use_hololink': True,
        'use_controller': True,
        'max_thinking_steps': 2,
    },
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_scaling_study(base_config, scale='small', output_dir='archive/scaling'):
    os.makedirs(output_dir, exist_ok=True)
    
    scale_config = SCALING_CONFIGS[scale]
    
    ana_config = ANAConfig(
        d_model=scale_config['d_model'],
        state_dim=scale_config['state_dim'],
        num_layers=scale_config['num_layers'],
        track_count=scale_config['track_count'],
        key_dim=scale_config['key_dim'],
        vocab_size=base_config.get('vocab_size', 50),
        use_parallel_scan=True,
    )
    
    train_config = TrainingConfig(
        batch_size=scale_config['batch_size'],
        epochs=scale_config['epochs'],
        stage='2a',
        save_checkpoints=False,
    )
    
    data_config = DataConfig(
        vocab_size=base_config.get('vocab_size', 50),
        min_noise=base_config.get('min_noise', 10),
        max_noise=base_config.get('max_noise', 50),
        dataset_size=base_config.get('dataset_size', 2000),
    )
    
    print(f"\n{'='*60}")
    print(f"Scaling Study: {scale.upper()}")
    print(f"{'='*60}")
    
    model = ANAModel(ana_config)
    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    history = run_training(ana_config, train_config, data_config, model_type='ana')
    train_time = time.time() - start_time
    
    final_loss = history['loss'][-1]
    final_acc = history['needle_acc'][-1]
    
    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Final Needle Accuracy: {final_acc:.2%}")
    print(f"Training Time: {train_time:.1f}s")
    
    results = {
        'scale': scale,
        'params': params,
        'final_loss': final_loss,
        'final_acc': final_acc,
        'train_time': train_time,
        'config': {
            'd_model': scale_config['d_model'],
            'state_dim': scale_config['state_dim'],
            'num_layers': scale_config['num_layers'],
            'track_count': scale_config['track_count'],
        }
    }
    
    results_path = os.path.join(output_dir, f'scaling_{scale}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_ablation_study(base_config, ablation='full', output_dir='archive/ablations'):
    os.makedirs(output_dir, exist_ok=True)
    
    ablation_config = ABLATION_CONFIGS[ablation]
    
    ana_config = ANAConfig(
        d_model=base_config.get('d_model', 64),
        state_dim=base_config.get('state_dim', 64),
        num_layers=base_config.get('num_layers', 2),
        track_count=base_config.get('track_count', 2),
        vocab_size=base_config.get('vocab_size', 50),
        use_parallel_scan=True,
        **ablation_config
    )
    
    train_config = TrainingConfig(
        batch_size=base_config.get('batch_size', 16),
        epochs=base_config.get('epochs', 10),
        stage='2a',
        save_checkpoints=False,
    )
    
    data_config = DataConfig(
        vocab_size=base_config.get('vocab_size', 50),
        min_noise=10,
        max_noise=50,
        dataset_size=2000,
    )
    
    print(f"\n{'='*60}")
    print(f"Ablation Study: {ablation}")
    print(f"{'='*60}")
    print(f"Config: {ablation_config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    history = run_training(ana_config, train_config, data_config, model_type='ana')
    train_time = time.time() - start_time
    
    model = ANAModel(ana_config).to(device)
    
    evaluator = BenchmarkEvaluator(model, device, ana_config.vocab_size)
    benchmark_results = evaluator.run_all_benchmarks()
    
    results = {
        'ablation': ablation,
        'config': ablation_config,
        'final_loss': history['loss'][-1],
        'final_acc': history['needle_acc'][-1],
        'train_time': train_time,
        'benchmarks': benchmark_results,
    }
    
    results_path = os.path.join(output_dir, f'ablation_{ablation}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_full_study(output_dir='archive/full_study'):
    os.makedirs(output_dir, exist_ok=True)
    
    base_config = {
        'vocab_size': 50,
        'min_noise': 10,
        'max_noise': 50,
        'dataset_size': 2000,
        'd_model': 64,
        'state_dim': 64,
        'num_layers': 2,
        'track_count': 2,
        'batch_size': 16,
        'epochs': 10,
    }
    
    all_results = {
        'scaling': {},
        'ablations': {},
    }
    
    print("\n" + "="*70)
    print("ANA FULL RESEARCH STUDY")
    print("="*70)
    
    for ablation_name in ABLATION_CONFIGS:
        try:
            results = run_ablation_study(base_config, ablation_name, output_dir)
            all_results['ablations'][ablation_name] = results
        except Exception as e:
            print(f"Error in ablation {ablation_name}: {e}")
            all_results['ablations'][ablation_name] = {'error': str(e)}
    
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"{'Ablation':<20} {'Loss':>10} {'Acc':>10} {'MQAR_16':>10}")
    print("-"*50)
    
    for name, results in all_results['ablations'].items():
        if 'error' not in results:
            loss = results['final_loss']
            acc = results['final_acc']
            mqar = results['benchmarks'].get('MQAR_16_pairs', 0)
            print(f"{name:<20} {loss:>10.4f} {acc:>10.2%} {mqar:>10.2%}")
    
    summary_path = os.path.join(output_dir, 'full_study_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nFull study saved to {summary_path}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ANA Research Experiments")
    parser.add_argument("--study", choices=['scaling', 'ablation', 'full'], default='ablation')
    parser.add_argument("--scale", choices=['tiny', 'small', 'medium', 'large'], default='small')
    parser.add_argument("--ablation", choices=list(ABLATION_CONFIGS.keys()), default='full')
    parser.add_argument("--output", type=str, default="archive/research")
    
    args = parser.parse_args()
    
    base_config = {
        'vocab_size': 50,
        'min_noise': 10,
        'max_noise': 50,
        'dataset_size': 2000,
    }
    
    if args.study == 'scaling':
        run_scaling_study(base_config, args.scale, args.output)
    elif args.study == 'ablation':
        run_ablation_study(base_config, args.ablation, args.output)
    else:
        run_full_study(args.output)
