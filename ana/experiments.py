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
        'epochs': 3,
        'batch_size': 32,
    },
    'small': {
        'd_model': 64,
        'state_dim': 64,
        'num_layers': 2,
        'track_count': 2,
        'key_dim': 64,
        'epochs': 3,
        'batch_size': 16,
    },
    'medium': {
        'd_model': 128,
        'state_dim': 128,
        'num_layers': 3,
        'track_count': 2,
        'key_dim': 128,
        'epochs': 3,
        'batch_size': 8,
    },
    'large': {
        'd_model': 256,
        'state_dim': 256,
        'num_layers': 4,
        'track_count': 3,
        'key_dim': 256,
        'epochs': 3,
        'batch_size': 4,
    },
    'xlarge': {
        'd_model': 512,
        'state_dim': 512,
        'num_layers': 8,
        'track_count': 4,
        'key_dim': 384,
        'epochs': 5,
        'batch_size': 2,
    },
    'xxlarge': {
        'd_model': 768,
        'state_dim': 768,
        'num_layers': 12,
        'track_count': 4,
        'key_dim': 512,
        'epochs': 5,
        'batch_size': 1,
    },
    '125M': {
        'd_model': 768,
        'state_dim': 768,
        'num_layers': 14,
        'track_count': 4,
        'key_dim': 512,
        'epochs': 5,
        'batch_size': 1,
    },
}

BASELINE_SCALING_CONFIGS = {
    'tiny': {
        'd_model': 32,
        'state_dim': 32,
        'num_layers': 1,
        'epochs': 3,
        'batch_size': 32,
    },
    'small': {
        'd_model': 64,
        'state_dim': 64,
        'num_layers': 2,
        'epochs': 3,
        'batch_size': 16,
    },
    'medium': {
        'd_model': 128,
        'state_dim': 128,
        'num_layers': 3,
        'epochs': 3,
        'batch_size': 8,
    },
    'large': {
        'd_model': 256,
        'state_dim': 256,
        'num_layers': 4,
        'epochs': 3,
        'batch_size': 4,
    },
    'xlarge': {
        'd_model': 512,
        'state_dim': 512,
        'num_layers': 8,
        'epochs': 5,
        'batch_size': 2,
    },
    'xxlarge': {
        'd_model': 1024,
        'state_dim': 1024,
        'num_layers': 12,
        'epochs': 5,
        'batch_size': 1,
    },
    '125M': {
        'd_model': 2048,
        'state_dim': 2048,
        'num_layers': 16,
        'epochs': 5,
        'batch_size': 1,
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
        dataset_size=base_config.get('dataset_size', 100),
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

def run_baseline_scaling_study(base_config, scale='small', output_dir='archive/baseline_scaling'):
    os.makedirs(output_dir, exist_ok=True)
    
    scale_config = BASELINE_SCALING_CONFIGS[scale]
    
    ana_config = ANAConfig(
        d_model=scale_config['d_model'],
        state_dim=scale_config['state_dim'],
        num_layers=scale_config['num_layers'],
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
    print(f"BASELINE Scaling Study: {scale.upper()}")
    print(f"{'='*60}")
    
    model = BaselineSSM(ana_config)
    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    history = run_training(ana_config, train_config, data_config, model_type='baseline')
    train_time = time.time() - start_time
    
    final_loss = history['loss'][-1]
    final_acc = history['needle_acc'][-1]
    
    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Final Needle Accuracy: {final_acc:.2%}")
    print(f"Training Time: {train_time:.1f}s")
    
    results = {
        'scale': scale,
        'model_type': 'baseline',
        'params': params,
        'final_loss': final_loss,
        'final_acc': final_acc,
        'train_time': train_time,
        'config': {
            'd_model': scale_config['d_model'],
            'state_dim': scale_config['state_dim'],
            'num_layers': scale_config['num_layers'],
        }
    }
    
    results_path = os.path.join(output_dir, f'baseline_scaling_{scale}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_comparison_study(scales=['small', 'medium', 'large', 'xlarge'], output_dir='archive/comparison'):
    os.makedirs(output_dir, exist_ok=True)
    
    base_config = {
        'vocab_size': 50,
        'min_noise': 10,
        'max_noise': 50,
        'dataset_size': 100,
    }
    
    all_results = {
        'ana': {},
        'baseline': {},
    }
    
    print("\n" + "="*70)
    print("ANA vs BASELINE COMPARISON STUDY")
    print("="*70)
    
    for scale in scales:
        print(f"\n--- Testing scale: {scale} ---")
        
        ana_results = run_scaling_study(base_config, scale, output_dir)
        baseline_results = run_baseline_scaling_study(base_config, scale, output_dir)
        
        all_results['ana'][scale] = ana_results
        all_results['baseline'][scale] = baseline_results
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Scale':<12} {'ANA Params':>12} {'Baseline Params':>15} {'ANA Acc':>10} {'Baseline Acc':>12} {'Diff':>8}")
    print("-"*70)
    
    for scale in scales:
        ana = all_results['ana'][scale]
        base = all_results['baseline'][scale]
        ana_acc = ana['final_acc']
        base_acc = base['final_acc']
        diff = ana_acc - base_acc
        print(f"{scale:<12} {ana['params']:>12,} {base['params']:>15,} {ana_acc:>10.2%} {base_acc:>12.2%} {diff:>8.2%}")
    
    summary_path = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComparison saved to {summary_path}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ANA Research Experiments")
    parser.add_argument("--study", choices=['scaling', 'ablation', 'full', 'baseline_scaling', 'comparison'], default='ablation')
    parser.add_argument("--scale", choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', '125M'], default='small')
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
    elif args.study == 'baseline_scaling':
        run_baseline_scaling_study(base_config, args.scale, args.output)
    elif args.study == 'comparison':
        run_comparison_study(['small', 'medium', 'large'], args.output)
    elif args.study == 'ablation':
        run_ablation_study(base_config, args.ablation, args.output)
    else:
        run_full_study(args.output)
