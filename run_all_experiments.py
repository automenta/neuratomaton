#!/usr/bin/env python3
"""
ANA Project Execution Script

Runs all three solution experiments and generates comprehensive results.

Usage:
    python run_all_experiments.py [--phase {all|curriculum|hybrid|cuda}]

Phases:
    all       - Run all experiments
    curriculum- Run scale-aware curriculum experiments
    hybrid    - Run hybrid architecture experiments  
    cuda      - Run CUDA/Triton benchmark experiments
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and capture output"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\n✓ {description} completed in {elapsed:.1f}s")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {description} failed after {elapsed:.1f}s")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False, e.stdout


def run_curriculum_experiments():
    """Run scale-aware curriculum experiments"""
    print("\n" + "="*80)
    print("PHASE 1: Scale-Aware Curriculum Experiments")
    print("="*80)
    
    cmd = [sys.executable, 'experiments/scale_aware/curriculum_bench.py']
    success, output = run_command(cmd, "Scale-Aware Curriculum Training")
    
    if success:
        results_path = Path('experiments/scale_aware/curriculum_results.json')
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
    
    return None


def run_hybrid_experiments():
    """Run hybrid architecture experiments"""
    print("\n" + "="*80)
    print("PHASE 2: Hybrid ANA-Transformer Experiments")
    print("="*80)
    
    cmd = [sys.executable, 'experiments/hybrid/mixed_tasks.py']
    success, output = run_command(cmd, "Hybrid Architecture Training")
    
    if success:
        results_path = Path('experiments/hybrid/hybrid_results.json')
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
    
    return None


def run_cuda_experiments():
    """Run CUDA/Triton benchmark experiments"""
    print("\n" + "="*80)
    print("PHASE 3: CUDA/Triton Parallel Scan Benchmarks")
    print("="*80)
    
    cmd = [sys.executable, 'experiments/cuda_benchmarks/speedup.py']
    success, output = run_command(cmd, "CUDA/Triton Benchmark")
    
    if success:
        results_path = Path('experiments/cuda_benchmarks/benchmark_results.json')
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
    
    return None


def generate_summary_report(curriculum_results, hybrid_results, cuda_results):
    """Generate comprehensive summary report"""
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phases': {},
        'overall_status': 'partial'
    }
    
    # Curriculum summary
    if curriculum_results:
        report['phases']['curriculum'] = {
            'status': 'complete',
            'summary': {}
        }
        
        for name, result in curriculum_results.items():
            if isinstance(result, dict) and 'final_metrics' in result:
                metrics = result['final_metrics']
                report['phases']['curriculum']['summary'][name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'train_loss': metrics.get('train_loss', 0),
                    'val_loss': metrics.get('val_loss', 0)
                }
    
    # Hybrid summary
    if hybrid_results:
        report['phases']['hybrid'] = {
            'status': 'complete',
            'summary': {}
        }
        
        for name, history in hybrid_results.items():
            if name != 'routing' and isinstance(history, dict) and 'val_acc' in history:
                report['phases']['hybrid']['summary'][name] = {
                    'final_accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
                    'best_accuracy': max(history['val_acc']) if history['val_acc'] else 0
                }
        
        if 'routing' in hybrid_results:
            report['phases']['hybrid']['routing_analysis'] = hybrid_results['routing']
    
    # CUDA summary
    if cuda_results:
        report['phases']['cuda'] = {
            'status': 'complete',
            'summary': {}
        }
        
        if 'seq_len' in cuda_results:
            seq_results = cuda_results['seq_len']
            speedups = [r.get('speedup', 0) for r in seq_results]
            
            report['phases']['cuda']['summary'] = {
                'best_speedup': max(speedups) if speedups else 0,
                'avg_speedup': sum(speedups) / len(speedups) if speedups else 0,
                'tested_seq_lengths': len(seq_results)
            }
    
    # Overall status
    phases = report['phases']
    if all(phases.get(p, {}).get('status') == 'complete' for p in ['curriculum', 'hybrid', 'cuda']):
        report['overall_status'] = 'complete'
    
    return report


def print_summary(report):
    """Print formatted summary report"""
    
    print("\n" + "="*80)
    print("ANA PROJECT EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\nTimestamp: {report['timestamp']}")
    print(f"Overall Status: {report['overall_status'].upper()}")
    
    phases = report.get('phases', {})
    
    if 'curriculum' in phases:
        print("\n" + "-"*80)
        print("PHASE 1: Scale-Aware Curriculum")
        print("-"*80)
        
        summary = phases['curriculum'].get('summary', {})
        for model_name, metrics in summary.items():
            acc = metrics.get('accuracy', 0)
            status = "✓ TARGET MET" if acc >= 1.0 else f"⚠ {acc:.2%}"
            print(f"{model_name}: Accuracy={acc:.2%} {status}")
    
    if 'hybrid' in phases:
        print("\n" + "-"*80)
        print("PHASE 2: Hybrid Architecture")
        print("-"*80)
        
        summary = phases['hybrid'].get('summary', {})
        for model_name, metrics in summary.items():
            acc = metrics.get('final_accuracy', 0)
            best = metrics.get('best_accuracy', 0)
            print(f"{model_name}: Final={acc:.2%}, Best={best:.2%}")
        
        routing = phases['hybrid'].get('routing_analysis', {})
        if routing:
            print(f"\nRouting Analysis:")
            print(f"  ANA Route Usage: {routing.get('route_0_usage', 0):.1%}")
            print(f"  Transformer Route Usage: {routing.get('route_1_usage', 0):.1%}")
    
    if 'cuda' in phases:
        print("\n" + "-"*80)
        print("PHASE 3: CUDA/Triton Benchmarks")
        print("-"*80)
        
        summary = phases['cuda'].get('summary', {})
        best_speedup = summary.get('best_speedup', 0)
        avg_speedup = summary.get('avg_speedup', 0)
        
        print(f"Best Speedup: {best_speedup:.2f}x")
        print(f"Average Speedup: {avg_speedup:.2f}x")
        
        if best_speedup > 5:
            print("✓ GOAL ACHIEVED: >5x speedup realized!")
        elif best_speedup > 2:
            print("✓ Moderate speedup achieved")
        else:
            print("⚠ Speedup not significant")
    
    print("\n" + "="*80)
    print("GENERATED FILES")
    print("="*80)
    
    files = [
        'experiments/scale_aware/curriculum_results.json',
        'experiments/hybrid/hybrid_results.json',
        'experiments/hybrid/hybrid_analysis.png',
        'experiments/cuda_benchmarks/benchmark_results.json',
        'experiments/cuda_benchmarks/seq_len_sweep.png',
        'experiments/cuda_benchmarks/dim_sweep.png',
        'experiments/cuda_benchmarks/batch_sweep.png',
        'experiments/summary_report.json',
        'papers/ana_synergy/paper_draft.md'
    ]
    
    for f in files:
        exists = "✓" if Path(f).exists() else "✗"
        print(f"  {exists} {f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Run ANA project experiments')
    parser.add_argument('--phase', choices=['all', 'curriculum', 'hybrid', 'cuda'], 
                       default='all', help='Which phase to run')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANA PROJECT EXECUTION SCRIPT")
    print("="*80)
    print(f"Phase: {args.phase}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {Path.cwd()}")
    print("="*80)
    
    curriculum_results = None
    hybrid_results = None
    cuda_results = None
    
    if args.phase in ['all', 'curriculum']:
        curriculum_results = run_curriculum_experiments()
    
    if args.phase in ['all', 'hybrid']:
        hybrid_results = run_hybrid_experiments()
    
    if args.phase in ['all', 'cuda']:
        cuda_results = run_cuda_experiments()
    
    # Generate and save summary report
    report = generate_summary_report(curriculum_results, hybrid_results, cuda_results)
    
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print_summary(report)
    
    print("\n✓ Execution complete!")
    print(f"\nView results:")
    print(f"  Summary: experiments/summary_report.json")
    print(f"  Paper Draft: papers/ana_synergy/paper_draft.md")
    print(f"  Salvage Plan: SALVAGE_PLAN.md")


if __name__ == '__main__':
    main()
