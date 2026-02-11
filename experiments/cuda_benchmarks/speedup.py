"""
CUDA/Triton Parallel Scan Benchmark

Measures the speedup achieved by Triton kernels over Python implementation
for parallel scan operations at various sequence lengths.

Key Experiments:
1. Hillis-Steele scan benchmark
2. Memory bandwidth analysis
3. Scaling analysis across sequence lengths
4. Comparison to PyTorch baseline
"""

import torch
import time
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ana.kernels import TritonParallelScan, PyTorchParallelScan, parallel_scan


class BenchmarkSuite:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.triton_scanner = TritonParallelScan()
        self.pytorch_scanner = PyTorchParallelScan()
        
    def warmup(self):
        """Warm up GPU and kernels"""
        for _ in range(10):
            u = torch.randn(32, 128, 512, device=self.device)
            a = torch.ones(32, 128, 512, device=self.device)
            b = torch.ones(32, 128, 512, device=self.device)
            _ = self.triton_scanner.parallel_scan(u, a, b)
            _ = self.pytorch_scanner.forward(u, a, b)
        torch.cuda.synchronize()
    
    def benchmark_forward(self, seq_len, dim=512, batch_size=32, num_iterations=100):
        """Benchmark forward pass at given sequence length"""
        
        u = torch.randn(batch_size, seq_len, dim, device=self.device)
        a = torch.ones(batch_size, seq_len, dim, device=self.device)
        b = torch.ones(batch_size, seq_len, dim, device=self.device)
        h_init = torch.zeros(batch_size, dim, device=self.device)
        
        # Triton benchmark
        start = time.time()
        for _ in range(num_iterations):
            _ = self.triton_scanner.parallel_scan(u, a, b, h_init)
            torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iterations
        
        # PyTorch benchmark
        start = time.time()
        for _ in range(num_iterations):
            _ = self.pytorch_scanner.forward(u, a, b, h_init)
            torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / num_iterations
        
        # Memory usage
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            _ = self.triton_scanner.parallel_scan(u, a, b, h_init)
            triton_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            torch.cuda.reset_peak_memory_stats()
            _ = self.pytorch_scanner.forward(u, a, b, h_init)
            pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            triton_memory = 0
            pytorch_memory = 0
        
        # Verify correctness
        triton_output = self.triton_scanner.parallel_scan(u, a, b, h_init)
        pytorch_output = self.pytorch_scanner.forward(u, a, b, h_init)
        
        max_diff = torch.max(torch.abs(triton_output - pytorch_output)).item()
        
        speedup = pytorch_time / triton_time if triton_time > 0 else 1.0
        
        return {
            'seq_len': seq_len,
            'triton_time_ms': triton_time * 1000,
            'pytorch_time_ms': pytorch_time * 1000,
            'speedup': speedup,
            'triton_memory_mb': triton_memory,
            'pytorch_memory_mb': pytorch_memory,
            'max_diff': max_diff,
            'correct': max_diff < 1e-3
        }
    
    def run_seq_len_sweep(self, seq_lengths=[64, 128, 256, 512, 1024, 2048, 4096, 8192], dim=512, batch_size=32):
        """Run benchmark across sequence lengths"""
        results = []
        
        print(f"{'Seq Len':<10} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10} {'Memory':<15} {'Correct':<8}")
        print("-" * 80)
        
        for seq_len in seq_lengths:
            result = self.benchmark_forward(seq_len, dim, batch_size)
            results.append(result)
            
            print(f"{result['seq_len']:<10} "
                  f"{result['triton_time_ms']:<15.4f} "
                  f"{result['pytorch_time_ms']:<15.4f} "
                  f"{result['speedup']:<10.2f}x "
                  f"{result['triton_memory_mb']:<15.1f} "
                  f"{'✓' if result['correct'] else '✗':<8}")
        
        return results
    
    def run_dim_sweep(self, dims=[128, 256, 512, 768, 1024], seq_len=512, batch_size=32):
        """Run benchmark across dimensions"""
        results = []
        
        print(f"\n{'Dim':<10} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10} {'Correct':<8}")
        print("-" * 70)
        
        for dim in dims:
            result = self.benchmark_forward(seq_len, dim, batch_size)
            results.append(result)
            
            print(f"{dim:<10} "
                  f"{result['triton_time_ms']:<15.4f} "
                  f"{result['pytorch_time_ms']:<15.4f} "
                  f"{result['speedup']:<10.2f}x "
                  f"{'✓' if result['correct'] else '✗':<8}")
        
        return results
    
    def run_batch_sweep(self, batch_sizes=[1, 8, 16, 32, 64, 128], seq_len=512, dim=512):
        """Run benchmark across batch sizes"""
        results = []
        
        print(f"\n{'Batch':<10} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10} {'Correct':<8}")
        print("-" * 70)
        
        for batch_size in batch_sizes:
            result = self.benchmark_forward(seq_len, dim, batch_size)
            results.append(result)
            
            print(f"{batch_size:<10} "
                  f"{result['triton_time_ms']:<15.4f} "
                  f"{result['pytorch_time_ms']:<15.4f} "
                  f"{result['speedup']:<10.2f}x "
                  f"{'✓' if result['correct'] else '✗':<8}")
        
        return results
    
    def plot_results(self, results, sweep_type='seq_len'):
        """Plot benchmark results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if sweep_type == 'seq_len':
            x = [r['seq_len'] for r in results]
            x_label = 'Sequence Length'
            x_log = True
        elif sweep_type == 'dim':
            x = [r['dim'] for r in results] if 'dim' in results[0] else [r['seq_len'] for r in results]
            x_label = 'Dimension'
            x_log = False
        else:
            x = [r['batch_size'] for r in results] if 'batch_size' in results[0] else list(range(len(results)))
            x_label = 'Batch Size'
            x_log = False
        
        # Time comparison
        axes[0, 0].plot(x, [r['triton_time_ms'] for r in results], 'o-', label='Triton')
        axes[0, 0].plot(x, [r['pytorch_time_ms'] for r in results], 's-', label='PyTorch')
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].set_title('Forward Pass Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        if x_log:
            axes[0, 0].set_xscale('log')
        
        # Speedup
        axes[0, 1].plot(x, [r['speedup'] for r in results], 'o-', color='green')
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel(x_label)
        axes[0, 1].set_ylabel('Speedup (x)')
        axes[0, 1].set_title('Speedup over PyTorch')
        axes[0, 1].grid(True)
        if x_log:
            axes[0, 1].set_xscale('log')
        
        # Memory
        axes[1, 0].plot(x, [r['triton_memory_mb'] for r in results], 'o-', label='Triton')
        axes[1, 0].plot(x, [r['pytorch_memory_mb'] for r in results], 's-', label='PyTorch')
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].set_title('Peak Memory Usage')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        if x_log:
            axes[1, 0].set_xscale('log')
        
        # Throughput
        if sweep_type == 'seq_len':
            throughput_triton = [x[i] / (r['triton_time_ms'] / 1000) for i, r in enumerate(results)]
            throughput_pytorch = [x[i] / (r['pytorch_time_ms'] / 1000) for i, r in enumerate(results)]
            axes[1, 1].plot(x, throughput_triton, 'o-', label='Triton')
            axes[1, 1].plot(x, throughput_pytorch, 's-', label='PyTorch')
            axes[1, 1].set_xlabel(x_label)
            axes[1, 1].set_ylabel('Throughput (tokens/sec)')
            axes[1, 1].set_title('Processing Throughput')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            if x_log:
                axes[1, 1].set_xscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'Throughput not applicable', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        return fig


def main():
    print("=" * 80)
    print("Triton Parallel Scan Benchmark")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("WARNING: CUDA not available. Running on CPU (no speedup expected).")
    
    suite = BenchmarkSuite(device=device)
    
    print("\nWarming up...")
    suite.warmup()
    
    output_dir = Path(__file__).parent.parent / 'experiments' / 'cuda_benchmarks'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Sequence length sweep
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Sequence Length Sweep")
    print("=" * 80)
    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    seq_results = suite.run_seq_len_sweep(seq_lengths=seq_lengths)
    all_results['seq_len'] = seq_results
    
    fig = suite.plot_results(seq_results, 'seq_len')
    fig.savefig(output_dir / 'seq_len_sweep.png', dpi=150)
    print(f"\n✓ Plot saved: {output_dir / 'seq_len_sweep.png'}")
    
    # 2. Dimension sweep
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Dimension Sweep")
    print("=" * 80)
    dims = [128, 256, 512, 768]
    dim_results = suite.run_dim_sweep(dims=dims)
    all_results['dim'] = dim_results
    
    fig = suite.plot_results(dim_results, 'dim')
    fig.savefig(output_dir / 'dim_sweep.png', dpi=150)
    print(f"\n✓ Plot saved: {output_dir / 'dim_sweep.png'}")
    
    # 3. Batch size sweep
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Batch Size Sweep")
    print("=" * 80)
    batch_sizes = [1, 8, 16, 32, 64]
    batch_results = suite.run_batch_sweep(batch_sizes=batch_sizes)
    all_results['batch'] = batch_results
    
    fig = suite.plot_results(batch_results, 'batch')
    fig.savefig(output_dir / 'batch_sweep.png', dpi=150)
    print(f"\n✓ Plot saved: {output_dir / 'batch_sweep.png'}")
    
    # Save results
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_dir / 'benchmark_results.json'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    best_speedup = max(r['speedup'] for r in seq_results)
    best_seq_len = seq_results[seq_results.index(max(seq_results, key=lambda r: r['speedup']))]['seq_len']
    
    print(f"\nBest Speedup: {best_speedup:.2f}x at seq_len={best_seq_len}")
    print(f"Average Speedup: {np.mean([r['speedup'] for r in seq_results]):.2f}x")
    
    if best_speedup > 5:
        print("\n✓ GOAL ACHIEVED: >5x speedup realized!")
    elif best_speedup > 2:
        print("\n✓ Moderate speedup achieved. Further optimization possible.")
    else:
        print("\n⚠ Speedup not significant. May need kernel optimization.")
    
    print("\n✓ Benchmark complete!")
    return all_results


if __name__ == '__main__':
    results = main()
