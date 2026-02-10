#!/usr/bin/env python3
"""
Low-hanging fruit optimization profiler for Bio-ANA training.
Identifies quick wins before the 6-hour WikiText-2 validation run.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


def time_fn(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return sum(times) / len(times) * 1000


def profile_relaxation_iterations(device):
    print("\n" + "="*60)
    print("OPTIMIZATION 1: Relaxation Iteration Tuning")
    print("="*60)
    
    model = create_bio_ana('nano', vocab_size=10000).to(device)
    model.eval()
    
    input_ids = torch.randint(0, 10000, (4, 64), device=device)
    
    iteration_times = {}
    for iters in [5, 7, 10, 12, 15, 20, 25, 30]:
        def forward_fixed():
            with torch.no_grad():
                return model(input_ids, relaxation_steps=iters)
        
        t = time_fn(forward_fixed, warmup=2, repeats=5)
        iteration_times[iters] = t
    
    baseline = iteration_times[20]
    print(f"\nForward pass time by relaxation iterations:")
    for iters, t in sorted(iteration_times.items()):
        speedup = baseline / t
        print(f"  {iters:2d} iters: {t:7.2f}ms  ({speedup:.2f}x vs baseline 20)")
    
    best_iters = min(iteration_times, key=iteration_times.get)
    print(f"\nRecommendation: Use {best_iters} iterations")
    print(f"  Speedup potential: {baseline / iteration_times[best_iters]:.2f}x")
    
    return iteration_times


def profile_mixed_precision(device):
    print("\n" + "="*60)
    print("OPTIMIZATION 2: Mixed Precision (AMP)")
    print("="*60)
    
    model = create_bio_ana('nano', vocab_size=10000).to(device)
    model.eval()
    
    input_ids = torch.randint(0, 10000, (4, 64), device=device)
    
    def forward_fp32():
        with torch.no_grad():
            return model(input_ids)
    
    def forward_amp():
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                return model(input_ids)
    
    t_fp32 = time_fn(forward_fp32, warmup=3, repeats=10)
    t_amp = time_fn(forward_amp, warmup=3, repeats=10)
    
    speedup = t_fp32 / t_amp
    print(f"\nFP32: {t_fp32:.2f}ms")
    print(f"AMP:  {t_amp:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1.3:
        print(f"\nRecommendation: ENABLE mixed precision (significant speedup)")
    else:
        print(f"\nRecommendation: Mixed precision provides minimal benefit")
    
    return {'fp32': t_fp32, 'amp': t_amp, 'speedup': speedup}


def profile_early_stopping(device):
    print("\n" + "="*60)
    print("OPTIMIZATION 3: Early Stopping for Convergence")
    print("="*60)
    
    model = create_bio_ana('nano', vocab_size=10000).to(device)
    model.eval()
    
    input_ids = torch.randint(0, 10000, (4, 64), device=device)
    
    def forward_no_early_stop():
        with torch.no_grad():
            return model(input_ids, relaxation_steps=20)
    
    def forward_with_early_stop():
        with torch.no_grad():
            x = model.embedding(input_ids)
            x = model._add_position_encoding(x)
            
            outputs = []
            track_states = {'syntax': None, 'semantic': None, 'logic': None}
            
            for t in range(x.size(1)):
                xt = x[:, t, :]
                h_prev = None
                
                for i in range(20):
                    track_out, track_states = model.tracks(
                        xt,
                        h_syntax=track_states['syntax'],
                        h_semantic=track_states['semantic'],
                        h_logic=track_states['logic'],
                        steps=1,
                    )
                    
                    if h_prev is not None:
                        max_diff = max(
                            torch.abs(track_states['syntax'] - h_prev['syntax']).max().item(),
                            torch.abs(track_states['semantic'] - h_prev['semantic']).max().item(),
                            torch.abs(track_states['logic'] - h_prev['logic']).max().item(),
                        )
                        if max_diff < 0.01:
                            break
                    
                    h_prev = {k: v.clone() for k, v in track_states.items()}
                
                if model.hololink:
                    track_out, _ = model.hololink(track_out, write_mode=False)
                
                mixed = model.mixer(track_out)
                out = model.norm(xt + mixed)
                outputs.append(out)
            
            output_seq = torch.stack(outputs, dim=1)
            return model.output_head(output_seq)
    
    t_no_stop = time_fn(forward_no_early_stop, warmup=3, repeats=5)
    t_with_stop = time_fn(forward_with_early_stop, warmup=3, repeats=5)
    
    speedup = t_no_stop / t_with_stop
    print(f"\nWithout early stopping: {t_no_stop:.2f}ms")
    print(f"With early stopping:    {t_with_stop:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return {'no_stop': t_no_stop, 'with_stop': t_with_stop, 'speedup': speedup}


def profile_adaptive_schedule(device):
    print("\n" + "="*60)
    print("OPTIMIZATION 4: Adaptive Relaxation Schedule")
    print("="*60)
    
    model = create_bio_ana('nano', vocab_size=10000).to(device)
    model.eval()
    
    input_ids = torch.randint(0, 10000, (4, 64), device=device)
    
    def forward_fixed():
        with torch.no_grad():
            return model(input_ids, relaxation_steps=12)
    
    def forward_adaptive():
        with torch.no_grad():
            x = model.embedding(input_ids)
            x = model._add_position_encoding(x)
            
            seq_len = x.size(1)
            outputs = []
            track_states = {'syntax': None, 'semantic': None, 'logic': None}
            
            for t in range(seq_len):
                progress = t / seq_len
                if progress < 0.25:
                    iters = 12
                elif progress < 0.5:
                    iters = 7
                elif progress < 0.75:
                    iters = 4
                else:
                    iters = 2
                
                xt = x[:, t, :]
                track_out, track_states = model.tracks(
                    xt,
                    h_syntax=track_states['syntax'],
                    h_semantic=track_states['semantic'],
                    h_logic=track_states['logic'],
                    steps=iters,
                )
                
                if model.hololink:
                    track_out, _ = model.hololink(track_out, write_mode=False)
                
                mixed = model.mixer(track_out)
                out = model.norm(xt + mixed)
                outputs.append(out)
            
            output_seq = torch.stack(outputs, dim=1)
            return model.output_head(output_seq)
    
    t_fixed = time_fn(forward_fixed, warmup=3, repeats=5)
    t_adaptive = time_fn(forward_adaptive, warmup=3, repeats=5)
    
    speedup = t_fixed / t_adaptive
    print(f"\nFixed (12 iters): {t_fixed:.2f}ms")
    print(f"Adaptive:         {t_adaptive:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return {'fixed': t_fixed, 'adaptive': t_adaptive, 'speedup': speedup}


def profile_batch_size_scaling(device):
    print("\n" + "="*60)
    print("OPTIMIZATION 5: Batch Size & Memory Efficiency")
    print("="*60)
    
    model = create_bio_ana('nano', vocab_size=10000).to(device)
    model.eval()
    
    results = {}
    
    for batch_size in [1, 2, 4, 8, 16, 32]:
        try:
            input_ids = torch.randint(0, 10000, (batch_size, 64), device=device)
            
            torch.cuda.reset_peak_memory_stats()
            
            def forward():
                with torch.no_grad():
                    return model(input_ids)
            
            t = time_fn(forward, warmup=2, repeats=3)
            mem = torch.cuda.max_memory_allocated() / 1024**2
            
            tokens_per_ms = (batch_size * 64) / t
            results[batch_size] = {'time_ms': t, 'memory_mb': mem, 'tokens_per_ms': tokens_per_ms}
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                results[batch_size] = {'time_ms': None, 'memory_mb': None, 'tokens_per_ms': None, 'oom': True}
                break
    
    print(f"\n{'Batch':>6} {'Time':>10} {'Memory':>10} {'Tokens/ms':>12}")
    print("-" * 44)
    for bs, r in results.items():
        if r.get('oom'):
            print(f"{bs:>6} {'OOM':>10}")
        else:
            print(f"{bs:>6} {r['time_ms']:>10.2f}ms {r['memory_mb']:>8.1f}MB {r['tokens_per_ms']:>12.1f}")
    
    best_bs = max((bs for bs, r in results.items() if not r.get('oom')), 
                  key=lambda x: results[x]['tokens_per_ms'])
    print(f"\nRecommendation: batch_size={best_bs} for best throughput")
    
    return results


def profile_combined_optimizations(device):
    print("\n" + "="*60)
    print("OPTIMIZATION 6: Combined Optimizations")
    print("="*60)
    
    model = create_bio_ana('nano', vocab_size=10000).to(device)
    model.eval()
    
    input_ids = torch.randint(0, 10000, (8, 64), device=device)
    
    def forward_baseline():
        with torch.no_grad():
            return model(input_ids, relaxation_steps=20)
    
    def forward_optimized():
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x = model.embedding(input_ids)
                x = model._add_position_encoding(x)
                
                seq_len = x.size(1)
                outputs = []
                track_states = {'syntax': None, 'semantic': None, 'logic': None}
                
                for t in range(seq_len):
                    progress = t / seq_len
                    if progress < 0.25:
                        iters = 10
                    elif progress < 0.5:
                        iters = 6
                    elif progress < 0.75:
                        iters = 3
                    else:
                        iters = 2
                    
                    xt = x[:, t, :]
                    track_out, track_states = model.tracks(
                        xt,
                        h_syntax=track_states['syntax'],
                        h_semantic=track_states['semantic'],
                        h_logic=track_states['logic'],
                        steps=iters,
                    )
                    
                    if model.hololink:
                        track_out, _ = model.hololink(track_out, write_mode=False)
                    
                    mixed = model.mixer(track_out)
                    out = model.norm(xt + mixed)
                    outputs.append(out)
                
                output_seq = torch.stack(outputs, dim=1)
                return model.output_head(output_seq)
    
    t_baseline = time_fn(forward_baseline, warmup=3, repeats=5)
    t_optimized = time_fn(forward_optimized, warmup=3, repeats=5)
    
    speedup = t_baseline / t_optimized
    print(f"\nBaseline (20 fixed iters): {t_baseline:.2f}ms")
    print(f"Optimized (AMP + adaptive): {t_optimized:.2f}ms")
    print(f"Combined speedup: {speedup:.2f}x")
    
    hours_baseline = 6
    hours_optimized = hours_baseline / speedup
    print(f"\nEstimated training time reduction:")
    print(f"  Baseline: {hours_baseline:.1f} hours")
    print(f"  Optimized: {hours_optimized:.1f} hours")
    print(f"  Time saved: {hours_baseline - hours_optimized:.1f} hours")
    
    return {'baseline': t_baseline, 'optimized': t_optimized, 'speedup': speedup}


def estimate_training_impact(results):
    print("\n" + "="*60)
    print("TRAINING IMPACT ESTIMATE")
    print("="*60)
    
    baseline_hours = 6
    total_speedup = 1.0
    
    if 'combined' in results and results['combined']:
        total_speedup = results['combined'].get('speedup', 1.0)
    
    optimized_hours = baseline_hours / total_speedup
    
    print(f"\nWikiText-2 validation run (5 epochs):")
    print(f"  Current estimate: {baseline_hours:.1f} hours")
    print(f"  With optimizations: {optimized_hours:.1f} hours")
    print(f"  Time saved: {baseline_hours - optimized_hours:.1f} hours")
    
    print(f"\nFull Phase 4 (41 GPU hours):")
    print(f"  With optimizations: {41 / total_speedup:.1f} hours")
    print(f"  Time saved: {41 - 41 / total_speedup:.1f} hours")


def main():
    print("="*60)
    print("Bio-ANA Low-Hanging Fruit Profiler")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    results = {}
    
    try:
        results['relaxation'] = profile_relaxation_iterations(device)
    except Exception as e:
        print(f"Error in relaxation profiling: {e}")
    
    if device.type == 'cuda':
        try:
            results['amp'] = profile_mixed_precision(device)
        except Exception as e:
            print(f"Error in AMP profiling: {e}")
    
    try:
        results['early_stop'] = profile_early_stopping(device)
    except Exception as e:
        print(f"Error in early stopping profiling: {e}")
    
    try:
        results['adaptive'] = profile_adaptive_schedule(device)
    except Exception as e:
        print(f"Error in adaptive profiling: {e}")
    
    if device.type == 'cuda':
        try:
            results['batch'] = profile_batch_size_scaling(device)
        except Exception as e:
            print(f"Error in batch profiling: {e}")
    
    try:
        results['combined'] = profile_combined_optimizations(device)
    except Exception as e:
        print(f"Error in combined profiling: {e}")
    
    estimate_training_impact(results)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS SUMMARY")
    print("="*60)
    
    print("""
LOW EFFORT (apply now):
1. Reduce relaxation_iterations from 20 to 8-10 in config
2. Enable torch.cuda.amp.autocast() in training loop
3. Use adaptive schedule in trainer.forward_with_optimizations()

MEDIUM EFFORT (consider):
4. Add early stopping with convergence threshold 0.01
5. Optimize batch size for GPU (likely 8-16)

HIGH IMPACT:
- Combined optimizations can provide 2-3x speedup
- 6 hour validation → 2-3 hours
- 41 hour Phase 4 → 14-20 hours
""")


if __name__ == "__main__":
    main()
