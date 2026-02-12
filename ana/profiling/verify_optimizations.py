import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ana import ANAConfig, ANAModel


def profile_model(config, seq_len=512, batch_size=16, warmup=3, steps=20, use_amp=False, use_compile=False):
    """Profile model performance and identify bottlenecks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ANAModel(config).to(device)
    model.eval()
    
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile failed: {e}")
    
    for _ in range(warmup):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        with torch.no_grad():
            if use_amp and device == "cuda":
                with torch.amp.autocast('cuda'):
                    _ = model(x)
            else:
                _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(steps):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        with torch.no_grad():
            if use_amp and device == "cuda":
                with torch.amp.autocast('cuda'):
                    _ = model(x)
            else:
                _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_sec = (batch_size * seq_len * steps) / elapsed
    
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"Latency: {elapsed/steps*1000:.2f} ms/batch")
    
    return {'tokens_per_sec': tokens_per_sec, 'latency_ms': elapsed/steps*1000}


if __name__ == "__main__":
    print("="*60)
    print("OPTIMIZATION COMPARISON")
    print("="*60)
    
    config = ANAConfig(d_model=64, vocab_size=100, state_dim=64)
    seq_len = 512
    batch_size = 16
    
    print("\n[1] Baseline (no optimizations)")
    baseline = profile_model(config, seq_len=seq_len, batch_size=batch_size)
    
    config_ps = ANAConfig(d_model=64, vocab_size=100, state_dim=64, use_parallel_scan=True)
    print("\n[2] Parallel Scan + AMP")
    with_both = profile_model(config_ps, seq_len=seq_len, batch_size=batch_size, use_amp=True)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline:           {baseline['tokens_per_sec']:,.0f} tokens/sec")
    print(f"PS + AMP:           {with_both['tokens_per_sec']:,.0f} tokens/sec ({with_both['tokens_per_sec']/baseline['tokens_per_sec']:.2f}x)")
    
    speedup = with_both['tokens_per_sec'] / baseline['tokens_per_sec']
    if speedup > 1.5:
        print(f"\n✅ Optimization successful: {speedup:.2f}x speedup")
    else:
        print(f"\n⚠️ Limited improvement: {speedup:.2f}x speedup")
