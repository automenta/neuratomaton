"""
E3: Memory Efficiency Profiling

Validate O(1) memory claim for HoloLink.
"""
import torch
import gc
from ana import ANAConfig, ANAModel


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def profile_memory(model, seq_len, batch_size=1, device='cuda'):
    """Profile memory usage for a given sequence length."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)
        _ = model(x)
    
    return get_memory_mb()


def run_memory_profile():
    """Profile memory across different sequence lengths."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping memory profile")
        return None
    
    print("="*60)
    print("E3: MEMORY EFFICIENCY PROFILING")
    print("="*60)
    
    config = ANAConfig(
        d_model=64, vocab_size=100, state_dim=64,
        track_count=1, num_layers=1,
        use_hololink=True, use_controller=False,
        use_parallel_scan=True
    )
    
    model = ANAModel(config).to(device)
    
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    results = {}
    
    print("\nSequence Length | Memory (MB) | Tokens/MB")
    print("-" * 45)
    
    for L in seq_lengths:
        mem = profile_memory(model, L, batch_size=1, device=device)
        tokens_per_mb = L / mem if mem > 0 else 0
        results[L] = mem
        print(f"     {L:5d}       |   {mem:7.1f}   |   {tokens_per_mb:6.0f}")
    
    # Check if memory is O(1) (constant) or O(n) (linear)
    mem_512 = results[512]
    mem_16384 = results.get(16384, results[max(results.keys())])
    
    growth_ratio = mem_16384 / mem_512
    expected_linear = 16384 / 512  # 32x
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print(f"Memory at 512: {mem_512:.1f} MB")
    print(f"Memory at {max(results.keys())}: {mem_16384:.1f} MB")
    print(f"Growth ratio: {growth_ratio:.1f}x")
    print(f"Expected for O(n): {expected_linear:.1f}x")
    
    if growth_ratio < expected_linear * 0.5:
        print("✅ Memory growth is sub-linear (HoloLink working as expected)")
    else:
        print("⚠️ Memory growth is near-linear (check for memory leaks)")
    
    return results


if __name__ == "__main__":
    run_memory_profile()
