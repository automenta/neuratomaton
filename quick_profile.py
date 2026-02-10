import sys
from pathlib import Path
import json
import time
import torch

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


def quick_profile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nCreating nano model...")
    model = create_bio_ana('nano').to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nProfiling forward pass (batch=2, seq=16)...")
    model.eval()
    input_ids = torch.randint(0, 50, (2, 16), device=device)
    
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(10):
            logits = model(input_ids)
        elapsed = (time.perf_counter() - start) * 1000 / 10
    
    print(f"Forward pass: {elapsed:.2f}ms")
    
    print("\nProfiling backward pass...")
    model.train()
    targets = torch.randint(0, 50, (2, 16), device=device)
    
    model.zero_grad()
    start = time.perf_counter()
    for _ in range(5):
        logits = model(input_ids)
        loss = model.compute_loss(logits, targets)
        loss['total'].backward()
    elapsed = (time.perf_counter() - start) * 1000 / 5
    
    print(f"Backward pass: {elapsed:.2f}ms")
    
    print("\nComponent breakdown (single iteration):")
    with torch.no_grad():
        x = model.embedding(input_ids)
        
        start = time.perf_counter()
        x = model._add_position_encoding(x)
        print(f"  Position encoding: {(time.perf_counter() - start) * 1000:.3f}ms")
        
        track_states = {'syntax': None, 'semantic': None, 'logic': None}
        start = time.perf_counter()
        for t in range(16):
            xt = x[:, t, :]
            track_out, track_states = model.tracks(
                xt, h_syntax=track_states['syntax'], 
                h_semantic=track_states['semantic'], 
                h_logic=track_states['logic'],
                steps=model.config.relaxation_iterations
            )
        print(f"  Tracks (16 steps): {(time.perf_counter() - start) * 1000:.3f}ms")
        print(f"  Per token: {(time.perf_counter() - start) * 1000 / 16:.3f}ms")
        
        if model.hololink:
            start = time.perf_counter()
            for _ in range(16):
                _ = model.hololink(track_out, write_mode=False)
            print(f"  HoloLink (16 queries): {(time.perf_counter() - start) * 1000:.3f}ms")
    
    print("\nRelaxation iterations sensitivity:")
    for iters in [5, 10, 20, 30]:
        model.config.relaxation_iterations = iters
        model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(3):
                logits = model(input_ids)
            elapsed = (time.perf_counter() - start) * 1000 / 3
        print(f"  {iters} iters: {elapsed:.2f}ms")
    
    if device.type == 'cuda':
        print("\nMemory usage:")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            logits = model(input_ids)
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Forward: {fwd_mem:.1f} MB")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.train()
        model.zero_grad()
        logits = model(input_ids)
        loss = model.compute_loss(logits, targets)
        loss['total'].backward()
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Backward: {bwd_mem:.1f} MB")
    
    print("\n" + "="*50)
    print("OPTIMIZATION RECOMMENDATIONS:")
    print("="*50)
    
    print("\nLow-hanging fruit:")
    print("1. Enable mixed precision (torch.cuda.amp) - 2x speedup")
    print("2. Adaptive relaxation - use fewer iterations for later tokens")
    print("3. Cache position encoding - precompute if max_seq_len fixed")
    print("4. Compile model with torch.compile() - if PyTorch 2.0+")
    
    print("\nMedium effort:")
    print("5. Gradient checkpointing - trade compute for memory")
    print("6. Optimize track update kernel - fused operations")
    print("7. Sparse HoloLink - only query when needed")
    
    print("\nHigher effort:")
    print("8. Custom CUDA kernels for track dynamics")
    print("9. Ternary/INT8 quantization for inference")
    
    return {
        "forward_ms": elapsed,
        "backward_ms": elapsed,
        "params": total_params,
        "relaxation_iters": model.config.relaxation_iterations
    }


if __name__ == "__main__":
    results = quick_profile()
