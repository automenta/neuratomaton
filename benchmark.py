import torch
import time
import json
import os
from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM

def benchmark_model(model, input_ids, device, name, num_iters=20, warmup=5):
    model.to(device)
    model.train()
    
    print(f"Benchmarking {name}...")
    
    for _ in range(warmup):
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()
        model.zero_grad()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(num_iters):
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()
        model.zero_grad()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iters
    tokens_per_sec = (input_ids.numel() * num_iters) / (end_time - start_time)
    
    mem_allocated = 0
    mem_reserved = 0
    if device.type == 'cuda':
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**2
        mem_reserved = torch.cuda.max_memory_reserved() / 1024**2
        torch.cuda.reset_peak_memory_stats()
    
    print(f"{name}: {avg_time*1000:.2f} ms | {tokens_per_sec:.0f} tok/s | Mem: {mem_allocated:.1f} MB")
    
    return {
        "model": name,
        "avg_time_ms": round(avg_time * 1000, 2),
        "throughput_tok_s": round(tokens_per_sec, 0),
        "memory_mb": round(mem_allocated, 1) if device.type == 'cuda' else 0
    }

def run_benchmarks(batch_sizes=[4, 8], seq_lens=[64, 128], d_models=[64, 128]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    os.makedirs("archive/results", exist_ok=True)
    
    all_results = []
    vocab_size = 40
    state_dim = 64
    num_layers = 2
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for d_model in d_models:
                config_dict = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'd_model': d_model
                }
                
                print(f"\nConfig: Batch={batch_size}, Seq={seq_len}, Dim={d_model}")
                
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
                
                config_base_seq = ANAConfig(
                    d_model=d_model, state_dim=state_dim, num_layers=num_layers,
                    vocab_size=vocab_size, use_parallel_scan=False
                )
                model_base_seq = BaselineSSM(config_base_seq)
                r1 = benchmark_model(model_base_seq, input_ids, device, f"Baseline_Seq_b{batch_size}s{seq_len}d{d_model}")
                r1.update(config_dict)
                all_results.append(r1)
                
                config_base_par = ANAConfig(
                    d_model=d_model, state_dim=state_dim, num_layers=num_layers,
                    vocab_size=vocab_size, use_parallel_scan=True
                )
                model_base_par = BaselineSSM(config_base_par)
                r2 = benchmark_model(model_base_par, input_ids, device, f"Baseline_Par_b{batch_size}s{seq_len}d{d_model}")
                r2.update(config_dict)
                all_results.append(r2)
                
                config_ana_seq = ANAConfig(
                    d_model=d_model, state_dim=state_dim, num_layers=num_layers,
                    vocab_size=vocab_size, use_parallel_scan=False
                )
                model_ana_seq = ANAModel(config_ana_seq)
                r3 = benchmark_model(model_ana_seq, input_ids, device, f"ANA_Seq_b{batch_size}s{seq_len}d{d_model}")
                r3.update(config_dict)
                all_results.append(r3)
                
                config_ana_par = ANAConfig(
                    d_model=d_model, state_dim=state_dim, num_layers=num_layers,
                    vocab_size=vocab_size, use_parallel_scan=True
                )
                model_ana_par = ANAModel(config_ana_par)
                r4 = benchmark_model(model_ana_par, input_ids, device, f"ANA_Par_b{batch_size}s{seq_len}d{d_model}")
                r4.update(config_dict)
                all_results.append(r4)
                
                del model_base_seq, model_base_par, model_ana_seq, model_ana_par
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    with open("archive/results/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to archive/results/benchmark_results.json")
    
    print("\n=== Summary ===")
    baseline_seq_avg = np.mean([r['throughput_tok_s'] for r in all_results if 'Baseline_Seq' in r['model']])
    baseline_par_avg = np.mean([r['throughput_tok_s'] for r in all_results if 'Baseline_Par' in r['model']])
    ana_seq_avg = np.mean([r['throughput_tok_s'] for r in all_results if 'ANA_Seq' in r['model']])
    ana_par_avg = np.mean([r['throughput_tok_s'] for r in all_results if 'ANA_Par' in r['model']])
    
    print(f"Baseline Sequential: {baseline_seq_avg:.0f} tok/s avg")
    print(f"Baseline Parallel:   {baseline_par_avg:.0f} tok/s avg ({baseline_par_avg/baseline_seq_avg:.2f}x)")
    print(f"ANA Sequential:      {ana_seq_avg:.0f} tok/s avg")
    print(f"ANA Parallel:        {ana_par_avg:.0f} tok/s avg ({ana_par_avg/ana_seq_avg:.2f}x)")
    
    return all_results

def main():
    import numpy as np
    run_benchmarks()

if __name__ == "__main__":
    main()
