
import torch
import time
import json
import os
from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM

def benchmark_model(model, input_ids, device, name):
    model.to(device)
    model.train()

    print(f"Benchmarking {name}...")

    # Warmup
    for _ in range(5):
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()
        model.zero_grad()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    start_time = time.time()
    num_iters = 20

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

    print(f"{name}: Avg Time {avg_time*1000:.2f} ms | Throughput {tokens_per_sec:.2f} tok/s")

    return {
        "model": name,
        "avg_time_ms": avg_time * 1000,
        "throughput_tok_s": tokens_per_sec
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")

    # Ensure output dir exists
    if not os.path.exists("archive/results"):
        os.makedirs("archive/results")

    # Hyperparams for benchmark
    batch_size = 4
    seq_len = 128
    vocab_size = 40
    d_model = 64
    state_dim = 64
    num_layers = 2

    print(f"Config: Batch={batch_size}, Seq={seq_len}, Dim={d_model}")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    results = []

    # 1. Baseline SSM (Sequential)
    config_base_seq = ANAConfig(d_model=d_model, state_dim=state_dim, num_layers=num_layers, vocab_size=vocab_size, use_parallel_scan=False)
    model_base_seq = BaselineSSM(config_base_seq)
    results.append(benchmark_model(model_base_seq, input_ids, device, "Baseline (Seq)"))

    # 2. Baseline SSM (Parallel)
    config_base_par = ANAConfig(d_model=d_model, state_dim=state_dim, num_layers=num_layers, vocab_size=vocab_size, use_parallel_scan=True)
    model_base_par = BaselineSSM(config_base_par)
    results.append(benchmark_model(model_base_par, input_ids, device, "Baseline (Par)"))

    # 3. ANA (Sequential)
    config_ana_seq = ANAConfig(d_model=d_model, state_dim=state_dim, num_layers=num_layers, vocab_size=vocab_size, use_parallel_scan=False)
    model_ana_seq = ANAModel(config_ana_seq)
    results.append(benchmark_model(model_ana_seq, input_ids, device, "ANA (Seq)"))

    # 4. ANA (Parallel)
    config_ana_par = ANAConfig(d_model=d_model, state_dim=state_dim, num_layers=num_layers, vocab_size=vocab_size, use_parallel_scan=True)
    model_ana_par = ANAModel(config_ana_par)
    results.append(benchmark_model(model_ana_par, input_ids, device, "ANA (Par)"))

    # Save
    with open("archive/results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
