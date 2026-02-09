import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
from tqdm import tqdm

class MultiQueryARDataset(Dataset):
    def __init__(self, size=1000, vocab_size=100, num_kv_pairs=16, noise_multiplier=2):
        self.size = size
        self.vocab_size = vocab_size
        self.num_kv_pairs = num_kv_pairs
        self.noise_multiplier = noise_multiplier
        
        self.KEY_MARKER = 1
        self.VAL_MARKER = 2
        self.QUERY_MARKER = 3
        self.SEP_MARKER = 4
        self.content_start = 5
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        content_tokens = list(range(self.content_start, self.vocab_size))
        
        kv_pairs = []
        keys_used = random.sample(content_tokens, self.num_kv_pairs)
        vals_used = random.sample([t for t in content_tokens if t not in keys_used], self.num_kv_pairs)
        
        for k, v in zip(keys_used, vals_used):
            kv_pairs.append([self.KEY_MARKER, k, self.VAL_MARKER, v])
        
        noise_len = self.num_kv_pairs * self.noise_multiplier * 4
        noise = [random.choice(content_tokens) for _ in range(noise_len)]
        
        query_idx = random.randint(0, self.num_kv_pairs - 1)
        query_key = keys_used[query_idx]
        target_val = vals_used[query_idx]
        
        query = [self.QUERY_MARKER, query_key]
        
        seq = []
        for kv in kv_pairs:
            seq.extend(kv)
        seq.extend(noise)
        seq.extend(query)
        seq.append(target_val)
        
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[-1] = 1.0
        
        return x, y, mask

class InductionHeadDataset(Dataset):
    def __init__(self, size=1000, vocab_size=50, seq_len=20, pattern_len=4):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pattern_len = pattern_len
        
        self.MARKER = 1
        self.content_start = 2
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        content_tokens = list(range(self.content_start, self.vocab_size))
        
        pattern = [random.choice(content_tokens) for _ in range(self.pattern_len)]
        
        noise_len = self.seq_len - 2 * self.pattern_len - 2
        noise = [random.choice(content_tokens) for _ in range(noise_len)]
        
        target = pattern[0]
        
        seq = [self.MARKER] + pattern + noise + pattern + [self.MARKER, target]
        
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[-1] = 1.0
        
        return x, y, mask

class LongContextARDataset(Dataset):
    def __init__(self, size=100, vocab_size=50, context_len=1000, kv_position='start'):
        self.size = size
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.kv_position = kv_position
        
        self.KEY_MARKER = 1
        self.VAL_MARKER = 2
        self.QUERY_MARKER = 3
        self.content_start = 4
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        content_tokens = list(range(self.content_start, self.vocab_size))
        
        key_token = random.choice(content_tokens)
        val_token = random.choice([t for t in content_tokens if t != key_token])
        
        kv_seq = [self.KEY_MARKER, key_token, self.VAL_MARKER, val_token]
        
        noise = [random.choice(content_tokens) for _ in range(self.context_len)]
        
        query = [self.QUERY_MARKER, key_token]
        target = val_token
        
        if self.kv_position == 'start':
            seq = kv_seq + noise + query + [target]
        elif self.kv_position == 'middle':
            mid = len(noise) // 2
            seq = noise[:mid] + kv_seq + noise[mid:] + query + [target]
        else:
            seq = noise + kv_seq + query + [target]
        
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[-1] = 1.0
        
        return x, y, mask

class BenchmarkEvaluator:
    def __init__(self, model, device, vocab_size=50):
        self.model = model
        self.device = device
        self.vocab_size = vocab_size
        self.results = {}
    
    def col_fn(self, batch):
        has_mask = len(batch[0]) == 3
        max_len = max(item[0].size(0) for item in batch)
        
        xs, ys, ms = [], [], []
        for item in batch:
            if has_mask:
                x, y, mask = item
            else:
                x, y = item
                mask = None
            
            pad = max_len - x.size(0)
            if pad > 0:
                x = torch.cat([x, torch.zeros(pad, dtype=torch.long)])
                y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])
                if mask is not None:
                    mask = torch.cat([mask, torch.zeros(pad, dtype=torch.float)])
            
            xs.append(x)
            ys.append(y)
            if mask is not None:
                ms.append(mask)
        
        if has_mask:
            return torch.stack(xs), torch.stack(ys), torch.stack(ms)
        return torch.stack(xs), torch.stack(ys)
    
    def evaluate_task(self, dataset, name, batch_size=16):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.col_fn)
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    x, y, mask = batch
                    mask = mask.to(self.device)
                else:
                    x, y = batch
                    mask = None
                
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                
                last_logits = logits[:, -1, :]
                last_targets = y[:, -1]
                preds = torch.argmax(last_logits, dim=-1)
                
                correct = (preds == last_targets).float().sum().item()
                total_correct += correct
                total_samples += x.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        self.results[name] = accuracy
        return accuracy
    
    def run_associative_recall_sweep(self):
        print("\n=== Associative Recall (Single KV) ===")
        noise_levels = [10, 20, 50, 100]
        
        for noise in noise_levels:
            from ana.data import AssociativeRecallDataset
            ds = AssociativeRecallDataset(
                size=200, vocab_size=self.vocab_size,
                min_noise=noise, max_noise=noise
            )
            acc = self.evaluate_task(ds, f"AR_noise_{noise}")
            print(f"  Noise={noise}: {acc:.2%}")
    
    def run_mqar_sweep(self):
        print("\n=== Multi-Query Associative Recall ===")
        kv_counts = [4, 8, 16, 32, 64]
        
        for num_kv in kv_counts:
            required_vocab = num_kv * 2 + 10
            if self.vocab_size < required_vocab:
                print(f"  {num_kv} KV pairs: SKIPPED (vocab too small, need {required_vocab})")
                continue
            
            ds = MultiQueryARDataset(
                size=200, vocab_size=self.vocab_size,
                num_kv_pairs=num_kv, noise_multiplier=2
            )
            acc = self.evaluate_task(ds, f"MQAR_{num_kv}_pairs")
            print(f"  {num_kv} KV pairs: {acc:.2%}")
    
    def run_induction_heads(self):
        print("\n=== Induction Heads ===")
        seq_lens = [20, 40, 80]
        
        for seq_len in seq_lens:
            ds = InductionHeadDataset(
                size=200, vocab_size=self.vocab_size,
                seq_len=seq_len, pattern_len=4
            )
            acc = self.evaluate_task(ds, f"Induction_len_{seq_len}")
            print(f"  Seq len={seq_len}: {acc:.2%}")
    
    def run_long_context(self, max_len=500):
        print("\n=== Long Context Retrieval ===")
        positions = ['start', 'middle']
        
        for pos in positions:
            ds = LongContextARDataset(
                size=50, vocab_size=self.vocab_size,
                context_len=min(max_len, 500), kv_position=pos
            )
            acc = self.evaluate_task(ds, f"LongContext_{pos}")
            print(f"  Position={pos}: {acc:.2%}")
    
    def run_all_benchmarks(self, max_context=500):
        print("=" * 50)
        print("Running ANA Benchmark Suite")
        print("=" * 50)
        
        self.run_associative_recall_sweep()
        self.run_mqar_sweep()
        self.run_induction_heads()
        self.run_long_context(max_context)
        
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        
        for name, acc in sorted(self.results.items()):
            print(f"  {name}: {acc:.2%}")
        
        return self.results
    
    def save_results(self, path):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {path}")

def compare_models(ana_config, baseline_config, device, output_dir="archive/benchmarks"):
    from ana.models import ANAModel, BaselineSSM
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ANA vs BaselineSSM Comparison")
    print("=" * 60)
    
    results = {}
    
    print("\nInitializing models...")
    ana_model = ANAModel(ana_config).to(device)
    baseline_model = BaselineSSM(baseline_config).to(device)
    
    ana_params = sum(p.numel() for p in ana_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    print(f"  ANA parameters: {ana_params:,}")
    print(f"  Baseline parameters: {baseline_params:,}")
    
    results['params'] = {'ana': ana_params, 'baseline': baseline_params}
    
    print("\n--- Benchmarking ANA ---")
    ana_eval = BenchmarkEvaluator(ana_model, device, ana_config.vocab_size)
    ana_results = ana_eval.run_all_benchmarks()
    results['ana'] = ana_results
    
    print("\n--- Benchmarking Baseline ---")
    baseline_eval = BenchmarkEvaluator(baseline_model, device, baseline_config.vocab_size)
    baseline_results = baseline_eval.run_all_benchmarks()
    results['baseline'] = baseline_results
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<30} {'ANA':>10} {'Baseline':>10} {'Delta':>10}")
    print("-" * 60)
    
    for key in ana_results:
        ana_val = ana_results[key]
        base_val = baseline_results.get(key, 0)
        delta = ana_val - base_val
        sign = '+' if delta > 0 else ''
        print(f"{key:<30} {ana_val:>10.2%} {base_val:>10.2%} {sign}{delta:>9.2%}")
    
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    from ana.config import ANAConfig
    
    parser = argparse.ArgumentParser(description="ANA Benchmark Suite")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--state-dim", type=int, default=64, help="State dimension")
    parser.add_argument("--vocab-size", type=int, default=50, help="Vocabulary size")
    parser.add_argument("--compare", action="store_true", help="Compare ANA vs Baseline")
    parser.add_argument("--output", type=str, default="archive/benchmarks", help="Output directory")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = ANAConfig(
        d_model=args.d_model,
        state_dim=args.state_dim,
        vocab_size=args.vocab_size,
        num_layers=2,
        track_count=2,
        use_parallel_scan=True
    )
    
    if args.compare:
        compare_models(config, config, device, args.output)
    else:
        from ana.models import ANAModel
        
        model = ANAModel(config).to(device)
        
        if args.model:
            print(f"Loading checkpoint: {args.model}")
            model.load_state_dict(torch.load(args.model, map_location=device))
        
        evaluator = BenchmarkEvaluator(model, device, args.vocab_size)
        results = evaluator.run_all_benchmarks()
        
        os.makedirs(args.output, exist_ok=True)
        evaluator.save_results(os.path.join(args.output, "benchmark_results.json"))
