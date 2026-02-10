import sys
from pathlib import Path
import json
import time
from typing import Dict, Any
import torch
import torch.autograd.profiler as profiler

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


class ProfilerMetrics:
    def __init__(self):
        self.metrics = {}
    
    def add(self, name: str, value: float, unit: str = "ms"):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"value": value, "unit": unit})
    
    def summary(self) -> Dict[str, Any]:
        summary = {}
        for name, values in self.metrics.items():
            if values:
                arr = [v["value"] for v in values]
                unit = values[0]["unit"]
                summary[name] = {
                    "mean": sum(arr) / len(arr),
                    "min": min(arr),
                    "max": max(arr),
                    "std": (sum((x - sum(arr) / len(arr)) ** 2 for x in arr) / len(arr)) ** 0.5,
                    "unit": unit,
                    "count": len(arr)
                }
        return summary


def profile_forward_pass(model, input_ids, iterations=10):
    metrics = ProfilerMetrics()
    
    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            start = time.perf_counter()
            _ = model(input_ids)
            elapsed = (time.perf_counter() - start) * 1000
            metrics.add("forward_pass", elapsed)
    
    return metrics.summary()


def profile_backward_pass(model, input_ids, targets, iterations=10):
    metrics = ProfilerMetrics()
    
    model.train()
    for i in range(iterations):
        model.zero_grad()
        start = time.perf_counter()
        logits = model(input_ids)
        loss = model.compute_loss(logits, targets)
        elapsed_forward = (time.perf_counter() - start) * 1000
        metrics.add("backward_forward", elapsed_forward)
        
        start = time.perf_counter()
        loss['total'].backward()
        elapsed_backward = (time.perf_counter() - start) * 1000
        metrics.add("backward_pass", elapsed_backward)
        
        start = time.perf_counter()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        elapsed_cleanup = (time.perf_counter() - start) * 1000
        metrics.add("backward_cleanup", elapsed_cleanup)
    
    return metrics.summary()


def profile_components(model, input_ids, iterations=10):
    metrics = ProfilerMetrics()
    
    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            x = model.embedding(input_ids)
            
            start = time.perf_counter()
            x = model._add_position_encoding(x)
            elapsed_pe = (time.perf_counter() - start) * 1000
            metrics.add("position_encoding", elapsed_pe)
            
            track_states = {
                'syntax': None,
                'semantic': None,
                'logic': None
            }
            
            start = time.perf_counter()
            for t in range(input_ids.size(1)):
                xt = x[:, t, :]
                track_out, track_states = model.tracks(
                    xt,
                    h_syntax=track_states['syntax'],
                    h_semantic=track_states['semantic'],
                    h_logic=track_states['logic'],
                    steps=model.config.relaxation_iterations,
                )
            elapsed_tracks = (time.perf_counter() - start) * 1000
            metrics.add("tracks_processing", elapsed_tracks)
            
            if model.hololink:
                start = time.perf_counter()
                track_states = {
                    'syntax': torch.randn(input_ids.size(0), model.config.syntax_dim),
                    'semantic': torch.randn(input_ids.size(0), model.config.semantic_dim),
                    'logic': torch.randn(input_ids.size(0), model.config.logic_dim),
                }
                track_out = torch.randn(input_ids.size(0), model.config.total_track_dim)
                _ = model.hololink(track_out, write_mode=False)
                elapsed_hololink = (time.perf_counter() - start) * 1000
                metrics.add("hololink_query", elapsed_hololink)
    
    return metrics.summary()


def profile_memory_usage(model, batch_size=4, seq_len=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    metrics = {}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)
    forward_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    metrics['forward_memory_mb'] = forward_mem
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    model.zero_grad()
    logits = model(input_ids)
    loss = model.compute_loss(logits, targets)
    loss['total'].backward()
    
    backward_mem = torch.cuda.max_memory_allocated() / 1024**2
    metrics['backward_memory_mb'] = backward_mem
    
    torch.cuda.empty_cache()
    
    return metrics


def profile_with_torch_profiler(model, input_ids, targets, output_dir: Path):
    device = next(model.parameters()).device
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA if device.type == 'cuda' else None
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        model.train()
        logits = model(input_ids)
        loss = model.compute_loss(logits, targets)
        loss['total'].backward()
    
    trace_file = output_dir / "profile_trace.json"
    prof.export_chrome_trace(str(trace_file))
    
    table_file = output_dir / "profile_table.txt"
    with open(table_file, 'w') as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total" if device.type == 'cuda' else "cpu_time_total", row_limit=20))
    
    return {
        "trace_file": str(trace_file),
        "table_file": str(table_file),
        "top_ops": prof.key_averages().table(sort_by="cuda_time_total" if device.type == 'cuda' else "cpu_time_total", row_limit=10)
    }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    details = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            details[name] = param.numel()
    
    return {
        "total": total,
        "trainable": trainable,
        "details": details
    }


def run_profiling_suite(output_dir: Path = None):
    if output_dir is None:
        output_dir = Path("results/profiling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on device: {device}")
    
    variants = ['nano', 'small']
    batch_sizes = [1, 4, 8]
    seq_lengths = [16, 64, 128]
    
    all_results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Profiling variant: {variant}")
        print(f"{'='*60}")
        
        config = get_bio_config(variant)
        model = create_bio_ana(variant).to(device)
        
        variant_results = {
            "variant": variant,
            "config": {
                "d_model": config.d_model,
                "syntax_dim": config.syntax_dim,
                "semantic_dim": config.semantic_dim,
                "logic_dim": config.logic_dim,
                "relaxation_iterations": config.relaxation_iterations,
            },
            "parameters": count_parameters(model),
            "profiling": {}
        }
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"  batch={batch_size}, seq_len={seq_len}")
                
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
                targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
                
                key = f"bs{batch_size}_seq{seq_len}"
                
                forward_metrics = profile_forward_pass(model, input_ids, iterations=5)
                backward_metrics = profile_backward_pass(model, input_ids, targets, iterations=5)
                
                if device.type == 'cuda':
                    memory_metrics = profile_memory_usage(model, batch_size, seq_len)
                else:
                    memory_metrics = {"cpu_mode": True}
                
                variant_results["profiling"][key] = {
                    "forward": forward_metrics,
                    "backward": backward_metrics,
                    "memory": memory_metrics
                }
        
        component_metrics = profile_components(model, input_ids, iterations=10)
        variant_results["component_breakdown"] = component_metrics
        
        if device.type == 'cuda':
            profiler_output = output_dir / variant
            profiler_output.mkdir(exist_ok=True)
            detailed_trace = profile_with_torch_profiler(model, input_ids, targets, profiler_output)
            variant_results["detailed_trace"] = detailed_trace
        
        all_results[variant] = variant_results
        
        output_file = output_dir / f"{variant}_profile.json"
        with open(output_file, 'w') as f:
            json.dump(variant_results, f, indent=2)
        
        print(f"\nSaved results to: {output_file}")
    
    summary_file = output_dir / "profiling_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PROFILING SUMMARY")
    print(f"{'='*60}")
    
    for variant, results in all_results.items():
        print(f"\n{variant.upper()} ({results['parameters']['total']:,} params)")
        print(f"  Forward pass times (ms):")
        for key, metrics in results["profiling"].items():
            fwd = metrics["forward"].get("forward_pass", {})
            if fwd:
                print(f"    {key}: {fwd['mean']:.2f}ms ± {fwd['std']:.2f}ms")
    
    return all_results


def identify_optimization_opportunities(results: Dict[str, Any]):
    opportunities = []
    
    for variant, data in results.items():
        for config_key, metrics in data["profiling"].items():
            fwd = metrics["forward"].get("forward_pass", {})
            bwd = metrics["backward"].get("backward_pass", {})
            
            if fwd and fwd["mean"] > 50:
                opportunities.append({
                    "variant": variant,
                    "config": config_key,
                    "issue": "Slow forward pass",
                    "metric": fwd["mean"],
                    "suggestion": "Consider reducing relaxation_iterations or enabling mixed precision"
                })
            
            if bwd and bwd.get("backward_pass", {}).get("mean", 0) > 100:
                opportunities.append({
                    "variant": variant,
                    "config": config_key,
                    "issue": "Slow backward pass",
                    "metric": bwd["backward_pass"]["mean"],
                    "suggestion": "Enable gradient checkpointing or reduce model size"
                })
            
            if metrics["memory"].get("backward_memory_mb", 0) > 4000:
                opportunities.append({
                    "variant": variant,
                    "config": config_key,
                    "issue": "High memory usage",
                    "metric": metrics["memory"]["backward_memory_mb"],
                    "suggestion": "Reduce batch size or enable gradient accumulation"
                })
    
    component_data = data.get("component_breakdown", {})
    for component, stats in component_data.items():
        if stats.get("mean", 0) > 10:
            opportunities.append({
                "variant": variant,
                "config": "components",
                "issue": f"Slow component: {component}",
                "metric": stats["mean"],
                "suggestion": "Optimize or cache {component} computation"
            })
    
    return opportunities


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))
    
    output_dir = Path("results/profiling")
    results = run_profiling_suite(output_dir)
    
    opportunities = identify_optimization_opportunities(results)
    
    opp_file = output_dir / "optimization_opportunities.json"
    with open(opp_file, 'w') as f:
        json.dump(opportunities, f, indent=2)
    
    print(f"\nFound {len(opportunities)} optimization opportunities")
    print(f"Saved to: {opp_file}")
