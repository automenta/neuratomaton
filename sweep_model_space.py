"""
Model Space Sweep - Systematic shallow exploration of architecture space

This script runs a comprehensive sweep over the model space to understand
which components and architectures work best for different tasks.
"""

import os
import sys
import torch
import json
from typing import Dict, List

from ana.model_space import (
    ArchitectureSpec,
    get_architecture,
    list_architectures,
    generate_ablation_study,
    generate_grid_search_space,
)
from ana.model_factory import build_model
from ana.benchmark import BenchmarkSuite, test_capacity


# ============================================================================
# CONFIGURATION
# ============================================================================

SWEEP_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs_per_task": 8,  # Shallow testing
    "output_dir": "archive/sweep_results",
    "save_checkpoints": False,
}

TASK_PRIORITIES = {
    "single_kv": 1,      # Most important - verifies basic learning
    "multi_kv": 2,       # Tests capacity
    "copy": 3,           # Tests sequential memory
    "reverse": 4,        # Tests bidirectional capability
    "arithmetic": 5,     # Tests computation
}


# ============================================================================
# SWEEP FUNCTIONS
# ============================================================================

def sweep_predefined_architectures() -> Dict[str, Dict]:
    """Test all predefined architectures."""
    print("=" * 70)
    print("SWEEP 1: PREDEFINED ARCHITECTURES")
    print("=" * 70)

    results = {}
    arch_names = list_architectures()

    for name in arch_names:
        print(f"\nTesting architecture: {name}")
        print("-" * 50)

        spec = get_architecture(name)
        model = build_model(spec).to(SWEEP_CONFIG["device"])

        suite = BenchmarkSuite(device=SWEEP_CONFIG["device"])
        results[name] = suite.run_quick_benchmark(model)

        # Print summary
        print(f"  Results:")
        for task, acc in results[name].items():
            print(f"    {task:15s}: {acc*100:>6.1f}%")

    return results


def sweep_ablations(base_name: str = "ana_full") -> Dict[str, Dict]:
    """Test ablation variants."""
    print("\n" + "=" * 70)
    print(f"SWEEP 2: ABLATION STUDY (base={base_name})")
    print("=" * 70)

    variants = generate_ablation_study(base_name)
    results = {}

    for variant_name, spec in variants.items():
        print(f"\nTesting variant: {variant_name}")
        print("-" * 50)

        model = build_model(spec).to(SWEEP_CONFIG["device"])

        suite = BenchmarkSuite(device=SWEEP_CONFIG["device"])
        results[variant_name] = suite.run_quick_benchmark(model)

        # Print summary
        print(f"  Results:")
        for task, acc in results[variant_name].items():
            print(f"    {task:15s}: {acc*100:>6.1f}%")

    return results


def sweep_grid_sample(n_samples: int = 20) -> Dict[str, Dict]:
    """Sample from the full grid search space."""
    print("\n" + "=" * 70)
    print(f"SWEEP 3: GRID SEARCH SAMPLE (n={n_samples})")
    print("=" * 70)

    all_configs = generate_grid_search_space()
    import random
    random.seed(42)
    sampled_configs = random.sample(all_configs, min(n_samples, len(all_configs)))

    results = {}

    for i, spec in enumerate(sampled_configs):
        print(f"\nTesting config {i+1}/{n_samples}: {spec.name}")
        print("-" * 50)

        model = build_model(spec).to(SWEEP_CONFIG["device"])

        suite = BenchmarkSuite(device=SWEEP_CONFIG["device"])
        results[spec.name] = suite.run_quick_benchmark(model)

        # Print summary
        print(f"  Results:")
        for task, acc in results[spec.name].items():
            print(f"    {task:15s}: {acc*100:>6.1f}%")

    return results


def sweep_capacity_test() -> Dict[str, Dict]:
    """Test capacity of top architectures."""
    print("\n" + "=" * 70)
    print("SWEEP 4: CAPACITY TEST (Multi-KV scaling)")
    print("=" * 70)

    # Test top architectures
    arch_names = ["baseline_ssm", "multi_track_ssm", "hololink_ssm", "ana_full", "transformer"]
    results = {}

    for name in arch_names:
        print(f"\nTesting capacity of: {name}")
        print("-" * 50)

        spec = get_architecture(name)

        def model_factory():
            return build_model(spec)

        capacity_results = test_capacity(model_factory, device=SWEEP_CONFIG["device"], max_kv_pairs=8)
        results[name] = capacity_results

        # Print summary
        print(f"  Capacity Results:")
        for kv, acc in capacity_results.items():
            print(f"    {kv} KV pairs: {acc*100:>6.1f}%")

    return results


# ============================================================================
# RESULT ANALYSIS
# ============================================================================

def analyze_results(all_results: Dict) -> Dict:
    """Analyze results and find patterns."""
    analysis = {
        "best_by_task": {},
        "top_overall": [],
        "capacity_analysis": {},
    }

    # Find best architecture for each task
    all_archs = set()
    for sweep_name, sweep_results in all_results.items():
        if sweep_name == "capacity":
            continue

        for arch_name, task_results in sweep_results.items():
            all_archs.add(arch_name)

    for task in TASK_PRIORITIES.keys():
        best_acc = 0
        best_arch = None

        for sweep_name, sweep_results in all_results.items():
            if sweep_name == "capacity":
                continue

            for arch_name, task_results in sweep_results.items():
                if task in task_results:
                    acc = task_results[task]
                    if acc > best_acc:
                        best_acc = acc
                        best_arch = arch_name

        analysis["best_by_task"][task] = {"arch": best_arch, "accuracy": best_acc}

    # Analyze capacity
    if "capacity" in all_results:
        capacity_data = all_results["capacity"]
        for arch_name, kv_results in capacity_data.items():
            # Find cliff (where accuracy drops below 2x random baseline)
            random_baseline = 1.0 / 16  # Approx
            cliff = 1
            for kv in sorted(kv_results.keys()):
                if kv_results[kv] < 2 * random_baseline:
                    cliff = kv
                    break

            analysis["capacity_analysis"][arch_name] = {
                "cliff_at": cliff,
                "kv_results": kv_results,
            }

    return analysis


def print_summary(all_results: Dict, analysis: Dict):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    print("\nBEST ARCHITECTURE PER TASK:")
    print("-" * 50)
    print(f"{'Task':<15} {'Architecture':<25} {'Accuracy':>10}")
    print("-" * 50)
    for task, best in analysis["best_by_task"].items():
        arch = best["arch"] or "None"
        acc = best["accuracy"] * 100
        print(f"{task:<15} {arch:<25} {acc:>9.1f}%")

    if analysis["capacity_analysis"]:
        print("\nCAPACITY ANALYSIS (KV pairs before degradation):")
        print("-" * 50)
        for arch_name, cap in analysis["capacity_analysis"].items():
            print(f"{arch_name:<25} {cap['cliff_at']:>2} KV pairs")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete sweep."""
    print("ANA MODEL SPACE SWEEP")
    print("=" * 70)
    print(f"Device: {SWEEP_CONFIG['device']}")
    print(f"Epochs per task: {SWEEP_CONFIG['epochs_per_task']}")
    print(f"Output dir: {SWEEP_CONFIG['output_dir']}")
    print("=" * 70)

    os.makedirs(SWEEP_CONFIG["output_dir"], exist_ok=True)

    all_results = {}

    # Run sweeps
    try:
        all_results["predefined"] = sweep_predefined_architectures()
    except Exception as e:
        print(f"Error in predefined sweep: {e}")
        all_results["predefined"] = {"error": str(e)}

    try:
        all_results["ablations"] = sweep_ablations("ana_full")
    except Exception as e:
        print(f"Error in ablation sweep: {e}")
        all_results["ablations"] = {"error": str(e)}

    try:
        all_results["grid_sample"] = sweep_grid_sample(n_samples=10)
    except Exception as e:
        print(f"Error in grid sweep: {e}")
        all_results["grid_sample"] = {"error": str(e)}

    try:
        all_results["capacity"] = sweep_capacity_test()
    except Exception as e:
        print(f"Error in capacity sweep: {e}")
        all_results["capacity"] = {"error": str(e)}

    # Analyze results
    analysis = analyze_results(all_results)
    print_summary(all_results, analysis)

    # Save results
    output_path = os.path.join(SWEEP_CONFIG["output_dir"], "sweep_results.json")

    # Convert results to JSON-serializable format
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, (float, int)):
            return obj
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return str(obj)

    serializable_results = convert(all_results)

    with open(output_path, 'w') as f:
        json.dump({
            "sweep_config": SWEEP_CONFIG,
            "results": serializable_results,
            "analysis": convert(analysis),
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
