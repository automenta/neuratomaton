import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ana.experiments.comprehensive import ComparisonRunner

def main():
    parser = argparse.ArgumentParser(description="Run ANA Comprehensive Research Experiments")
    parser.add_argument("--task", type=str, choices=['scaling', 'ablation', 'throughput', 'all'], default='all', help="Task to run")
    parser.add_argument("--output_dir", type=str, default="results/comprehensive", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick smoketests")
    parser.add_argument("--steps", type=int, default=500, help="Training steps per experiment")

    args = parser.parse_args()

    runner = ComparisonRunner(output_dir=args.output_dir)

    if args.task in ['scaling', 'all']:
        runner.run_scaling_benchmark(steps_per_len=args.steps, quick=args.quick)

    if args.task in ['ablation', 'all']:
        runner.run_ablation_study(steps=args.steps, quick=args.quick)

    if args.task in ['throughput', 'all']:
        runner.run_throughput_benchmark(quick=args.quick)

    runner.generate_report()
    print("Experiments completed. Report generated.")

if __name__ == "__main__":
    main()
