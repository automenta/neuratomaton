#!/usr/bin/env python
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ana.experiments.automated_researcher import AutomatedResearcher

def main():
    parser = argparse.ArgumentParser(description="ANA Automated Research Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run in fast smoketest mode")
    parser.add_argument("--tune", action="store_true", help="Force hyperparameter tuning stage")
    parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--output_dir", type=str, default="results/automated", help="Output directory")

    args = parser.parse_args()

    print("="*60)
    print("ANA AUTOMATED RESEARCHER")
    print("Maximally automated discovery process.")
    print(f"Quick Mode: {args.quick}")
    print(f"Adaptive Tuning: {'Enabled (Auto)' if not args.tune else 'Forced'}")
    print("="*60)

    researcher = AutomatedResearcher(output_dir=args.output_dir)
    researcher.run_pipeline(quick=args.quick, tune=args.tune, trials=args.trials)

    print("\nCheck results in:", researcher.runner.output_dir)

if __name__ == "__main__":
    main()
