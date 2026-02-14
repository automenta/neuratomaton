#!/usr/bin/env python
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ana.experiments.automated_researcher import AutomatedResearcher
from ana.experiments.potential_reveal import PotentialRevealer

def main():
    parser = argparse.ArgumentParser(description="ANA Automated Research Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run in fast smoketest mode")
    parser.add_argument("--tune", action="store_true", help="Force hyperparameter tuning stage")
    parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--output_dir", type=str, default="results/automated", help="Output directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation/tuning phase")
    parser.add_argument("--skip-potential", action="store_true", help="Skip potential/capability phase")

    # Flags for potential experiments
    parser.add_argument("--induction", action="store_true")
    parser.add_argument("--generalization", action="store_true")
    parser.add_argument("--multiquery", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all experiments")

    args = parser.parse_args()

    # Default to all if no specific flag set for potential
    if not (args.induction or args.generalization or args.multiquery or
            args.reasoning or args.noise or args.curriculum or args.sensitivity):
        args.all = True

    print("="*60)
    print("ANA AUTOMATED RESEARCHER & POTENTIAL REVEALER")
    print("Maximally automated discovery process.")
    print(f"Quick Mode: {args.quick}")
    print(f"Adaptive Tuning: {'Enabled (Auto)' if not args.tune else 'Forced'}")
    print("="*60)

    # We use the same timestamped directory for both if possible
    # AutomatedResearcher creates a timestamped dir. We need to capture it.

    current_output_dir = None

    if not args.skip_validation:
        print("\n\033[1;35m>>> PHASE 1: VALIDATION & SCALING <<<\033[0m")
        researcher = AutomatedResearcher(output_dir=args.output_dir)
        researcher.run_pipeline(quick=args.quick, tune=args.tune, trials=args.trials)

        current_output_dir = researcher.runner.output_dir

        if researcher.status != "completed":
            print("\033[1;31m[STOP] Validation failed. Aborting Phase 2.\033[0m")
            return

    if not args.skip_potential:
        print("\n\033[1;35m>>> PHASE 2: POTENTIAL & CAPABILITIES <<<\033[0m")
        # If we skipped validation, we create a new dir
        if current_output_dir is None:
             # PotentialRevealer handles creation
             revealer = PotentialRevealer(output_dir=args.output_dir)
        else:
             # Use the same dir
             revealer = PotentialRevealer(output_dir=os.path.dirname(current_output_dir))
             # Hack: PotentialRevealer creates its own timestamp usually.
             # Let's just point it to the same base and let it make a new timestamp or subfolder?
             # ComparisonRunner (parent) creates timestamped dir in __init__.
             # If we want to reuse, we'd need to modify ComparisonRunner.
             # For simplicity, let's let it create a new timestamped folder, but we link them mentally.
             pass

        revealer = PotentialRevealer(output_dir=args.output_dir)

        print(f"Running Potential Experiments in: {revealer.output_dir}")

        print("--- 2.1 Induction ---")
        revealer.run_induction_head_experiment(quick=args.quick)

        print("--- 2.2 Generalization ---")
        revealer.run_length_generalization_experiment(quick=args.quick)

        print("--- 2.3 Reasoning ---")
        revealer.run_reasoning_experiment(quick=args.quick)

        print("--- 2.4 Multi-Query ---")
        revealer.run_multi_query_experiment(quick=args.quick)

        print("--- 2.5 Noise Robustness ---")
        revealer.run_noise_robustness_experiment(quick=args.quick)

        print("--- 2.6 Curriculum Learning ---")
        revealer.run_curriculum_experiment(quick=args.quick)

        print("--- 2.7 Sensitivity ---")
        revealer.run_sensitivity_experiment(quick=args.quick)

        revealer.generate_potential_report()
        print(f"Potential Report generated at: {os.path.join(revealer.output_dir, 'POTENTIAL_REPORT.md')}")

    print("\n\033[1;32m=== TURNKEY RESEARCH COMPLETE ===\033[0m")

if __name__ == "__main__":
    main()
