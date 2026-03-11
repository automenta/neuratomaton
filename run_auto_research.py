#!/usr/bin/env python
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ana.experiments.automated_researcher import AutomatedResearcher
from ana.experiments.potential_reveal import PotentialRevealer
from ana.experiments.discovery import DiscoveryEngine
from ana.experiments.interactive_tuner import InteractiveTuner

def main():
    parser = argparse.ArgumentParser(description="ANA Automated Research Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run in fast smoketest mode")
    parser.add_argument("--tune", action="store_true", help="Force hyperparameter tuning stage")
    parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--output_dir", type=str, default="results/automated", help="Output directory")

    # Phases
    parser.add_argument("--validation", action="store_true", help="Run Phase 1: Validation & Scaling")
    parser.add_argument("--potential", action="store_true", help="Run Phase 2: Potential & Capabilities")
    parser.add_argument("--discovery", action="store_true", help="Run Phase 3: Scientific Discovery (Baselines, Optuna, Ablation)")
    parser.add_argument("--interactive", action="store_true", help="Run Interactive Hyperparameter Tuner")
    parser.add_argument("--all", action="store_true", help="Run all phases")

    # Flags for potential experiments (Phase 2 sub-flags)
    parser.add_argument("--induction", action="store_true")
    parser.add_argument("--generalization", action="store_true")
    parser.add_argument("--multiquery", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")

    args = parser.parse_args()

    # Interactive Mode takes precedence
    if args.interactive:
        print("="*60)
        print("ANA INTERACTIVE TUNER")
        print("Starting interactive session...")
        print("="*60)
        tuner = InteractiveTuner(output_dir=args.output_dir)
        try:
            tuner.cmdloop()
        except KeyboardInterrupt:
            tuner.do_exit(None)
        return

    # Check if any potential sub-flags are set, imply potential phase
    potential_flags = [args.induction, args.generalization, args.multiquery,
                       args.reasoning, args.noise, args.curriculum, args.sensitivity]
    if any(potential_flags):
        args.potential = True

    # Default logic: If no phase specified, run discovery (since that's the new enhancement)
    # Or if args.all, run all.
    if args.all:
        args.validation = True
        args.potential = True
        args.discovery = True

    # If nothing specified (and no implied potential), default to Discovery
    if not (args.validation or args.potential or args.discovery):
        print("No phase specified. Defaulting to Phase 3: Discovery.")
        args.discovery = True

    print("="*60)
    print("ANA AUTOMATED RESEARCHER")
    print("Maximally automated discovery process.")
    print(f"Quick Mode: {args.quick}")
    print("="*60)

    # Phase 1: Validation
    if args.validation:
        print("\n\033[1;35m>>> PHASE 1: VALIDATION & SCALING <<<\033[0m")
        researcher = AutomatedResearcher(output_dir=args.output_dir)
        researcher.run_pipeline(quick=args.quick, tune=args.tune, trials=args.trials)

        if researcher.status != "completed":
            print("\033[1;31m[STOP] Validation failed. Aborting subsequent phases.\033[0m")
            return

    # Phase 2: Potential
    if args.potential:
        print("\n\033[1;35m>>> PHASE 2: POTENTIAL & CAPABILITIES <<<\033[0m")

        # Determine sub-experiments
        run_all_potential = not (args.induction or args.generalization or args.multiquery or
                                 args.reasoning or args.noise or args.curriculum or args.sensitivity)

        revealer = PotentialRevealer(output_dir=args.output_dir)
        print(f"Running Potential Experiments in: {revealer.output_dir}")

        if run_all_potential or args.induction:
            print("--- 2.1 Induction ---")
            revealer.run_induction_head_experiment(quick=args.quick)

        if run_all_potential or args.generalization:
            print("--- 2.2 Generalization ---")
            revealer.run_length_generalization_experiment(quick=args.quick)

        if run_all_potential or args.reasoning:
            print("--- 2.3 Reasoning ---")
            revealer.run_reasoning_experiment(quick=args.quick)

        if run_all_potential or args.multiquery:
            print("--- 2.4 Multi-Query ---")
            revealer.run_multi_query_experiment(quick=args.quick)

        if run_all_potential or args.noise:
            print("--- 2.5 Noise Robustness ---")
            revealer.run_noise_robustness_experiment(quick=args.quick)

        if run_all_potential or args.curriculum:
            print("--- 2.6 Curriculum Learning ---")
            revealer.run_curriculum_experiment(quick=args.quick)

        if run_all_potential or args.sensitivity:
            print("--- 2.7 Sensitivity ---")
            revealer.run_sensitivity_experiment(quick=args.quick)

        revealer.generate_potential_report()
        print(f"Potential Report generated at: {os.path.join(revealer.output_dir, 'POTENTIAL_REPORT.md')}")

    # Phase 3: Discovery
    if args.discovery:
        print("\n\033[1;35m>>> PHASE 3: SCIENTIFIC DISCOVERY <<<\033[0m")
        discovery = DiscoveryEngine(output_dir=args.output_dir)
        discovery.run_full_suite(quick=args.quick)

    print("\n\033[1;32m=== RESEARCH COMPLETE ===\033[0m")

if __name__ == "__main__":
    main()
