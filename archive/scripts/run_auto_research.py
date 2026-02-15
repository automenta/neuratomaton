#!/usr/bin/env python
import argparse
import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import research framework to register experiments
import ana.research
from ana.research.core import ExperimentRegistry

def main():
    parser = argparse.ArgumentParser(description="ANA Automated Research Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run in fast smoketest mode")
    parser.add_argument("--tune", action="store_true", help="Force hyperparameter tuning stage")
    parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--study_name", type=str, default="main", help="Name of the study (accumulates results in results/<name>/)")
    parser.add_argument("--output_dir", type=str, default="results/automated", help="Output directory (deprecated, managed by framework)")

    # Phases
    parser.add_argument("--validation", action="store_true", help="Run Phase 1: Validation & Scaling")
    parser.add_argument("--potential", action="store_true", help="Run Phase 2: Potential & Capabilities")
    parser.add_argument("--discovery", action="store_true", help="Run Phase 3: Scientific Discovery (Baselines, Optuna, Ablation)")
    parser.add_argument("--action", action="store_true", help="Run Phase 4: Action & RL")
    parser.add_argument("--series", action="store_true", help="Run Phase 5: Time Series & Audio")
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

    # Check if any potential sub-flags are set, imply potential phase
    potential_flags = ["induction", "generalization", "multiquery", "reasoning", "noise", "curriculum", "sensitivity"]
    active_potential_flags = [flag for flag in potential_flags if getattr(args, flag)]

    if active_potential_flags:
        args.potential = True

    # Default logic: If no phase specified, run discovery (since that's the new enhancement)
    # Or if args.all, run all.
    if args.all:
        args.validation = True
        args.potential = True
        args.discovery = True
        args.action = True
        args.series = True

    # If nothing specified (and no implied potential), default to Discovery
    if not (args.validation or args.potential or args.discovery or args.action or args.series):
        print("No phase specified. Defaulting to Phase 3: Discovery.")
        args.discovery = True

    print("="*60)
    print("ANA AUTOMATED RESEARCHER (Turnkey Edition)")
    print("Maximally automated discovery process.")
    print(f"Study Name: {args.study_name}")
    print(f"Quick Mode: {args.quick}")
    print("="*60)

    try:
        # Phase 1: Validation
        if args.validation:
            print("\n\033[1;35m>>> PHASE 1: VALIDATION & SCALING <<<\033[0m")
            exp_cls = ExperimentRegistry.get(1, "validation")
            if not exp_cls:
                print("Error: Validation experiment not found in registry.")
            else:
                exp = exp_cls()
                exp.run(study_name=args.study_name, quick=args.quick, tune=args.tune, trials=args.trials)

        # Phase 2: Potential
        if args.potential:
            print("\n\033[1;35m>>> PHASE 2: POTENTIAL & CAPABILITIES <<<\033[0m")
            exp_cls = ExperimentRegistry.get(2, "potential")
            if not exp_cls:
                print("Error: Potential experiment not found in registry.")
            else:
                exp = exp_cls()
                # Pass active sub-experiments if any
                exp.run(study_name=args.study_name, quick=args.quick, sub_experiments=active_potential_flags)

        # Phase 3: Discovery
        if args.discovery:
            print("\n\033[1;35m>>> PHASE 3: SCIENTIFIC DISCOVERY <<<\033[0m")
            exp_cls = ExperimentRegistry.get(3, "discovery")
            if not exp_cls:
                print("Error: Discovery experiment not found in registry.")
            else:
                exp = exp_cls()
                exp.run(study_name=args.study_name, quick=args.quick)

        # Phase 4: Action
        if args.action:
            print("\n\033[1;35m>>> PHASE 4: ACTION & RL <<<\033[0m")
            exp_cls = ExperimentRegistry.get(4, "action")
            if not exp_cls:
                print("Error: Action experiment not found in registry.")
            else:
                exp = exp_cls()
                exp.run(study_name=args.study_name, quick=args.quick)

        # Phase 5: Series
        if args.series:
            print("\n\033[1;35m>>> PHASE 5: SERIES & AUDIO <<<\033[0m")
            exp_cls = ExperimentRegistry.get(5, "series")
            if not exp_cls:
                print("Error: Series experiment not found in registry.")
            else:
                exp = exp_cls()
                exp.run(study_name=args.study_name, quick=args.quick)

        print("\n\033[1;32m=== RESEARCH COMPLETE ===\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m[ERROR] Research failed: {e}\033[0m")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
