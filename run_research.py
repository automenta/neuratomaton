import argparse
import sys
import os
from ana.config import ANAConfig
from ana.research.core import ExperimentRegistry, load_config_overrides

# Ensure ana package is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="ANA Research Agenda Execution Framework")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], help="Research Phase (1-6)")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (Phase 2)")
    parser.add_argument("--config", type=str, default="", help="Config overrides (e.g. 'd_model=128,dropout=0.1')")

    args = parser.parse_args()

    # Pre-load modules to ensure registration
    try:
        if args.phase == 1:
            import ana.research.phase1_validation.benchmarks
            import ana.research.phase1_validation.scaling
            import ana.research.phase1_validation.baseline_comparison
        elif args.phase == 2:
            import ana.research.phase2_text.long_context
            import ana.research.phase2_text.inference
        elif args.phase == 3:
            import ana.research.phase3_vision.train
            import ana.research.phase3_vision.captioning
        elif args.phase == 4:
            import ana.research.phase4_rl.train
        elif args.phase == 5:
            import ana.research.phase5_specialized.train
        elif args.phase == 6:
            import ana.research.deployment.export
    except ImportError as e:
        print(f"Warning: Could not import experiments for Phase {args.phase}: {e}")

    # Check Registry
    experiment_cls = ExperimentRegistry.get(args.phase, args.experiment)

    if experiment_cls:
        print(f"Running Experiment via Registry: {args.experiment} (Phase {args.phase})")

        # Default config
        config = ANAConfig()
        if args.device == "cuda":
            config.device = "cuda"

        # Apply overrides
        config = load_config_overrides(config, args.config)

        experiment = experiment_cls(config, device=args.device)
        experiment.run()
        return

    # Fallback/Help
    print(f"Experiment '{args.experiment}' not found for Phase {args.phase}.")
    available = ExperimentRegistry.list_experiments().get(args.phase, {})
    if available:
        print(f"Available experiments: {', '.join(available.keys())}")
    else:
        print("No experiments registered for this phase.")

if __name__ == "__main__":
    main()
