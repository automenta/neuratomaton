#!/usr/bin/env python
import argparse
import sys
import os
import traceback
import logging
import torch
from torch.utils.data import DataLoader

# Import research framework to register experiments
import ana.research
from ana.research.core import ExperimentRegistry, load_config_overrides
from ana.models.config import ANAConfig
from ana.models.core import ANAModel
from ana.training.utils import Trainer
from ana.utils.datasets import TASK_REGISTRY, HuggingFaceDataset

def train_command(args):
    """
    Handler for the 'train' subcommand.
    """
    print("="*60)
    print("ANA TRAINING CLI")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print("="*60)

    # 1. Configuration
    config = ANAConfig()

    # Apply overrides
    if args.overrides:
        config = load_config_overrides(config, args.overrides)
        print(f"Applied config overrides: {args.overrides}")

    # 2. Dataset
    print(f"Loading dataset '{args.dataset}'...")
    if args.dataset in TASK_REGISTRY:
        # Synthetic tasks
        if args.dataset == 'huggingface':
             # Special case for HF
             if not args.hf_dataset:
                 print("Error: --hf_dataset required for huggingface dataset type.")
                 sys.exit(1)
             dataset = HuggingFaceDataset(args.hf_dataset, seq_len=config.max_position) # Or specific seq_len
             config.vocab_size = dataset.vocab_size
        else:
             dataset_cls = TASK_REGISTRY[args.dataset]
             # Instantiate with some default args or what?
             # Most synthetic tasks take num_samples
             # We might need to map args to dataset init
             try:
                 dataset = dataset_cls(num_samples=1000, vocab_size=config.vocab_size)
             except TypeError:
                 # Fallback for datasets with different init
                 dataset = dataset_cls()
    else:
        print(f"Error: Dataset '{args.dataset}' not found in registry.")
        print(f"Available datasets: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Dataset loaded. Size: {len(dataset)}")

    # Update config vocab_size if dataset exposes it
    if hasattr(dataset, "vocab_size"):
        config.vocab_size = dataset.vocab_size
        print(f"Updated vocab_size to {config.vocab_size} from dataset.")
    elif hasattr(dataset, "get_vocab_size"):
        config.vocab_size = dataset.get_vocab_size()
        print(f"Updated vocab_size to {config.vocab_size} from dataset.")

    # 3. Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    model = ANAModel(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # 4. Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=10
    )

    print("Starting training...")
    try:
        trainer.fit(train_loader, epochs=config.epochs)
        print("Training complete.")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ANA Automated Research Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Research Command (Default/Legacy)
    research_parser = subparsers.add_parser("research", help="Run automated research experiments")
    research_parser.add_argument("--quick", action="store_true", help="Run in fast smoketest mode")
    research_parser.add_argument("--tune", action="store_true", help="Force hyperparameter tuning stage")
    research_parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    research_parser.add_argument("--study_name", type=str, default="main", help="Name of the study")
    research_parser.add_argument("--validation", action="store_true", help="Run Phase 1: Validation")
    research_parser.add_argument("--potential", action="store_true", help="Run Phase 2: Potential")
    research_parser.add_argument("--discovery", action="store_true", help="Run Phase 3: Discovery")
    research_parser.add_argument("--action", action="store_true", help="Run Phase 4: Action")
    research_parser.add_argument("--series", action="store_true", help="Run Phase 5: Series")
    research_parser.add_argument("--all", action="store_true", help="Run all phases")

    # Potential sub-flags
    research_parser.add_argument("--induction", action="store_true")
    research_parser.add_argument("--generalization", action="store_true")
    research_parser.add_argument("--multiquery", action="store_true")
    research_parser.add_argument("--reasoning", action="store_true")
    research_parser.add_argument("--noise", action="store_true")
    research_parser.add_argument("--curriculum", action="store_true")
    research_parser.add_argument("--sensitivity", action="store_true")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train ANA model on a dataset")
    train_parser.add_argument("--dataset", type=str, required=True, help="Dataset name (from registry or 'huggingface')")
    train_parser.add_argument("--hf_dataset", type=str, help="Hugging Face dataset path (if dataset='huggingface')")
    train_parser.add_argument("--overrides", type=str, help="Config overrides (e.g. 'd_model=128,epochs=10')")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    # Parse args
    # Note: If no subcommand is provided, we might want to default to 'research' for backward compatibility,
    # but argparse usually requires explicit subcommand if subparsers are defined.
    # To maintain backward compat: check sys.argv.

    # Hack for backward compatibility: if first arg is not a command, insert 'research'
    if len(sys.argv) > 1 and sys.argv[1] not in ['research', 'train', '-h', '--help']:
        sys.argv.insert(1, 'research')
    elif len(sys.argv) == 1:
        sys.argv.append('--help')

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "research":
        # Check potential sub-flags
        potential_flags = ["induction", "generalization", "multiquery", "reasoning", "noise", "curriculum", "sensitivity"]
        active_potential_flags = [flag for flag in potential_flags if getattr(args, flag)]

        if active_potential_flags:
            args.potential = True

        if args.all:
            args.validation = True
            args.potential = True
            args.discovery = True
            args.action = True
            args.series = True

        if not (args.validation or args.potential or args.discovery or args.action or args.series):
            print("No phase specified. Defaulting to Phase 3: Discovery.")
            args.discovery = True

        print("="*60)
        print("ANA AUTOMATED RESEARCHER")
        print(f"Study Name: {args.study_name}")
        print(f"Quick Mode: {args.quick}")
        print("="*60)

        try:
            if args.validation:
                print("\n>>> PHASE 1: VALIDATION <<<")
                exp_cls = ExperimentRegistry.get(1, "validation")
                if exp_cls: exp_cls().run(study_name=args.study_name, quick=args.quick, tune=args.tune, trials=args.trials)

            if args.potential:
                print("\n>>> PHASE 2: POTENTIAL <<<")
                exp_cls = ExperimentRegistry.get(2, "potential")
                if exp_cls: exp_cls().run(study_name=args.study_name, quick=args.quick, sub_experiments=active_potential_flags)

            if args.discovery:
                print("\n>>> PHASE 3: DISCOVERY <<<")
                exp_cls = ExperimentRegistry.get(3, "discovery")
                if exp_cls: exp_cls().run(study_name=args.study_name, quick=args.quick)

            if args.action:
                print("\n>>> PHASE 4: ACTION <<<")
                exp_cls = ExperimentRegistry.get(4, "action")
                if exp_cls: exp_cls().run(study_name=args.study_name, quick=args.quick)

            if args.series:
                print("\n>>> PHASE 5: SERIES <<<")
                exp_cls = ExperimentRegistry.get(5, "series")
                if exp_cls: exp_cls().run(study_name=args.study_name, quick=args.quick)

            print("\n=== RESEARCH COMPLETE ===")

        except Exception as e:
            print(f"\n[ERROR] Research failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
