#!/usr/bin/env python3
"""
ANA v2 Experiment Runner

Usage:
    python run_experiment_v2.py --stage 0 --epochs 20
    python run_experiment_v2.py --stage full --epochs 30
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ana.config_v2 import ANAv2Config, Trainingv2Config, Datav2Config
from ana.training_v2 import TrainerV2


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ANA v2 Training Runner")
    parser.add_argument("--stage", type=str, default="0", 
                       choices=["0", "1", "2", "full"], 
                       help="Training stage to run")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--vocab-size", type=int, default=50, help="Vocabulary size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--stack-depth", type=int, default=5, help="Stack max depth")
    parser.add_argument("--output-dir", type=str, default="archive/results_v2", help="Output directory")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ANA V2 EXPERIMENT RUNNER")
    print("="*60)
    
    config = ANAv2Config(
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        stack_depth=args.stack_depth,
    )
    
    train_config = Trainingv2Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        stage=args.stage
    )
    
    data_config = Datav2Config(
        vocab_size=args.vocab_size
    )
    
    trainer = TrainerV2(config, train_config, data_config)
    
    print(f"\nConfiguration:")
    print(f"  Model dim: {config.d_model}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Stack depth: {config.stack_depth}")
    print(f"  Total track dim: {config.total_track_dim}")
    print(f"  Device: {trainer.get_device()}")
    
    if args.stage == "full":
        results = trainer.run_full_curriculum()
    elif args.stage == "0":
        results = trainer.run_stage0()
    elif args.stage == "1":
        results = trainer.run_stage1()
    elif args.stage == "2":
        results = trainer.run_stage2()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
