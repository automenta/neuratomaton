#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

import torch
from ana.bio_ana import get_bio_config
from ana.bio_training import (
    BioANATrainer,
    create_curriculum_dataloader,
    CurriculumStage,
    create_scheduler,
    SchedulerConfig,
)


def train_stage(
    trainer: BioANATrainer,
    stage: str,
    num_epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    eval_every: int = 100,
    checkpoint_dir: Path = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Training Stage {stage}")
    print(f"{'='*60}")
    
    trainer.current_stage = stage
    
    train_loader = create_curriculum_dataloader(
        stage=stage,
        batch_size=batch_size,
        num_samples=steps_per_epoch * batch_size,
        seed=42 + int(stage),
    )
    
    val_loader = create_curriculum_dataloader(
        stage=stage,
        batch_size=batch_size,
        num_samples=steps_per_epoch * batch_size // 5,
        seed=123 + int(stage),
    )
    
    best_accuracy = 0.0
    epoch_metrics = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_iterations = 0.0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(trainer.device)
            target_ids = target_ids.to(trainer.device)
            
            metrics = trainer.train_step(input_ids, target_ids)
            
            epoch_loss += metrics['loss']
            epoch_accuracy += metrics['accuracy']
            epoch_iterations += metrics['avg_iterations']
            num_batches += 1
            
            if (batch_idx + 1) % eval_every == 0:
                val_metrics = trainer.evaluate(val_loader, max_batches=10)
                print(
                    f"  Step {trainer.global_step}: "
                    f"train_loss={metrics['loss']:.4f}, "
                    f"train_acc={metrics['accuracy']:.2%}, "
                    f"val_acc={val_metrics['accuracy']:.2%}, "
                    f"iters={metrics['avg_iterations']:.1f}"
                )
                
                if val_metrics['accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                    if checkpoint_dir:
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        trainer.save_checkpoint(
                            checkpoint_dir / f"stage{stage}_best.pt"
                        )
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_iterations = epoch_iterations / num_batches
        epoch_time = time.time() - epoch_start
        
        val_metrics = trainer.evaluate(val_loader)
        
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"loss={avg_loss:.4f}, "
            f"train_acc={avg_accuracy:.2%}, "
            f"val_acc={val_metrics['accuracy']:.2%}, "
            f"time={epoch_time:.1f}s"
        )
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'val_accuracy': val_metrics['accuracy'],
            'avg_iterations': avg_iterations,
            'time_s': epoch_time,
        })
        
        if trainer.check_stage_advancement(val_metrics['accuracy']):
            print(f"  ✓ Stage {stage} completed! Advancing...")
            break
    
    return {
        'stage': stage,
        'epochs_trained': len(epoch_metrics),
        'best_accuracy': best_accuracy,
        'final_val_accuracy': val_metrics['accuracy'],
        'epoch_metrics': epoch_metrics,
    }


def run_experiment(
    variant: str = 'nano',
    stages: str = '0',
    num_epochs: int = 10,
    steps_per_epoch: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    eval_every: int = 50,
    output_dir: Path = None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    
    output_dir = output_dir or Path("results/experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Bio-ANA Training Experiment")
    print(f"{'='*60}")
    print(f"Variant: {variant}")
    print(f"Stages: {stages}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output: {output_dir}")
    
    config = get_bio_config(variant)
    print(f"\nModel config:")
    print(f"  d_model: {config.d_model}")
    print(f"  relaxation_iterations: {config.relaxation_iterations}")
    
    trainer = BioANATrainer(
        config=config,
        learning_rate=learning_rate,
    )
    
    print(f"\nTrainer stats:")
    stats = trainer.get_model_stats()
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Device: {stats['device']}")
    
    all_results = {
        'variant': variant,
        'stages': stages,
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'steps_per_epoch': steps_per_epoch,
            'seed': seed,
        },
        'stage_results': [],
    }
    
    checkpoint_dir = output_dir / "checkpoints"
    
    for stage in stages.split(','):
        stage_result = train_stage(
            trainer=trainer,
            stage=stage.strip(),
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            eval_every=eval_every,
            checkpoint_dir=checkpoint_dir,
        )
        all_results['stage_results'].append(stage_result)
    
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    for sr in all_results['stage_results']:
        print(f"Stage {sr['stage']}: {sr['final_val_accuracy']:.2%} accuracy")
    print(f"\nResults saved to: {results_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Bio-ANA Training Experiments')
    parser.add_argument('--variant', type=str, default='nano',
                        choices=['nano', 'small', 'base', 'large'],
                        help='Model variant')
    parser.add_argument('--stages', type=str, default='0',
                        help='Comma-separated stages (0, 1, 2)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per stage')
    parser.add_argument('--steps', type=int, default=100,
                        help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=50,
                        help='Evaluate every N steps')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    run_experiment(
        variant=args.variant,
        stages=args.stages,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        output_dir=Path(args.output) if args.output else None,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
