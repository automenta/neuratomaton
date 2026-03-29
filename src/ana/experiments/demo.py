import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
import argparse
import numpy as np

# Add src to path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ana.experiments.comprehensive import ComparisonRunner
from ana.models.config import ANAConfig
from ana.models.core import ANAModel
from ana.utils.datasets import AssociativeRecallDataset
from ana.utils.plotting import plot_all

def run_live_demo(steps: int = 500, output_dir: str = "results/demo"):
    """
    Runs a 'Fast & Rich' experiment designed for immediate feedback.
    Target: Learn Associative Recall (N=4) quickly and visualize internal dynamics.
    """
    print("="*60)
    print("ANA FAST BREAKTHROUGH DEMO")
    print("Goal: Watch the model learn to recall associations in real-time.")
    print("="*60)

    # Setup runner
    runner = ComparisonRunner(output_dir=output_dir)
    plots_dir = os.path.join(runner.output_dir, "plots")
    latest_dir = os.path.join(runner.output_dir, "latest")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    # Task: Associative Recall (Small scale for speed)
    # 4 pairs, vocab 40, noise 32.
    # This requires memory search.
    task = AssociativeRecallDataset(num_samples=2000, vocab_size=40, num_pairs=4, noise_len=32)
    train_loader = DataLoader(task, batch_size=16, shuffle=True)
    val_loader = DataLoader(task, batch_size=16, shuffle=False)

    # Model: Small but feature-complete
    config = ANAConfig(
        vocab_size=40,
        d_model=64,
        state_dim=64,
        key_dim=32,
        num_layers=2,
        track_count=2,
        use_hololink=True,
        use_controller=True,
        use_parallel_scan=True
    )

    model = ANAModel(config)
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Validation helper
    def validate(step):
        loss, acc = runner.evaluate_model(model, val_loader)
        return acc, loss

    # Plotting helper
    def generate_plots(step):
        model.eval()
        # Get one sample
        x, y, mask = next(iter(val_loader))
        x = x.to(runner.device)

        with torch.no_grad():
            _, info_log = model(x, return_info=True)

        # Save to history
        step_dir = os.path.join(plots_dir, f"step_{step:04d}")
        os.makedirs(step_dir, exist_ok=True)
        plot_all(info_log, step_dir, prefix="")

        # Save to latest (overwrite) for "Live View"
        plot_all(info_log, latest_dir, prefix="")
        print(f"[Step {step}] Plots updated in {latest_dir} (and saved to {step_dir})")

    # Callback
    history = {'step': [], 'loss': [], 'acc': []}

    def training_callback(step, current_loss, model_ref):
        # 1. Validation every 20 steps
        acc = 0.0
        if step > 0 and step % 20 == 0:
            acc, val_loss = validate(step)
            history['step'].append(step)
            history['acc'].append(acc)

            # Clear screen and print dashboard
            os.system('cls' if os.name == 'nt' else 'clear')
            print("="*60)
            print("ANA FAST BREAKTHROUGH DEMO - LIVE DASHBOARD")
            print("="*60)
            print(f"Step: {step}/{steps}")
            print(f"Current Loss: {current_loss:.4f}")
            print(f"Validation Acc: {acc*100:.1f}%")

            # Visual Progress Bar
            bar_len = 30
            filled = int(bar_len * acc)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"Progress: [{bar}]")

            # Insight triggers
            if acc > 0.95:
                print(f"\n\033[1;32m>>> BREAKTHROUGH: Model has mastered the task! <<<\033[0m")
            elif acc > 0.5:
                print(f"\n\033[1;33m>>> PROGRESS: Model is learning associations. <<<\033[0m")
            elif acc < 0.1:
                print(f"\n>>> STATUS: Model is still exploring randomly. <<<")

        # 2. Plotting every 50 steps (or on breakthrough)
        if step > 0 and step % 50 == 0:
            generate_plots(step)
            print(f"\nPlots updated in: {latest_dir}")

    # Run Training
    print("\nStarting Training... (Open 'results/demo/latest' to watch plots update)")
    runner.train_model(model, train_loader, max_steps=steps, lr=2e-3, callback=training_callback)

    # Final eval
    final_acc = validate(steps)
    generate_plots(steps)

    print("="*60)
    print(f"DEMO COMPLETE. Final Accuracy: {final_acc*100:.1f}%")
    print(f"Results saved to: {runner.output_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Number of steps")
    args = parser.parse_args()

    run_live_demo(steps=args.steps)
