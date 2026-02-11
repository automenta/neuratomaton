"""
ANA Analysis Tools
Visualization for gating dynamics and thinking steps.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from ana import ANAConfig, ANAModel

def plot_gating_dynamics(info_log, save_path="gating_dynamics.png"):
    """
    Plot the evolution of gates over time.
    Args:
        info_log: List of dicts from model forward pass.
        save_path: Path to save the plot.
    """
    if not info_log:
        print("No info log to plot.")
        return

    # Extract keys
    keys = info_log[0].keys()
    data = {k: [] for k in keys}

    for entry in info_log:
        for k in keys:
            if k in entry:
                data[k].append(entry[k])

    plt.figure(figsize=(10, 6))
    for k, v in data.items():
        if k == 'avg_steps': continue # Plot separately
        plt.plot(v, label=k)

    plt.xlabel("Time Step (Token)")
    plt.ylabel("Gate Value / Probability")
    plt.title("ANA Gating Dynamics")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved gating dynamics to {save_path}")

def plot_thinking_steps(info_log, save_path="thinking_steps.png"):
    """
    Plot the average thinking steps taken per token.
    """
    if not info_log:
        return

    if 'avg_steps' not in info_log[0]:
        print("No thinking steps data found.")
        return

    steps = [entry['avg_steps'] for entry in info_log]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, marker='o', linestyle='-', color='purple')
    plt.xlabel("Time Step (Token)")
    plt.ylabel("Avg Thinking Steps")
    plt.title("Adaptive Computation: Thinking Steps per Token")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved thinking steps plot to {save_path}")

if __name__ == "__main__":
    # Test run
    print("Running dummy analysis...")
    config = ANAConfig(
        d_model=32,
        vocab_size=20,
        state_dim=32,
        track_count=2,
        max_thinking_steps=2,
        use_controller=True
    )
    model = ANAModel(config)

    # Mock input
    x = torch.randint(0, 20, (4, 15)) # Batch 4, Seq 15

    # Forward
    logits, info = model(x, return_info=True)

    # Plot
    plot_gating_dynamics(info, "test_gating.png")
    plot_thinking_steps(info, "test_thinking.png")
