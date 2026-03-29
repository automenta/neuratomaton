import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
from typing import Dict, List, Optional

def plot_track_mixing(info_log: Dict,
                      save_path: Optional[str] = None,
                      title: str = "Track Mixing Dynamics"):
    """
    Plots the mixing weights for each layer and track over sequence length.
    Expects info_log['layers'][i]['mix_weights'] to be [Batch, Seq, Tracks].
    Plots the FIRST sample in the batch.
    """
    layers = info_log['layers']
    num_layers = len(layers)

    # Check if mix_weights exists and has data
    if 'mix_weights' not in layers[0] or len(layers[0]['mix_weights']) == 0:
        print("No mix_weights found in info_log")
        return

    # Extract data for first batch element
    # Data shape: [Seq, Tracks] per layer
    layer_data = []
    for l_idx, l_stats in enumerate(layers):
        if 'mix_weights' in l_stats and len(l_stats['mix_weights']) > 0:
            mw = l_stats['mix_weights'][0].float().numpy() # [Seq, Tracks]
            layer_data.append(mw)

    if not layer_data:
        return

    seq_len, num_tracks = layer_data[0].shape

    # Create subplots: one per layer
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 3 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    for i, data in enumerate(layer_data):
        ax = axes[i]
        # Transpose to [Tracks, Seq] for heatmap where X is Time
        sns.heatmap(data.T, ax=ax, cmap="viridis", vmin=0, vmax=1, cbar=True)
        ax.set_ylabel(f"Layer {i}\nTracks")
        ax.set_yticks(np.arange(num_tracks) + 0.5)
        ax.set_yticklabels([f"T{t}" for t in range(num_tracks)], rotation=0)
        ax.set_title(f"Layer {i} Mixing Weights")

    axes[-1].set_xlabel("Sequence Position")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_gating_dynamics(info_log: Dict,
                         save_path: Optional[str] = None,
                         title: str = "Gating Dynamics"):
    """
    Plots Retrieval Gate and Halt Gate (if present) over time.
    Plots the FIRST sample in the batch.
    """
    layers = info_log['layers']
    num_layers = len(layers)

    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 3 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    for i, l_stats in enumerate(layers):
        ax = axes[i]

        has_data = False
        if 'ret_gate' in l_stats and len(l_stats['ret_gate']) > 0:
            rg = l_stats['ret_gate'][0].squeeze().float().numpy() # [Seq]
            ax.plot(rg, label="Retrieval Gate", color="blue", linewidth=2, alpha=0.8)
            has_data = True

        if 'halt_gate' in l_stats and len(l_stats['halt_gate']) > 0:
            hg = l_stats['halt_gate'][0].squeeze().float().numpy() # [Seq]
            ax.plot(hg, label="Halt Gate", color="red", linestyle="--", linewidth=2, alpha=0.8)
            # Add threshold line
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label="Halt Threshold")
            has_data = True

        if has_data:
            ax.set_ylabel(f"Layer {i}\nGate Value")
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Gating Data", ha='center', va='center')

    axes[-1].set_xlabel("Sequence Position")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_cognitive_state(info_log: Dict,
                         save_path: Optional[str] = None,
                         title: str = "Cognitive State (Track vs Memory)"):
    """
    Plots a stacked area chart showing the relative contribution of:
    - Each Track (scaled by 1 - ret_gate)
    - Memory Retrieval (ret_gate)
    """
    layers = info_log['layers']
    num_layers = len(layers)

    # Check if we have data
    if 'mix_weights' not in layers[0] or len(layers[0]['mix_weights']) == 0:
        # Cannot plot cognitive state without mix weights
        return

    # Create subplots
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    for i, l_stats in enumerate(layers):
        ax = axes[i]

        # Get mix weights [Seq, Tracks]
        # Ensure we have data
        if 'mix_weights' not in l_stats or len(l_stats['mix_weights']) == 0:
             continue

        mw = l_stats['mix_weights'][0].float().numpy()
        if mw.ndim == 1: # Handle edge case
             mw = mw[:, None]

        seq_len, num_tracks = mw.shape

        # Get retrieval gate [Seq]
        if 'ret_gate' in l_stats and len(l_stats['ret_gate']) > 0:
            rg = l_stats['ret_gate'][0].squeeze().float().numpy()
        else:
            rg = np.zeros(seq_len)

        # Ensure shapes match (sometimes squeezing single dim leaves scalar)
        if rg.ndim == 0: rg = np.full(seq_len, rg)
        if rg.shape[0] != seq_len:
            # Handle mismatch (truncate or pad)
            min_len = min(rg.shape[0], seq_len)
            rg = rg[:min_len]
            mw = mw[:min_len]
            seq_len = min_len

        # Calculate contributions
        # Memory = rg
        # Track k = mw[:, k] * (1 - rg)

        contributions = []
        labels = []
        colors = []

        # Tracks
        for t in range(num_tracks):
            contrib = mw[:, t] * (1 - rg)
            contributions.append(contrib)
            labels.append(f"Track {t}")
            # Use distinct colors for tracks
            colors.append(plt.cm.viridis(t / num_tracks))

        # Memory
        contributions.append(rg)
        labels.append("Memory (HoloLink)")
        colors.append("gold") # Gold for memory

        # Stackplot
        ax.stackplot(range(seq_len), contributions, labels=labels, colors=colors, alpha=0.8)

        ax.set_ylabel(f"Layer {i}\nContribution")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title(f"Layer {i} Cognitive State Allocation")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sequence Position")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_all(info_log: Dict, output_dir: str, prefix: str = ""):
    """
    Wrapper to plot all available visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_track_mixing(info_log,
                      save_path=os.path.join(output_dir, f"{prefix}track_mixing.png"))

    plot_gating_dynamics(info_log,
                         save_path=os.path.join(output_dir, f"{prefix}gating.png"))

    plot_cognitive_state(info_log,
                         save_path=os.path.join(output_dir, f"{prefix}cognitive_state.png"))
