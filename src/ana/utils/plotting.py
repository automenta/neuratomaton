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
            ax.plot(rg, label="Retrieval Gate", color="blue", alpha=0.8)
            has_data = True

        if 'halt_gate' in l_stats and len(l_stats['halt_gate']) > 0:
            hg = l_stats['halt_gate'][0].squeeze().float().numpy() # [Seq]
            ax.plot(hg, label="Halt Gate", color="red", linestyle="--", alpha=0.8)
            has_data = True

        if has_data:
            ax.set_ylabel(f"Layer {i}\nGate Value")
            ax.set_ylim(0, 1.1)
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

def plot_all(info_log: Dict, output_dir: str, prefix: str = ""):
    """
    Wrapper to plot all available visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_track_mixing(info_log,
                      save_path=os.path.join(output_dir, f"{prefix}track_mixing.png"))

    plot_gating_dynamics(info_log,
                         save_path=os.path.join(output_dir, f"{prefix}gating.png"))
