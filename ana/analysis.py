
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from ana.models import ANAModel
from ana.config import ANAConfig

def analyze_gating(model, input_ids, save_path="analysis.png"):
    model.eval()

    captured_gates = []

    def hook_fn(module, input, output):
        # output is gates dict {key: tensor}
        # detach and move to cpu
        gates_cpu = {k: v.detach().cpu() for k, v in output.items()}
        captured_gates.append(gates_cpu)

    # Register hook on first layer controller
    handle = None
    if model.config.use_controller:
        handle = model.layers[0].controller.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_ids)

    if handle:
        handle.remove()

    # Process captured gates
    if not captured_gates:
        print("No gates captured (Controller disabled?)")
        return

    # Check if we have one big tensor (Parallel) or list of small tensors (Sequential)
    first_gate = captured_gates[0]
    is_parallel = first_gate['alpha_0'].dim() == 3 # [batch, seq, 1]

    aggregated = {}

    if is_parallel:
        # Just one dict, but maybe captured multiple times if multiple layers?
        # We attached to layer[0], so it runs once per forward pass.
        # captured_gates[0] is the full sequence.
        aggregated = captured_gates[0]
        # Remove batch dim -> [seq, 1]
        for k in aggregated:
            aggregated[k] = aggregated[k][0]
    else:
        # Sequential: captured_gates is list of length seq_len
        # Each dict has [batch, 1]
        # Concatenate over time
        keys = first_gate.keys()
        for k in keys:
            # stack [batch, 1] -> [seq, batch, 1] -> [seq, 1] (batch=0)
            tensors = [step[k][0] for step in captured_gates]
            aggregated[k] = torch.stack(tensors, dim=0)

    # Plotting
    # We want to plot alpha_0, beta_0, alpha_1, beta_1, ret
    keys = sorted(aggregated.keys())
    num_plots = len(keys)

    plt.figure(figsize=(10, 2 * num_plots))

    for i, k in enumerate(keys):
        plt.subplot(num_plots, 1, i+1)
        data = aggregated[k].numpy() # [seq, 1]

        # If data is sigmoid (0-1), plot directly.
        # Controller returns raw linear outputs?
        # Let's check HyperController.
        # It returns raw output.
        # But we want to see the GATE value (sigmoid).
        # We should apply sigmoid for visualization.

        plt.plot(1.0 / (1.0 + np.exp(-data)), label=k) # Sigmoid
        plt.legend(loc='upper right')
        plt.ylim(-0.1, 1.1)
        plt.title(f"Gate Dynamics: {k}")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Analysis saved to {save_path}")

def main():
    # Helper to run analysis on a dummy model
    config = ANAConfig(d_model=32, state_dim=16, num_layers=2, use_parallel_scan=True)
    model = ANAModel(config)

    # Dummy input
    seq_len = 50
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    if not os.path.exists("archive/analysis"):
        os.makedirs("archive/analysis")

    analyze_gating(model, input_ids, "archive/analysis/dummy_analysis.png")

if __name__ == "__main__":
    main()
