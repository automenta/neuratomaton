import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from .models import ANAModel
from .config import ANAConfig

def analyze_gating(model, input_ids, save_path="analysis.png"):
    model.eval()
    
    captured_gates = []
    
    def hook_fn(module, input, output):
        track_outputs, g_ret, g_halt = output
        gates_cpu = {
            'track_outputs': [(a.detach().cpu(), b.detach().cpu(), m.detach().cpu()) for a, b, m in track_outputs],
            'ret_gate': g_ret.detach().cpu(),
            'halt': g_halt.detach().cpu()
        }
        captured_gates.append(gates_cpu)
    
    handle = None
    if model.config.use_controller:
        handle = model.layers[0]['controller'].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(input_ids)
    
    if handle:
        handle.remove()
    
    if not captured_gates:
        print("No gates captured (Controller disabled?)")
        return
    
    is_parallel = captured_gates[0]['ret_gate'].dim() == 3
    
    aggregated = {}
    
    if is_parallel:
        first_gate = captured_gates[0]
        for t_idx, (alpha, beta, mix) in enumerate(first_gate['track_outputs']):
            aggregated[f'alpha_{t_idx}'] = alpha[0].numpy()
            aggregated[f'beta_{t_idx}'] = beta[0].numpy()
            aggregated[f'mix_{t_idx}'] = mix[0].numpy()
        aggregated['ret_gate'] = first_gate['ret_gate'][0].numpy()
        aggregated['halt'] = first_gate['halt'][0].numpy()
    else:
        keys_template = ['alpha_0', 'beta_0', 'mix_0', 'ret_gate', 'halt']
        for k in keys_template:
            aggregated[k] = []
        
        for gate_dict in captured_gates:
            for t_idx, (alpha, beta, mix) in enumerate(gate_dict['track_outputs']):
                if f'alpha_{t_idx}' not in aggregated:
                    aggregated[f'alpha_{t_idx}'] = []
                    aggregated[f'beta_{t_idx}'] = []
                    aggregated[f'mix_{t_idx}'] = []
                aggregated[f'alpha_{t_idx}'].append(alpha[0].numpy())
                aggregated[f'beta_{t_idx}'].append(beta[0].numpy())
                aggregated[f'mix_{t_idx}'].append(mix[0].numpy())
            aggregated['ret_gate'].append(gate_dict['ret_gate'][0].numpy())
            aggregated['halt'].append(gate_dict['halt'][0].numpy())
        
        for k in aggregated:
            aggregated[k] = np.stack(aggregated[k], axis=0)
    
    keys = sorted(aggregated.keys())
    num_plots = len(keys)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    for i, k in enumerate(keys):
        data = aggregated[k]
        if data.ndim == 2:
            data = data.mean(axis=-1)
        
        sigmoid_data = 1.0 / (1.0 + np.exp(-data))
        axes[i].plot(sigmoid_data, label=k)
        axes[i].legend(loc='upper right')
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_title(f"Gate Dynamics: {k}")
        axes[i].set_xlabel("Sequence Position")
        axes[i].set_ylabel("Gate Value (sigmoid)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Analysis saved to {save_path}")

def analyze_attention_pattern(model, input_ids, save_path="attention.png"):
    model.eval()
    
    captured_memory = []
    
    def hook_holo(module, input, output):
        _, M = output
        captured_memory.append(M.detach().cpu())
    
    handles = []
    if model.config.use_hololink:
        for i, layer in enumerate(model.layers):
            if 'holo' in layer:
                handles.append(layer['holo'].register_forward_hook(hook_holo))
    
    with torch.no_grad():
        model(input_ids)
    
    for h in handles:
        h.remove()
    
    if not captured_memory:
        print("No memory captured (HoloLink disabled?)")
        return
    
    M = captured_memory[0][0]
    key_dim, d_model = M.shape
    seq_len = input_ids.shape[1]
    
    if seq_len > key_dim:
        attention = M[:seq_len].numpy()
    else:
        attention = M[:seq_len].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(attention.T, aspect='auto', cmap='viridis')
    axes[0].set_title("HoloLink Memory State")
    axes[0].set_xlabel("Sequence Position")
    axes[0].set_ylabel("Key Dimension")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(np.abs(attention.T), aspect='auto', cmap='hot')
    axes[1].set_title("HoloLink Memory Magnitude")
    axes[1].set_xlabel("Sequence Position")
    axes[1].set_ylabel("Key Dimension")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Attention pattern saved to {save_path}")

def main():
    config = ANAConfig(d_model=32, state_dim=16, num_layers=2, use_parallel_scan=True)
    model = ANAModel(config)
    
    seq_len = 50
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    
    os.makedirs("archive/analysis", exist_ok=True)
    
    analyze_gating(model, input_ids, "archive/analysis/gate_dynamics.png")
    analyze_attention_pattern(model, input_ids, "archive/analysis/attention_pattern.png")

if __name__ == "__main__":
    main()
