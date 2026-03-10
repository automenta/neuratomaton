"""
Interpretability tools for ANA models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import umap
import umap.plot


class ModelInterpretability:
    """
    Interpretability tools for understanding ANA model behavior
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.activations = {}
        self.hooks = []
    
    def register_activation_hooks(self):
        """
        Register hooks to capture activations during forward pass
        """
        # Clear any existing hooks
        self.remove_hooks()
        
        def get_activation(name):
            def hook(model, input, output):
                # Store the input to the layer as the activation
                if isinstance(input, tuple) and len(input) > 0:
                    self.activations[name] = input[0].detach().cpu()
                else:
                    self.activations[name] = (input if torch.is_tensor(input) else output).detach().cpu()
            return hook
        
        # Register hooks for key layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.SiLU)):
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append((name, handle))
    
    def remove_hooks(self):
        """
        Remove all registered hooks
        """
        for _, handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def analyze_hololink_memory_patterns(self, input_sequences: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze HoloLink memory patterns and associations
        """
        self.model.eval()
        self.register_activation_hooks()
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                _ = self.model(input_sequences)
            else:
                _ = self.model(input_sequences)
        
        # Look for HoloLink-related activations
        hololink_activations = {}
        for name, activation in self.activations.items():
            if 'holo' in name.lower():
                hololink_activations[name] = activation
        
        # Analyze memory patterns
        memory_analysis = {}
        for name, activation in hololink_activations.items():
            # Compute statistics about the activation patterns
            mean_act = activation.mean().item()
            std_act = activation.std().item()
            max_act = activation.max().item()
            min_act = activation.min().item()
            
            memory_analysis[name] = {
                'mean_activation': mean_act,
                'std_activation': std_act,
                'max_activation': max_act,
                'min_activation': min_act,
                'shape': activation.shape
            }
        
        self.remove_hooks()
        return memory_analysis
    
    def analyze_attention_patterns(self, input_sequences: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze attention-like patterns in the model
        """
        # For ANA, we'll analyze how the HoloLink component attends to different parts
        self.model.eval()
        self.register_activation_hooks()
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                logits, info = self.model(input_sequences)
            else:
                logits = self.model(input_sequences)
        
        # Analyze the relationship between different layer activations
        attention_analysis = {}
        
        # Compute correlations between layer activations
        layer_names = list(self.activations.keys())
        if len(layer_names) > 1:
            for i, name1 in enumerate(layer_names):
                for j, name2 in enumerate(layer_names):
                    if i < j:  # Only compute upper triangle
                        act1 = self.activations[name1]
                        act2 = self.activations[name2]
                        
                        # Flatten activations for correlation
                        flat_act1 = act1.view(-1).numpy()
                        flat_act2 = act2.view(-1).numpy()
                        
                        # Compute cosine similarity
                        cos_sim = cosine_similarity([flat_act1], [flat_act2])[0][0]
                        
                        pair_name = f"{name1}_vs_{name2}"
                        attention_analysis[pair_name] = {
                            'cosine_similarity': cos_sim,
                            'shape1': act1.shape,
                            'shape2': act2.shape
                        }
        
        self.remove_hooks()
        return attention_analysis
    
    def generate_feature_importance(self, input_sequences: torch.Tensor, 
                                   target_output: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate feature importance scores for different model components
        """
        # Compute gradients with respect to inputs to understand feature importance
        self.model.train()
        
        input_seq = input_sequences.clone().requires_grad_(True)
        
        if hasattr(self.model, 'forward'):
            output, _ = self.model(input_seq)
        else:
            output = self.model(input_seq)
        
        # If target output is provided, use it for loss; otherwise use output itself
        if target_output is not None:
            loss = torch.nn.functional.mse_loss(output, target_output)
        else:
            loss = output.mean()  # Just use mean as proxy
        
        # Backpropagate to get gradients
        loss.backward()
        
        # Get gradients as importance scores
        input_grads = input_seq.grad
        
        feature_importance = {
            'input_gradients': input_grads.detach().cpu().numpy(),
            'mean_abs_grad': input_grads.abs().mean().item(),
            'max_abs_grad': input_grads.abs().max().item(),
            'grad_shape': input_grads.shape
        }
        
        # Zero out gradients
        self.model.zero_grad()
        
        return feature_importance
    
    def visualize_interpretability_report(self, input_sequences: torch.Tensor, 
                                       save_path: Optional[str] = None):
        """
        Generate a comprehensive interpretability report
        """
        # Get all analyses
        memory_analysis = self.analyze_hololink_memory_patterns(input_sequences)
        attention_analysis = self.analyze_attention_patterns(input_sequences)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Memory activation patterns
        if memory_analysis:
            names = list(memory_analysis.keys())
            means = [memory_analysis[name]['mean_activation'] for name in names]
            stds = [memory_analysis[name]['std_activation'] for name in names]
            
            axes[0, 0].bar(range(len(names)), means, yerr=stds, capsize=5, alpha=0.7)
            axes[0, 0].set_title('HoloLink Memory Activation Patterns')
            axes[0, 0].set_xlabel('Layer Name')
            axes[0, 0].set_ylabel('Activation')
            axes[0, 0].set_xticks(range(len(names)))
            axes[0, 0].set_xticklabels([name.split('.')[-1][:15] for name in names], rotation=45)
        
        # Plot 2: Attention correlations
        if attention_analysis:
            pairs = list(attention_analysis.keys())[:10]  # Limit to first 10 for readability
            sims = [attention_analysis[pair]['cosine_similarity'] for pair in pairs]
            
            axes[0, 1].bar(range(len(pairs)), sims, alpha=0.7)
            axes[0, 1].set_title('Activation Correlations')
            axes[0, 1].set_xlabel('Layer Pairs')
            axes[0, 1].set_ylabel('Cosine Similarity')
            axes[0, 1].set_xticks(range(len(pairs)))
            axes[0, 1].set_xticklabels([pair.split('_vs_')[0][-10:] for pair in pairs], rotation=45)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 3: Feature importance (if we can compute it)
        try:
            feature_imp = self.generate_feature_importance(input_sequences)
            grad_data = feature_imp['input_gradients']
            
            # Take mean across all dims except the last one (features)
            if grad_data.ndim > 1:
                mean_grads = np.mean(np.abs(grad_data), axis=tuple(range(grad_data.ndim-1)))
                axes[1, 0].bar(range(len(mean_grads)), mean_grads, alpha=0.7)
                axes[1, 0].set_title('Feature Importance (Gradient Magnitude)')
                axes[1, 0].set_xlabel('Feature Index')
                axes[1, 0].set_ylabel('Mean Absolute Gradient')
        except:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\\ncalculation failed', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')
        
        # Plot 4: Summary statistics
        summary_data = []
        summary_labels = []
        
        if memory_analysis:
            for name, stats in list(memory_analysis.items())[:5]:  # Limit to first 5
                summary_data.extend([stats['mean_activation'], stats['std_activation']])
                summary_labels.extend([f'{name.split(\".\")[-1]}_mean', f'{name.split(\".\")[-1]}_std'])
        
        if summary_data:
            axes[1, 1].bar(range(len(summary_data)), summary_data, alpha=0.7)
            axes[1, 1].set_title('Summary Statistics')
            axes[1, 1].set_xlabel('Metric')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_xticks(range(len(summary_labels)))
            axes[1, 1].set_xticklabels(summary_labels, rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No data to display', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return {
            'memory_analysis': memory_analysis,
            'attention_analysis': attention_analysis,
            'visualization_saved': save_path is not None
        }


class LayerWiseRelevancePropagation:
    """
    Layer-wise relevance propagation for ANA models
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def propagate_relevance(self, input_tensor: torch.Tensor, 
                          target_class: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Propagate relevance backwards through the model layers
        """
        # Forward pass to get activations
        self.model.eval()
        
        # Store intermediate outputs
        stored_outputs = {}
        
        def store_output(name):
            def hook(module, input, output):
                stored_outputs[name] = output.detach().clone()
            return hook
        
        # Register hooks
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                handle = module.register_forward_hook(store_output(name))
                handles.append(handle)
        
        # Forward pass
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            if hasattr(self.model, 'forward'):
                output, _ = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
            
            # If no target class specified, use the predicted class
            if target_class is None:
                target_class = output.argmax(-1).item()
            
            # Zero out all but the target class
            output_single = torch.zeros_like(output)
            output_single[0, target_class] = output[0, target_class]
            
            # Backward pass to get gradients
            output_single.backward(retain_graph=True)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Compute relevance scores based on gradients
        relevance_scores = {}
        for name, output in stored_outputs.items():
            if hasattr(output, 'grad') and output.grad is not None:
                relevance_scores[name] = output.grad.abs().mean(dim=-1)  # Average over last dimension
        
        # Also compute input relevance
        if input_tensor.grad is not None:
            relevance_scores['input'] = input_tensor.grad.abs().mean(dim=-1)
        
        return relevance_scores
    
    def visualize_relevance(self, relevance_scores: Dict[str, torch.Tensor], 
                          save_path: Optional[str] = None):
        """
        Visualize relevance scores
        """
        # Create a heatmap of relevance scores
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        layer_names = list(relevance_scores.keys())
        max_len = max(r.shape[-1] if r.dim() > 0 else 1 for r in relevance_scores.values())
        
        # Pad or truncate relevance scores to same length
        relevance_matrices = []
        for name in layer_names:
            rel = relevance_scores[name]
            if rel.dim() == 0:
                rel = rel.unsqueeze(0)
            if rel.shape[-1] < max_len:
                # Pad with zeros
                pad_size = max_len - rel.shape[-1]
                rel = torch.cat([rel, torch.zeros(rel.shape[:-1] + (pad_size,))], dim=-1)
            elif rel.shape[-1] > max_len:
                # Truncate
                rel = rel[..., :max_len]
            relevance_matrices.append(rel.flatten().unsqueeze(0))
        
        if relevance_matrices:
            relevance_matrix = torch.cat(relevance_matrices, dim=0).numpy()
            
            im = ax.imshow(relevance_matrix, cmap='Reds', aspect='auto')
            ax.set_xlabel('Position')
            ax.set_ylabel('Layer')
            ax.set_title('Layer-wise Relevance Scores')
            
            # Set tick labels
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels([name.split('.')[-1][:20] for name in layer_names])
            
            plt.colorbar(im, ax=ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def create_interpretability_dashboard(model: nn.Module, sample_inputs: torch.Tensor):
    """
    Create a comprehensive interpretability dashboard
    """
    interpreter = ModelInterpretability(model)
    
    # Run all analyses
    memory_analysis = interpreter.analyze_hololink_memory_patterns(sample_inputs)
    attention_analysis = interpreter.analyze_attention_patterns(sample_inputs)
    
    # Generate visualizations
    viz_path = f"interpretability_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    report = interpreter.visualize_interpretability_report(sample_inputs, save_path=viz_path)
    
    # LRP analysis
    lrp = LayerWiseRelevancePropagation(model)
    relevance_scores = lrp.propagate_relevance(sample_inputs)
    
    lrp_viz_path = f"lrp_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    lrp.visualize_relevance(relevance_scores, save_path=lrp_viz_path)
    
    return {
        'memory_analysis': memory_analysis,
        'attention_analysis': attention_analysis,
        'interpretability_report': report,
        'relevance_scores': relevance_scores,
        'visualizations': {
            'interpretability': viz_path,
            'lrp': lrp_viz_path
        }
    }