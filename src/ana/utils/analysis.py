"""
Model analysis and visualization tools for ANA
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from datetime import datetime


class ModelAnalyzer:
    """
    Comprehensive model analysis tools for ANA models
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def analyze_hololink_memory(self) -> Dict[str, Any]:
        """
        Analyze the HoloLink memory component specifically
        """
        hololink_stats = {}
        
        for name, module in self.model.named_modules():
            if 'holo' in name.lower() and hasattr(module, 'binding_strength'):
                # Analyze binding strength
                binding_val = torch.sigmoid(module.binding_strength).item()
                hololink_stats[f'{name}.binding_strength'] = binding_val
                
                # Analyze projection matrices
                if hasattr(module, 'k_proj'):
                    k_norm = module.k_proj.weight.norm().item()
                    hololink_stats[f'{name}.k_proj_norm'] = k_norm
                
                if hasattr(module, 'v_proj'):
                    v_norm = module.v_proj.weight.norm().item()
                    hololink_stats[f'{name}.v_proj_norm'] = v_norm
                
                if hasattr(module, 'q_proj'):
                    q_norm = module.q_proj.weight.norm().item()
                    hololink_stats[f'{name}.q_proj_norm'] = q_norm
        
        return hololink_stats
    
    def analyze_gradient_flow(self) -> Dict[str, Any]:
        """
        Analyze gradient flow through the network
        """
        grad_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_abs_mean = param.grad.abs().mean().item()
                grad_abs_max = param.grad.abs().max().item()
                grad_std = param.grad.std().item()
                
                grad_stats[name] = {
                    'grad_norm': grad_norm,
                    'grad_abs_mean': grad_abs_mean,
                    'grad_abs_max': grad_abs_max,
                    'grad_std': grad_std
                }
        
        return grad_stats
    
    def analyze_parameter_statistics(self) -> Dict[str, Any]:
        """
        Analyze parameter distributions
        """
        param_stats = {}
        
        for name, param in self.model.named_parameters():
            param_norm = param.norm().item()
            param_mean = param.mean().item()
            param_std = param.std().item()
            param_abs_mean = param.abs().mean().item()
            param_min = param.min().item()
            param_max = param.max().item()
            
            param_stats[name] = {
                'param_norm': param_norm,
                'param_mean': param_mean,
                'param_std': param_std,
                'param_abs_mean': param_abs_mean,
                'param_min': param_min,
                'param_max': param_max
            }
        
        return param_stats
    
    def analyze_activation_statistics(self, sample_input: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze activation statistics during forward pass
        """
        activation_stats = {}
        
        # Register hooks to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                # Use input[0] as the activation (first input to the module)
                if len(input) > 0 and input[0] is not None:
                    activations[name] = input[0].detach()
            return hook
        
        # Register hooks for key layers
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.SiLU)):
                handle = module.register_forward_hook(get_activation(name))
                handles.append(handle)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                _ = self.model(sample_input)
            else:
                _ = self.model(sample_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Calculate statistics
        for name, activation in activations.items():
            act_norm = activation.norm().item()
            act_mean = activation.mean().item()
            act_std = activation.std().item()
            act_abs_mean = activation.abs().mean().item()
            act_min = activation.min().item()
            act_max = activation.max().item()
            
            activation_stats[name] = {
                'activation_norm': act_norm,
                'activation_mean': act_mean,
                'activation_std': act_std,
                'activation_abs_mean': act_abs_mean,
                'activation_min': act_min,
                'activation_max': act_max
            }
        
        return activation_stats


class ModelVisualizer:
    """
    Visualization tools for ANA models
    """
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_parameter_heatmap(self, param_dict: Dict[str, Dict], 
                              title: str = "Parameter Statistics Heatmap", 
                              save_path: Optional[str] = None):
        """
        Plot heatmap of parameter statistics
        """
        if not param_dict:
            print("No parameter data to plot")
            return
        
        # Extract parameter names and statistics
        param_names = list(param_dict.keys())
        if not param_names:
            print("No parameter names found")
            return
        
        # Get all available statistics
        stats_keys = list(param_dict[param_names[0]].keys())
        
        # Create data matrix
        data_matrix = []
        for name in param_names:
            row = [param_dict[name].get(stat, 0.0) for stat in stats_keys]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, len(param_names) * 0.3)))
        
        im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=np.nanmin(data_matrix), vmax=np.nanmax(data_matrix))
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(stats_keys)))
        ax.set_yticks(np.arange(len(param_names)))
        ax.set_xticklabels(stats_keys, rotation=45, ha="right")
        ax.set_yticklabels([name.split('.')[-1][:30] for name in param_names])  # Shorten names
        
        # Add text annotations
        for i in range(len(param_names)):
            for j in range(len(stats_keys)):
                text = ax.text(j, i, f"{data_matrix[i, j]:.3f}",
                              ha="center", va="center", color="white" if abs(data_matrix[i, j]) > np.nanmax(abs(data_matrix))/2 else "black",
                              fontsize=8)
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_curves(self, train_losses: List[float], 
                           val_losses: Optional[List[float]] = None,
                           title: str = "Training Curves",
                           save_path: Optional[str] = None):
        """
        Plot training curves
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(train_losses, label='Training Loss', alpha=0.7, linewidth=2)
        if val_losses is not None:
            ax.plot(val_losses, label='Validation Loss', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_gradient_flow(self, grad_dict: Dict[str, Dict], 
                         title: str = "Gradient Flow Analysis",
                         save_path: Optional[str] = None):
        """
        Plot gradient flow analysis
        """
        if not grad_dict:
            print("No gradient data to plot")
            return
        
        param_names = list(grad_dict.keys())
        grad_norms = [grad_dict[name]['grad_norm'] for name in param_names]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars = ax.bar(range(len(param_names)), grad_norms)
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(title)
        
        # Rotate x-axis labels
        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels([name.split('.')[-1][:20] for name in param_names], 
                          rotation=45, ha="right")
        
        # Color bars based on magnitude
        colors = plt.cm.viridis(np.array(grad_norms) / max(grad_norms))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_embeddings(self, embeddings: torch.Tensor, 
                           labels: Optional[List[str]] = None,
                           method: str = 'tsne',
                           title: str = "Embedding Visualization",
                           save_path: Optional[str] = None):
        """
        Visualize embeddings using dimensionality reduction
        """
        # Convert to numpy
        emb_np = embeddings.detach().cpu().numpy()
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(emb_np)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                           c=range(len(reduced)), cmap='tab10', alpha=0.7)
        
        if labels:
            # Add text labels for points (only if few points to avoid clutter)
            if len(labels) <= 20:
                for i, label in enumerate(labels):
                    ax.annotate(label[:10], (reduced[i, 0], reduced[i, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)
        
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class PerformanceAnalyzer:
    """
    Performance analysis tools
    """
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
    
    def profile_memory_usage(self) -> Dict[str, Any]:
        """
        Profile memory usage of the model
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        buffer_count = sum(b.numel() for b in self.model.buffers())
        total_params = param_count + buffer_count
        
        param_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        buffer_size_mb = sum(b.numel() * b.element_size() for b in self.model.buffers()) / (1024**2)
        total_size_mb = param_size_mb + buffer_size_mb
        
        if self.device == 'cuda' and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            reserved_memory = torch.cuda.max_memory_reserved() / (1024**2)
        else:
            peak_memory = 0.0
            reserved_memory = 0.0
        
        return {
            'parameter_count': param_count,
            'buffer_count': buffer_count,
            'total_params': total_params,
            'parameter_size_mb': param_size_mb,
            'buffer_size_mb': buffer_size_mb,
            'total_size_mb': total_size_mb,
            'peak_gpu_memory_mb': peak_memory,
            'reserved_gpu_memory_mb': reserved_memory
        }
    
    def profile_compute_complexity(self, input_shapes: List[Tuple]) -> Dict[str, Any]:
        """
        Profile compute complexity for different input shapes
        """
        results = {}
        
        for shape in input_shapes:
            # Create dummy input
            dummy_input = torch.randn(shape).to(self.device)
            
            # Count operations
            flops = 0
            
            # Forward pass with hooks to count operations
            def count_flops(module, input, output):
                nonlocal flops
                if isinstance(module, nn.Linear):
                    # Linear layer: input_size * output_size * batch_size * seq_len
                    batch_size = input[0].size(0) if input[0].dim() > 1 else 1
                    seq_len = input[0].size(-2) if input[0].dim() > 1 else 1
                    input_features = module.in_features
                    output_features = module.out_features
                    flops += batch_size * seq_len * input_features * output_features * 2  # multiply-adds
                    
            # Register hooks
            handles = []
            for layer in self.model.modules():
                if isinstance(layer, nn.Linear):
                    handle = layer.register_forward_hook(count_flops)
                    handles.append(handle)
            
            # Run forward pass
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            results[str(shape)] = {
                'approximate_flops': flops,
                'shape': shape,
                'flops_per_element': flops / dummy_input.numel() if dummy_input.numel() > 0 else 0
            }
        
        return results


class ExperimentAnalyzer:
    """
    Analysis tools for experiments
    """
    def __init__(self):
        pass
    
    def compare_models(self, model_results: List[Dict]) -> pd.DataFrame:
        """
        Compare results from multiple models
        """
        df_data = []
        
        for i, result in enumerate(model_results):
            row = {
                'model_id': i,
                'train_loss': result.get('train_loss', np.nan),
                'val_loss': result.get('val_loss', np.nan),
                'perplexity': result.get('perplexity', np.nan),
                'accuracy': result.get('accuracy', np.nan),
                'parameters': result.get('parameters', np.nan),
                'training_time': result.get('training_time', np.nan),
                'model_type': result.get('model_type', 'unknown')
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def plot_model_comparison(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot model comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Perplexity vs Parameters
        axes[0, 0].scatter(df['parameters'], df['perplexity'], alpha=0.7)
        axes[0, 0].set_xlabel('Parameters')
        axes[0, 0].set_ylabel('Perplexity')
        axes[0, 0].set_title('Perplexity vs Parameters')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy vs Parameters
        axes[0, 1].scatter(df['parameters'], df['accuracy'], alpha=0.7)
        axes[0, 1].set_xlabel('Parameters')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Parameters')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training Time vs Parameters
        axes[1, 0].scatter(df['parameters'], df['training_time'], alpha=0.7)
        axes[1, 0].set_xlabel('Parameters')
        axes[1, 0].set_ylabel('Training Time (s)')
        axes[1, 0].set_title('Training Time vs Parameters')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Perplexity vs Accuracy
        axes[1, 1].scatter(df['accuracy'], df['perplexity'], alpha=0.7)
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].set_title('Perplexity vs Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()