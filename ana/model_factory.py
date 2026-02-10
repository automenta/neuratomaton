"""
Model Factory - Construct models from ArchitectureSpec

This module provides a factory that builds actual PyTorch models from
ArchitectureSpec configurations. It implements all the modular components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

from .model_space import (
    ArchitectureSpec,
    EmbeddingConfig,
    StateSpaceConfig,
    MultiTrackConfig,
    HoloLinkConfig,
    ControllerConfig,
    StackConfig,
    FaultTraceConfig,
    AttentionConfig,
    OutputConfig,
    NormalizationConfig,
    ActivationConfig,
)


# ============================================================================
# BASIC BUILDING BLOCKS
# ============================================================================

def create_normalization(norm_config: NormalizationConfig, dim: int) -> nn.Module:
    """Create normalization layer based on config."""
    if norm_config.variant == "layer_norm":
        return nn.LayerNorm(dim, eps=norm_config.epsilon)
    elif norm_config.variant == "rms_norm":
        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-5):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return RMSNorm(dim, eps=norm_config.epsilon)
    elif norm_config.variant == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization: {norm_config.variant}")


def create_activation(act_config: ActivationConfig) -> nn.Module:
    """Create activation function based on config."""
    if act_config.variant == "silu":
        return nn.SiLU()
    elif act_config.variant == "gelu":
        return nn.GELU()
    elif act_config.variant == "relu":
        return nn.ReLU()
    elif act_config.variant == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {act_config.variant}")


def sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Create sinusoidal position encoding."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


# ============================================================================
# COMPONENT IMPLEMENTATIONS
# ============================================================================

class StateSpaceLayer(nn.Module):
    """Basic SSM layer (linear recurrence) - matches original ANA implementation."""

    def __init__(self, config: StateSpaceConfig, norm: NormalizationConfig, act: ActivationConfig):
        super().__init__()
        self.d_model = config.d_model
        self.state_dim = config.state_dim

        # Input/output projections
        self.input_proj = nn.Linear(config.d_model, config.state_dim)
        self.output_proj = nn.Linear(config.state_dim, config.d_model)

        # Learned decay/retention rates (per dimension) - KEY DIFFERENCE
        self.alpha_logit = nn.Parameter(torch.Tensor(config.state_dim).uniform_(2, 4))
        self.beta_logit = nn.Parameter(torch.Tensor(config.state_dim).uniform_(-2, 0))

    def forward(self, x, h_init=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            h_init: (batch, state_dim) or None
        Returns:
            y: (batch, seq_len, d_model)
            h_final: (batch, state_dim)
        """
        batch, seq_len, _ = x.shape

        # Project input to state space
        u = self.input_proj(x)  # (batch, seq_len, state_dim)

        # Compute alpha and beta (learned decay rates)
        alpha = torch.sigmoid(self.alpha_logit)  # (state_dim,) typically ~0.9-0.98
        beta = torch.sigmoid(self.beta_logit)    # (state_dim,) typically ~0.1-0.5

        # Initialize hidden state
        if h_init is None:
            h = torch.zeros(batch, self.state_dim, device=x.device)
        else:
            h = h_init

        outputs = []
        for t in range(seq_len):
            # SSM update: h[t] = alpha * h[t-1] + beta * u[t] (element-wise!)
            h = alpha * h + beta * u[:, t]
            y = self.output_proj(h)
            outputs.append(y)

        outputs = torch.stack(outputs, dim=1)
        return outputs, h


class MultiTrackLayer(nn.Module):
    """Multi-track SSM with specialized tracks."""

    def __init__(self, config: MultiTrackConfig, state_config: StateSpaceConfig,
                 norm: NormalizationConfig, act: ActivationConfig):
        super().__init__()
        self.track_count = config.track_count
        self.d_model = state_config.d_model
        self.state_dim = state_config.state_dim

        # Determine track dimensions
        if config.track_dims is None:
            self.track_dims = [self.state_dim // self.track_count] * self.track_count
        else:
            self.track_dims = config.track_dims

        # Ensure sum matches state_dim
        total_track_dim = sum(self.track_dims)
        if total_track_dim != self.state_dim:
            # Adjust last track
            self.track_dims[-1] += self.state_dim - total_track_dim

        # Input projections per track
        self.track_input_projs = nn.ModuleList([
            nn.Linear(self.d_model, dim) for dim in self.track_dims
        ])

        # Output projections per track
        self.track_output_projs = nn.ModuleList([
            nn.Linear(dim, self.d_model) for dim in self.track_dims
        ])

        # Learned decay rates per track (matching original ANA)
        self.track_alpha_logits = nn.ParameterList([
            nn.Parameter(torch.Tensor(dim).uniform_(2, 4)) for dim in self.track_dims
        ])
        self.track_beta_logits = nn.ParameterList([
            nn.Parameter(torch.Tensor(dim).uniform_(-2, 0)) for dim in self.track_dims
        ])

        # Combine all track outputs
        self.combine_proj = nn.Linear(self.d_model * self.track_count, self.d_model)

    def forward(self, x, h_init=None):
        """Forward pass through all tracks in parallel."""
        batch, seq_len, d_model = x.shape

        if h_init is None:
            h_init = [torch.zeros(batch, dim, device=x.device) for dim in self.track_dims]

        track_outputs = []

        for t in range(seq_len):
            xt = x[:, t]  # (batch, d_model)

            layer_outputs = []
            new_states = []

            for i in range(self.track_count):
                # Project to track space
                u = self.track_input_projs[i](xt)

                # Get learned decay rates
                alpha = torch.sigmoid(self.track_alpha_logits[i])
                beta = torch.sigmoid(self.track_beta_logits[i])

                # SSM update: h = alpha * h + beta * u (element-wise)
                h = alpha * h_init[i] + beta * u

                # Project back
                y = self.track_output_projs[i](h)

                layer_outputs.append(y)
                new_states.append(h)

            # Combine track outputs
            combined = torch.cat(layer_outputs, dim=-1)
            output = self.combine_proj(combined)

            track_outputs.append(output)
            track_states = new_states

        outputs = torch.stack(track_outputs, dim=1)
        return outputs, track_states


class HoloLinkMemory(nn.Module):
    """Holographic associative memory via outer product binding."""

    def __init__(self, config: HoloLinkConfig, d_model: int):
        super().__init__()
        self.key_dim = config.key_dim
        self.memory_dim = config.memory_dim
        self.use_binding = config.use_binding

        # Key and value projections
        self.key_proj = nn.Linear(d_model, config.key_dim)
        self.value_proj = nn.Linear(d_model, config.memory_dim)

        if config.orthogonal_init:
            nn.init.orthogonal_(self.key_proj.weight)

        # Query projection
        self.query_proj = nn.Linear(d_model, config.key_dim)
        self.output_proj = nn.Linear(config.memory_dim, d_model)

        # Learned mixing gate
        self.gate = nn.Linear(d_model + config.memory_dim, d_model)

    def forward(self, x, h_mem=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            h_mem: (batch, memory_dim) or None (current memory state)
        Returns:
            y: (batch, seq_len, d_model)
            h_mem: (batch, memory_dim) updated memory
        """
        batch, seq_len, d_model = x.shape

        if h_mem is None:
            h_mem = torch.zeros(batch, self.memory_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            xt = x[:, t]

            # Project to key and value
            k = F.normalize(self.key_proj(xt), dim=-1)  # (batch, key_dim)
            v = self.value_proj(xt)  # (batch, memory_dim)

            if self.use_binding:
                # Holographic binding: simple weighted update
                # Store value weighted by key norm
                weight = k.norm(dim=-1, keepdim=True)  # (batch, 1)
                h_mem = 0.9 * h_mem + 0.1 * weight * v
            else:
                # Simple update
                h_mem = 0.9 * h_mem + 0.1 * v

            # Query memory
            q = F.normalize(self.query_proj(xt), dim=-1)
            # Content-addressed retrieval using key as query
            k_norm = F.normalize(k, dim=-1)
            similarity = torch.sum(k_norm * q, dim=-1, keepdim=True)  # (batch, 1)
            retrieved = similarity * h_mem

            # Combine with original
            combined = torch.cat([xt, retrieved], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            y = gate * xt + (1 - gate) * self.output_proj(retrieved)

            outputs.append(y)

        outputs = torch.stack(outputs, dim=1)
        return outputs, h_mem


class Controller(nn.Module):
    """Dynamic gating controller."""

    def __init__(self, config: ControllerConfig, d_model: int):
        super().__init__()
        self.control_signals = config.control_signals

        self.hidden = nn.ModuleList()
        for i in range(config.num_layers):
            in_dim = d_model if i == 0 else config.hidden_dim
            self.hidden.append(nn.Linear(in_dim, config.hidden_dim))

        # Output projections for each control signal
        self.gate_proj = nn.Linear(config.hidden_dim, d_model)
        self.reset_proj = nn.Linear(config.hidden_dim, d_model)
        self.update_proj = nn.Linear(config.hidden_dim, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            control: dict with {signal_name: tensor}
        """
        batch, seq_len, _ = x.shape

        # Process through hidden layers
        h = x
        for layer in self.hidden:
            h = F.silu(layer(h))

        # Generate control signals
        control = {}
        if "gate" in self.control_signals:
            control["gate"] = torch.sigmoid(self.gate_proj(h))
        if "reset" in self.control_signals:
            control["reset"] = torch.sigmoid(self.reset_proj(h))
        if "update" in self.control_signals:
            control["update"] = torch.sigmoid(self.update_proj(h))

        return control


class SimpleStack(nn.Module):
    """Simple stack-based memory."""

    def __init__(self, config: StackConfig, d_model: int):
        super().__init__()
        self.stack_dim = config.stack_dim
        self.max_depth = config.stack_depth

        self.input_proj = nn.Linear(d_model, self.stack_dim)
        self.output_proj = nn.Linear(self.stack_dim, d_model)

        # Push/pop controller
        self.push_prob = nn.Linear(d_model, 1)
        self.pop_prob = nn.Linear(d_model, 1)

    def forward(self, x, stack_init=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            stack_init: optional initial stack state
        Returns:
            y: (batch, seq_len, d_model)
            stacks: list of stack states (one per batch)
        """
        batch, seq_len, d_model = x.shape

        stacks = [torch.zeros(self.max_depth, self.stack_dim, device=x.device) for _ in range(batch)]
        stack_pointers = [0 for _ in range(batch)]

        outputs = []
        for t in range(seq_len):
            xt = x[:, t]

            batch_outputs = []
            for b in range(batch):
                # Decide push or pop
                p_push = torch.sigmoid(self.push_prob(xt[b:b+1]))
                p_pop = torch.sigmoid(self.pop_prob(xt[b:b+1]))

                if p_push > 0.5 and stack_pointers[b] < self.max_depth:
                    # Push
                    val = self.input_proj(xt[b:b+1])
                    stacks[b][stack_pointers[b]] = val.squeeze(0)
                    stack_pointers[b] += 1
                elif p_pop > 0.5 and stack_pointers[b] > 0:
                    # Pop
                    stack_pointers[b] -= 1

                # Read top of stack
                if stack_pointers[b] > 0:
                    top = stacks[b][stack_pointers[b] - 1]
                else:
                    top = torch.zeros(self.stack_dim, device=x.device)

                # Output
                y = self.output_proj(top)
                batch_outputs.append(y)

            outputs.append(torch.stack(batch_outputs, dim=0))

        outputs = torch.stack(outputs, dim=1)
        return outputs, stacks


class FaultTraceBuffer(nn.Module):
    """Error trace memory."""

    def __init__(self, config: FaultTraceConfig, d_model: int):
        super().__init__()
        self.buffer_size = config.buffer_size
        self.fault_dim = config.fault_dim

        self.fault_proj = nn.Linear(d_model, self.fault_dim)
        self.summary_proj = nn.Linear(self.fault_dim, d_model)

        # Buffer as learnable parameters (no in-place updates during forward)
        self.buffer = nn.Parameter(torch.zeros(1, self.fault_dim))

    def forward(self, x, error_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            error_mask: (batch, seq_len) or None
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        summary = self.summary_proj(self.buffer.expand(batch, -1))
        outputs = x + summary.unsqueeze(1)

        # Note: Buffer is NOT updated during forward to avoid in-place issues
        # Buffer learning happens via gradients through the summary

        return outputs, summary


class AttentionLayer(nn.Module):
    """Standard self-attention."""

    def __init__(self, config: AttentionConfig, d_model: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = d_model // config.num_heads
        self.context_window = config.context_window

        assert d_model % config.num_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Apply context window
        if self.context_window and seq_len > self.context_window:
            x = x[:, -self.context_window:, :]
            seq_len = self.context_window

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        y = torch.matmul(attn, v)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        y = self.out_proj(y)

        return y


# ============================================================================
# MAIN MODEL BUILDER
# ============================================================================

class UniversalModel(nn.Module):
    """
    Universal model that can be configured from any ArchitectureSpec.
    This is the main entry point for the model factory.
    """

    def __init__(self, spec: ArchitectureSpec):
        super().__init__()
        self.spec = spec
        self.d_model = spec.embedding.d_model

        # Embedding
        self.embedding = nn.Embedding(spec.embedding.vocab_size, spec.embedding.d_model)

        if spec.embedding.use_position_encoding and spec.embedding.pos_encoding_type == "sinusoidal":
            max_len = spec.state_space.num_layers * 512  # Default max length
            self.register_buffer('pos_encoding', sinusoidal_position_encoding(max_len, spec.embedding.d_model))

        # Main processing layers
        self.layers = nn.ModuleList()

        for layer_idx in range(spec.state_space.num_layers):
            layer_modules = nn.ModuleDict()

            # Multi-track (if enabled)
            if spec.multi_track.enabled:
                layer_modules["track"] = MultiTrackLayer(
                    spec.multi_track, spec.state_space, spec.normalization, spec.activation
                )
            else:
                layer_modules["track"] = StateSpaceLayer(
                    spec.state_space, spec.normalization, spec.activation
                )

            # HoloLink (if enabled)
            if spec.hololink.enabled:
                layer_modules["hololink"] = HoloLinkMemory(spec.hololink, spec.embedding.d_model)

            # Controller (if enabled)
            if spec.controller.enabled:
                layer_modules["controller"] = Controller(spec.controller, spec.embedding.d_model)

            # Stack (if enabled)
            if spec.stack.enabled:
                layer_modules["stack"] = SimpleStack(spec.stack, spec.embedding.d_model)

            # Fault trace (if enabled)
            if spec.fault_trace.enabled:
                layer_modules["fault"] = FaultTraceBuffer(spec.fault_trace, self.d_model)

            # Attention (if enabled, for hybrid models)
            if spec.attention.enabled:
                layer_modules["attention"] = AttentionLayer(spec.attention, self.d_model)

            self.layers.append(layer_modules)

        # Output projection
        self.output_proj = nn.Linear(self.d_model, spec.output.vocab_size)

    def forward(self, input_ids, return_info=False):
        """
        Args:
            input_ids: (batch, seq_len)
            return_info: if True, return additional information
        Returns:
            logits: (batch, seq_len, vocab_size)
            info: optional dict with internal states
        """
        x = self.embedding(input_ids)

        # Add position encoding if configured
        if hasattr(self, 'pos_encoding'):
            seq_len = input_ids.size(1)
            x = x + self.pos_encoding[:, :seq_len, :]

        # Process through layers
        layer_outputs = []
        states = {}

        for layer_idx, layer in enumerate(self.layers):
            info = {}

            # Multi-track / SSM
            if "track" in layer:
                x, h = layer["track"](x)
                info["track_state"] = h

            # HoloLink
            if "hololink" in layer:
                x, h_mem = layer["hololink"](x)
                info["hololink_mem"] = h_mem

            # Controller
            if "controller" in layer:
                control = layer["controller"](x)
                info["control"] = control
                # Apply gating
                if "gate" in control:
                    x = x * control["gate"]

            # Stack
            if "stack" in layer:
                x, stacks = layer["stack"](x)
                info["stacks"] = stacks

            # Fault trace
            if "fault" in layer:
                x, summary = layer["fault"](x)
                info["fault_summary"] = summary

            # Attention
            if "attention" in layer:
                x_attn = layer["attention"](x)
                x = x + x_attn  # Residual connection

            layer_outputs.append(x)
            states[f"layer_{layer_idx}"] = info

        logits = self.output_proj(x)

        if return_info:
            return logits, states
        return logits, {}


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def build_model(spec: ArchitectureSpec) -> nn.Module:
    """Build a model from ArchitectureSpec."""
    model = UniversalModel(spec)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built model '{spec.name}' with {total_params:,} parameters")

    return model


def build_model_from_name(name: str) -> nn.Module:
    """Build a model from a predefined architecture name."""
    from .model_space import get_architecture
    spec = get_architecture(name)
    return build_model(spec)
