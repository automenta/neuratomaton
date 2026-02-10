"""
ANA Model Space - Systematic Architecture Exploration

Defines a taxonomy of possible architectures with modular components.
Each model is defined by a configuration that specifies which components to use.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MODEL TYPE TAXONOMY
# ============================================================================

class ModelType:
    """Base type for all model variants."""
    pass


class SSMType(ModelType):
    """State Space Model variants."""
    pass


class AttentionType(ModelType):
    """Attention-based variants."""
    pass


class HybridType(ModelType):
    """Hybrid SSM + Attention variants."""
    pass


# ============================================================================
# COMPONENT REGISTRY
# ============================================================================

@dataclass
class ComponentConfig:
    """Base configuration for any component."""
    enabled: bool = True
    variant: str = "default"  # Allows multiple implementations of same component


# ----------------------------------------------------------------------------
# Core Components
# ----------------------------------------------------------------------------

@dataclass
class EmbeddingConfig(ComponentConfig):
    """Token embedding component."""
    d_model: int = 64
    vocab_size: int = 30
    use_position_encoding: bool = True
    pos_encoding_type: str = "sinusoidal"  # "sinusoidal", "learned", "relative"


@dataclass
class StateSpaceConfig(ComponentConfig):
    """Core state space recurrence."""
    d_model: int = 64
    state_dim: int = 64
    num_layers: int = 2
    variant: str = "ssm"  # "ssm", "mamba", "s4", "rwkv"


@dataclass
class MultiTrackConfig(ComponentConfig):
    """Multi-track specialization."""
    enabled: bool = False
    track_count: int = 2
    track_dims: Optional[List[int]] = None  # If None, equal split
    track_names: Optional[List[str]] = None  # If None, ["track_0", "track_1", ...]
    shared_projection: bool = False


@dataclass
class HoloLinkConfig(ComponentConfig):
    """Holographic associative memory."""
    enabled: bool = False
    key_dim: int = 64
    memory_dim: int = 128
    use_binding: bool = True  # Outer product binding
    use_content_addressing: bool = False  # Learnable key hashing
    orthogonal_init: bool = False


@dataclass
class ControllerConfig(ComponentConfig):
    """Dynamic gating controller."""
    enabled: bool = False
    hidden_dim: int = 64
    num_layers: int = 2
    control_signals: List[str] = field(default_factory=lambda: ["gate", "reset", "update"])


@dataclass
class StackConfig(ComponentConfig):
    """Meta-state stack."""
    enabled: bool = False
    stack_dim: int = 64
    stack_depth: int = 5
    use_gumbel_routing: bool = False
    num_opcodes: int = 4


@dataclass
class FaultTraceConfig(ComponentConfig):
    """Fault/error trace memory."""
    enabled: bool = False
    buffer_size: int = 100
    fault_dim: int = 128
    holographic: bool = True


@dataclass
class AttentionConfig(ComponentConfig):
    """Attention mechanism (for hybrid models)."""
    enabled: bool = False
    num_heads: int = 4
    attention_type: str = "standard"  # "standard", "linear", "performer"
    context_window: Optional[int] = None


# ----------------------------------------------------------------------------
# Training / Optimization Components
# ----------------------------------------------------------------------------

@dataclass
class OutputConfig(ComponentConfig):
    """Output projection."""
    d_model: int = 64
    vocab_size: int = 30
    num_output_heads: int = 1  # 1 = just LM, 2 = LM + auxiliary


@dataclass
class NormalizationConfig(ComponentConfig):
    """Normalization strategy."""
    variant: str = "layer_norm"  # "layer_norm", "rms_norm", "none"
    epsilon: float = 1e-5


@dataclass
class ActivationConfig(ComponentConfig):
    """Activation function."""
    variant: str = "silu"  # "silu", "gelu", "relu", "tanh"


# ============================================================================
# MODEL ARCHITECTURE SPECIFICATIONS
# ============================================================================

@dataclass
class ArchitectureSpec:
    """
    Complete specification of an architecture.
    This is the "DNA" of a model - all possible configurations.
    """
    # Model type
    model_type: str = "ssm"  # "ssm", "attention", "hybrid"

    # Core components
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    state_space: StateSpaceConfig = field(default_factory=StateSpaceConfig)

    # Optional SSM components
    multi_track: MultiTrackConfig = field(default_factory=MultiTrackConfig)
    hololink: HoloLinkConfig = field(default_factory=HoloLinkConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    stack: StackConfig = field(default_factory=StackConfig)
    fault_trace: FaultTraceConfig = field(default_factory=FaultTraceConfig)

    # Optional attention components (for hybrid)
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    # Output and normalization
    output: OutputConfig = field(default_factory=OutputConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)

    # Training parameters
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    # Metadata
    name: str = "unnamed"
    description: str = ""

    def get_all_enabled_components(self) -> Dict[str, bool]:
        """Get dictionary of which optional components are enabled."""
        return {
            "multi_track": self.multi_track.enabled,
            "hololink": self.hololink.enabled,
            "controller": self.controller.enabled,
            "stack": self.stack.enabled,
            "fault_trace": self.fault_trace.enabled,
            "attention": self.attention.enabled,
        }


# ============================================================================
# PREDEFINED ARCHITECTURE FAMILIES
# ============================================================================

def create_baseline_ssm() -> ArchitectureSpec:
    """Simple SSM baseline - minimal architecture."""
    return ArchitectureSpec(
        name="baseline_ssm",
        description="Simple state space model with single track",
        model_type="ssm",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


def create_multi_track_ssm() -> ArchitectureSpec:
    """Multi-track SSM without controller."""
    return ArchitectureSpec(
        name="multi_track_ssm",
        description="SSM with multiple parallel tracks",
        model_type="ssm",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
        multi_track=MultiTrackConfig(
            enabled=True,
            track_count=2,
            track_names=["syntax", "semantic"],
        ),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


def create_hololink_ssm() -> ArchitectureSpec:
    """SSM with holographic associative memory."""
    return ArchitectureSpec(
        name="hololink_ssm",
        description="SSM with HoloLink holographic memory",
        model_type="ssm",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
        multi_track=MultiTrackConfig(enabled=True, track_count=2),
        hololink=HoloLinkConfig(
            enabled=True,
            key_dim=64,
            memory_dim=128,
            use_binding=True,
        ),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


def create_controlled_ssm() -> ArchitectureSpec:
    """SSM with dynamic gating controller."""
    return ArchitectureSpec(
        name="controlled_ssm",
        description="SSM with controller gating",
        model_type="ssm",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
        multi_track=MultiTrackConfig(enabled=True, track_count=2),
        controller=ControllerConfig(enabled=True, hidden_dim=64, num_layers=2),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


def create_ana_full() -> ArchitectureSpec:
    """Full ANA with all components."""
    return ArchitectureSpec(
        name="ana_full",
        description="Full ANA with all SSM components",
        model_type="ssm",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
        multi_track=MultiTrackConfig(enabled=True, track_count=3, track_names=["syntax", "semantic", "logic"]),
        hololink=HoloLinkConfig(enabled=True, key_dim=64, memory_dim=128, use_binding=True),
        controller=ControllerConfig(enabled=True, hidden_dim=64, num_layers=2),
        stack=StackConfig(enabled=True, stack_dim=64, stack_depth=5),
        fault_trace=FaultTraceConfig(enabled=True, buffer_size=100, fault_dim=128),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


def create_ana_v2() -> ArchitectureSpec:
    """ANA v2 with Gumbel routing and cortex."""
    return ArchitectureSpec(
        name="ana_v2",
        description="ANA v2 with Gumbel stack routing and cortex controller",
        model_type="ssm",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
        multi_track=MultiTrackConfig(enabled=True, track_count=3),
        hololink=HoloLinkConfig(enabled=False),  # Uses fault buffer instead
        controller=ControllerConfig(enabled=True, hidden_dim=64, num_layers=2),
        stack=StackConfig(enabled=True, stack_dim=64, stack_depth=5, use_gumbel_routing=True),
        fault_trace=FaultTraceConfig(enabled=True, buffer_size=100, fault_dim=128),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


def create_hybrid_ssm_attention() -> ArchitectureSpec:
    """Hybrid SSM + Attention model."""
    return ArchitectureSpec(
        name="hybrid_ssm_attention",
        description="Hybrid model combining SSM and attention",
        model_type="hybrid",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
        multi_track=MultiTrackConfig(enabled=True, track_count=2),
        attention=AttentionConfig(enabled=True, num_heads=4, attention_type="standard"),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


def create_transformer() -> ArchitectureSpec:
    """Standard transformer baseline."""
    return ArchitectureSpec(
        name="transformer",
        description="Standard transformer (attention-only)",
        model_type="attention",
        embedding=EmbeddingConfig(d_model=64, vocab_size=30),
        attention=AttentionConfig(enabled=True, num_heads=4, attention_type="standard"),
        output=OutputConfig(d_model=64, vocab_size=30),
    )


# ============================================================================
# MODEL SPACE EXPLORATION
# ============================================================================

PREDEFINED_ARCHITECTURES = {
    "baseline_ssm": create_baseline_ssm,
    "multi_track_ssm": create_multi_track_ssm,
    "hololink_ssm": create_hololink_ssm,
    "controlled_ssm": create_controlled_ssm,
    "ana_full": create_ana_full,
    "ana_v2": create_ana_v2,
    "hybrid_ssm_attention": create_hybrid_ssm_attention,
    "transformer": create_transformer,
}


def get_architecture(name: str) -> ArchitectureSpec:
    """Get a predefined architecture by name."""
    if name not in PREDEFINED_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}. Available: {list(PREDEFINED_ARCHITECTURES.keys())}")
    return PREDEFINED_ARCHITECTURES[name]()


def list_architectures() -> List[str]:
    """List all available architecture names."""
    return list(PREDEFINED_ARCHITECTURES.keys())


def generate_ablation_study(base_name: str = "ana_full") -> Dict[str, ArchitectureSpec]:
    """
    Generate ablation variants by disabling components one at a time.
    Returns dictionary of {variant_name: ArchitectureSpec}.
    """
    base = get_architecture(base_name)
    variants = {"original": base}

    # Disable each component individually
    if base.multi_track.enabled:
        no_track = ArchitectureSpec(**base.__dict__)
        no_track.multi_track.enabled = False
        no_track.name = f"{base_name}_no_multi_track"
        variants["no_multi_track"] = no_track

    if base.hololink.enabled:
        no_holo = ArchitectureSpec(**base.__dict__)
        no_holo.hololink.enabled = False
        no_holo.name = f"{base_name}_no_hololink"
        variants["no_hololink"] = no_holo

    if base.controller.enabled:
        no_ctrl = ArchitectureSpec(**base.__dict__)
        no_ctrl.controller.enabled = False
        no_ctrl.name = f"{base_name}_no_controller"
        variants["no_controller"] = no_ctrl

    if base.stack.enabled:
        no_stack = ArchitectureSpec(**base.__dict__)
        no_stack.stack.enabled = False
        no_stack.name = f"{base_name}_no_stack"
        variants["no_stack"] = no_stack

    if base.fault_trace.enabled:
        no_fault = ArchitectureSpec(**base.__dict__)
        no_fault.fault_trace.enabled = False
        no_fault.name = f"{base_name}_no_fault_trace"
        variants["no_fault_trace"] = no_fault

    return variants


def generate_grid_search_space() -> List[ArchitectureSpec]:
    """
    Generate a grid of architectures for systematic exploration.
    Each architecture is defined by which optional components are enabled.
    """
    configs = []

    # Components that can be toggled
    components = ["multi_track", "hololink", "controller", "stack", "fault_trace", "attention"]

    # Generate all 2^N combinations (N = number of components)
    import itertools
    for combo in itertools.product([False, True], repeat=len(components)):
        enabled = dict(zip(components, combo))

        # Skip attention-only (no SSM)
        if enabled["attention"] and not any(enabled[k] for k in ["multi_track", "hololink", "controller", "stack", "fault_trace"]):
            continue

        spec = ArchitectureSpec(
            name=f"grid_{'_'.join(k if enabled[k] else f'no_{k}' for k in components)}",
            model_type="hybrid" if enabled["attention"] else "ssm",
            embedding=EmbeddingConfig(d_model=64, vocab_size=30),
            state_space=StateSpaceConfig(d_model=64, state_dim=64, num_layers=2),
            multi_track=MultiTrackConfig(enabled=enabled["multi_track"], track_count=2),
            hololink=HoloLinkConfig(enabled=enabled["hololink"], key_dim=64),
            controller=ControllerConfig(enabled=enabled["controller"]),
            stack=StackConfig(enabled=enabled["stack"]),
            fault_trace=FaultTraceConfig(enabled=enabled["fault_trace"]),
            attention=AttentionConfig(enabled=enabled["attention"], num_heads=4),
        )
        configs.append(spec)

    return configs
