from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ANAConfig:
    """
    Configuration for ANA-1 Production Models.
    Defaults to ANA-Small (125M).
    """
    vocab_size: int = 50304 # GPT-NeoX
    d_model: int = 768
    n_layers: int = 12
    max_seq_len: int = 2048
    
    # Dual-Track Config
    # Split d_model equally by default
    d_state_A: int = 384 # Reflex
    d_state_B: int = 384 # Reasoning
    
    # Holo-Link Config
    holo_key_dim: int = 64
    holo_enc_heads: int = 12 # For the Induction/Copy Head
    
    # Initialization
    init_std: float = 0.02
    
    # Controller
    ctrl_dim: int = 64 # Small bottleneck for controller
    
    # Loss Weights
    lambda_ret: float = 0.1
    lambda_spar: float = 0.01

    @property
    def num_params(self) -> int:
        """Estimate parameter count."""
        # Embeddings: V * D
        emb = self.vocab_size * self.d_model
        # Layers: 12 * ( ... )
        # Rough calc: 
        # LRU: 3 linear matrices (in, recurrent, out) -> 3 * D^2 ? No Is diagonal.
        # Actually standard input projection is D*D.
        # Controller: D*64 + 64*5
        # Holo: D*K + D*V ...
        # Output: D * V
        return "~125M"

@dataclass
class ANAMiniConfig(ANAConfig):
    """
    Mini Configuration for rapid validation (approx 10M params).
    """
    d_model: int = 256
    n_layers: int = 6
    vocab_size: int = 50304
    d_state_A: int = 128
    d_state_B: int = 128
    holo_key_dim: int = 32
    holo_enc_heads: int = 4
    
    @property
    def num_params(self) -> int:
        return "~10M"

@dataclass
class ANAMicroConfig(ANAConfig):
    """
    Micro Configuration for GPU smoke testing (approx 1M params).
    Fits comfortably on shared GPUs.
    """
    d_model: int = 128
    n_layers: int = 2
    vocab_size: int = 50304
    d_state_A: int = 64
    d_state_B: int = 64
    holo_key_dim: int = 16
    holo_enc_heads: int = 2
    ctrl_dim: int = 32
    
    @property
    def num_params(self) -> int:
        return "~1M"
