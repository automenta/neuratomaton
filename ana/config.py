from dataclasses import dataclass
from typing import Optional

@dataclass
class ANAConfig:
    # Model Architecture
    vocab_size: int = 40
    d_model: int = 64
    state_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0

    # HoloLink
    key_dim: int = 64

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 6
    device: str = "cuda" if "cuda" in "cuda" else "cpu" # Placeholder, will be set at runtime

    # Ablations & Flags
    use_hololink: bool = True
    use_controller: bool = True
    track_count: int = 2
    use_parallel_scan: bool = False
    force_prob: float = 0.0 # For curriculum learning
    max_thinking_steps: int = 0
