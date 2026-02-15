from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class ANAConfig:
    """
    Configuration for Adaptive Neural Automaton (ANA) models.

    This configuration object centralizes all hyperparameters for the ANA architecture,
    including model dimensions, training settings, and experimental flags.

    Attributes:
        vocab_size (int): Size of the vocabulary. Default: 40.
        d_model (int): Dimension of the model's hidden states and embeddings. Default: 64.
        state_dim (int): Dimension of the internal state space for LRUs. Default: 64.
        num_layers (int): Number of ANA layers. Default: 2.
        dropout (float): Dropout probability. Default: 0.0.
        max_position (int): Maximum sequence length for positional encodings. Default: 8192.

        key_dim (int): Dimension of keys in the HoloLink associative memory. Default: 64.

        batch_size (int): Batch size for training. Default: 16.
        learning_rate (float): Learning rate for the optimizer. Default: 1e-3.
        epochs (int): Number of training epochs. Default: 6.
        device (str): Device to run the model on ('cuda' or 'cpu'). Default: 'cuda' if available.

        use_hololink (bool): Whether to enable the HoloLink associative memory module. Default: True.
        use_controller (bool): Whether to enable the HyperController for dynamic gating. Default: True.
        track_count (int): Number of parallel processing tracks in each layer. Default: 2.
        use_parallel_scan (bool): Whether to use the parallel scan algorithm (O(log N)) for sequence processing. Default: False.
        force_prob (float): Probability of forcing a specific cognitive path during training (curriculum learning). Default: 0.0.
        max_thinking_steps (int): Maximum number of internal "thinking" steps (adaptive computation time) per token. Default: 0 (disabled).

        image_size (int): Input image size for vision tasks. Default: 224.
        patch_size (int): Patch size for vision transformers. Default: 16.
        vision_encoder (str): Type of vision encoder to use (e.g., 'vit'). Default: 'vit'.

        action_space (int): Size of the action space for RL tasks. Default: 4.
        observation_space (int): Size of the observation space for RL tasks. Default: 10.

        audio_sample_rate (int): Sample rate for audio tasks. Default: 16000.
        series_dim (int): Number of features per time step for time series/audio tasks. Default: 1.
    """
    # Model Architecture
    vocab_size: int = 40
    d_model: int = 64
    state_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    max_position: int = 8192

    # HoloLink
    key_dim: int = 64

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Ablations & Flags
    use_hololink: bool = True
    use_controller: bool = True
    track_count: int = 2
    use_parallel_scan: bool = False
    force_prob: float = 0.0 # For curriculum learning
    max_thinking_steps: int = 0

    # Vision (Phase 3)
    image_size: int = 224
    patch_size: int = 16
    vision_encoder: str = "vit" # Defaulting to ViT-style patch embedding

    # RL (Phase 4)
    action_space: int = 4
    observation_space: int = 10

    # Audio/Scientific (Phase 5)
    audio_sample_rate: int = 16000
    series_dim: int = 1 # Number of features per time step (e.g., 1 for audio, N for multi-variate)
