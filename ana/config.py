from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ANAConfig:
    d_model: int = 64
    state_dim: int = 64
    num_layers: int = 2
    num_tracks: int = 2
    vocab_size: int = 40
    use_hololink: bool = True
    use_controller: bool = True
    use_parallel_scan: bool = False
    orthogonal_init: bool = False
    dropout: float = 0.1
    controller_hidden_dim: int = 64
    controller_layers: int = 2
    hololink_decay: float = 1.0 # 1.0 = No decay, 0.9 = decay

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 10
    device: str = "cuda" if "cuda" in "cuda" else "cpu" # Placeholder logic
    output_dir: str = "archive/results"
    stage: str = "2a" # 2a, 2b, 3a
    seed: int = 42
    # Curriculum
    curriculum_epochs: int = 5
    start_force_prob: float = 1.0

@dataclass
class DataConfig:
    vocab_size: int = 40
    seq_len: int = 64
    min_noise: int = 10
    max_noise: int = 50
    dataset_size: int = 2000
    dataset_path: Optional[str] = None
