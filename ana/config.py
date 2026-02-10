from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ANAConfig:
    d_model: int = 64
    state_dim: int = 64
    num_layers: int = 2
    track_count: int = 2
    vocab_size: int = 40
    
    controller_hidden_dim: int = 64
    controller_layers: int = 2
    
    key_dim: int = 64
    hololink_decay: float = 1.0
    orthogonal_init: bool = False
    use_learned_binding: bool = True
    
    max_thinking_steps: int = 0
    use_act: bool = False
    act_epsilon: float = 0.01
    
    use_parallel_scan: bool = True
    use_hololink: bool = True
    use_controller: bool = True
    
    use_position_encoding: bool = True
    max_seq_len: int = 512
    
    dropout: float = 0.0

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 10
    device: str = "auto"
    output_dir: str = "archive/results"
    stage: str = "2a"
    seed: int = 42
    
    curriculum_epochs: int = 5
    start_force_prob: float = 1.0
    complexity_curriculum: bool = True
    
    log_interval: int = 10
    save_checkpoints: bool = True

@dataclass
class DataConfig:
    vocab_size: int = 40
    seq_len: int = 64
    min_noise: int = 10
    max_noise: int = 50
    dataset_size: int = 2000
    dataset_path: Optional[str] = None
    
    def __post_init__(self):
        if self.dataset_path is None:
            self.dataset_path = "data/corpus.txt"
