from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ANAv2Config:
    d_model: int = 128
    vocab_size: int = 50
    
    syntax_dim: int = 64
    semantic_dim: int = 128
    logic_dim: int = 64
    
    num_layers: int = 2
    
    stack_depth: int = 5
    stack_dim: int = 64
    num_opcodes: int = 4
    
    cortex_hidden_dim: int = 128
    cortex_layers: int = 2
    
    fault_buffer_size: int = 100
    fault_dim: int = 512
    fault_threshold: float = 2.0
    
    gumbel_temp_init: float = 1.0
    gumbel_temp_min: float = 0.1
    gumbel_decay_steps: int = 10000
    
    use_parallel_scan: bool = True
    use_position_encoding: bool = True
    max_seq_len: int = 512
    
    meta_loss_weight: float = 0.1
    density_reg_weight: float = 0.01
    
    @property
    def total_track_dim(self):
        return self.syntax_dim + self.semantic_dim + self.logic_dim

@dataclass
class Trainingv2Config:
    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 30
    device: str = "auto"
    output_dir: str = "archive/results_v2"
    stage: str = "0"
    seed: int = 42
    
    log_interval: int = 10
    save_checkpoints: bool = True
    
    grad_clip: float = 1.0
    warmup_steps: int = 500
    
    use_mixed_precision: bool = False

@dataclass
class Datav2Config:
    vocab_size: int = 50
    seq_len: int = 64
    min_noise: int = 10
    max_noise: int = 50
    dataset_size: int = 2000
    dataset_path: Optional[str] = None
