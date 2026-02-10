from dataclasses import dataclass, field
from typing import Optional, List
from ana.config_v2 import ANAv2Config, Trainingv2Config


@dataclass
class BioANAConfig(ANAv2Config):
    variant: str = "nano"
    
    relaxation_iterations: int = 20
    nudge_strength: float = 0.1
    spectral_radius: float = 0.99
    
    sparsity: float = 0.1
    dale_constraint: bool = True
    noise_injection: float = 0.05
    
    use_hebbian_memory: bool = True
    hebbian_lr: float = 0.01
    hebbian_decay: float = 0.001
    
    hololink_key_dim: int = 128
    hololink_capacity: int = 1000
    
    @property
    def relaxation_config(self):
        from ana.eqprop.bioplausible.models.eqprop_base import EqPropModel
        return {
            'max_steps': self.relaxation_iterations,
            'epsilon': self.nudge_strength,
            'use_spectral_norm': True,
        }


@dataclass
class BioTrainingConfig(Trainingv2Config):
    curriculum_stages: List[str] = field(default_factory=lambda: ['0', '1', '2'])
    current_stage: str = '0'
    
    relaxation_schedule: List[int] = field(default_factory=lambda: [50, 40, 30, 20, 10])
    adaptive_relaxation: bool = True
    
    stage_0_threshold: float = 0.98
    stage_1_threshold: float = 0.90
    stage_2_threshold: float = 0.85
    
    convergence_patience: int = 3
    
    noise_tokens_min: int = 5
    noise_tokens_max: int = 50
    
    use_curriculum: bool = True


VARIANT_CONFIGS = {
    'nano': {
        'd_model': 128,
        'syntax_dim': 32,
        'semantic_dim': 64,
        'logic_dim': 32,
        'num_layers': 2,
        'stack_depth': 3,
        'params': '~10M'
    },
    'small': {
        'd_model': 512,
        'syntax_dim': 128,
        'semantic_dim': 256,
        'logic_dim': 128,
        'num_layers': 4,
        'stack_depth': 5,
        'params': '~125M'
    },
    'base': {
        'd_model': 768,
        'syntax_dim': 192,
        'semantic_dim': 384,
        'logic_dim': 192,
        'num_layers': 6,
        'stack_depth': 5,
        'params': '~360M'
    },
    'large': {
        'd_model': 1024,
        'syntax_dim': 256,
        'semantic_dim': 512,
        'logic_dim': 256,
        'num_layers': 8,
        'stack_depth': 8,
        'params': '~1.4B'
    }
}


def get_bio_config(variant: str = 'nano', **kwargs) -> BioANAConfig:
    variant_defaults = VARIANT_CONFIGS.get(variant, VARIANT_CONFIGS['nano'])
    config_dict = {k: v for k, v in variant_defaults.items() if k != 'params'}
    config_dict.update(kwargs)
    config_dict['variant'] = variant
    return BioANAConfig(**config_dict)
