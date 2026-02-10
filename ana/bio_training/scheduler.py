import math
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    warmup_steps: int = 100
    total_steps: int = 10000
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    
    relaxation_warmup_steps: int = 50
    relaxation_start: int = 12
    relaxation_end: int = 7


class CosineScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
    
    def get_lr(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.max_lr * step / self.config.warmup_steps
        
        progress = (step - self.config.warmup_steps) / (
            self.config.total_steps - self.config.warmup_steps
        )
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine_factor
    
    def get_relaxation_iters(self, step: int) -> int:
        if step < self.config.relaxation_warmup_steps:
            return self.config.relaxation_start
        
        progress = (step - self.config.relaxation_warmup_steps) / (
            self.config.total_steps - self.config.relaxation_warmup_steps
        )
        
        current = self.config.relaxation_start + (
            self.config.relaxation_end - self.config.relaxation_start
        ) * progress
        
        return max(self.config.relaxation_end, int(current))


class LinearScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
    
    def get_lr(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.max_lr * step / self.config.warmup_steps
        
        progress = (step - self.config.warmup_steps) / (
            self.config.total_steps - self.config.warmup_steps
        )
        
        return self.config.max_lr - (
            self.config.max_lr - self.config.min_lr
        ) * progress
    
    def get_relaxation_iters(self, step: int) -> int:
        if step < self.config.relaxation_warmup_steps:
            return self.config.relaxation_start
        
        progress = (step - self.config.relaxation_warmup_steps) / (
            self.config.total_steps - self.config.relaxation_warmup_steps
        )
        
        current = self.config.relaxation_start + (
            self.config.relaxation_end - self.config.relaxation_start
        ) * progress
        
        return max(self.config.relaxation_end, int(current))


class ConstantScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
    
    def get_lr(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.max_lr * step / self.config.warmup_steps
        return self.config.max_lr
    
    def get_relaxation_iters(self, step: int) -> int:
        return self.config.relaxation_end


def create_scheduler(
    scheduler_type: str = 'cosine',
    config: Optional[SchedulerConfig] = None,
) -> object:
    config = config or SchedulerConfig()
    
    schedulers = {
        'cosine': CosineScheduler,
        'linear': LinearScheduler,
        'constant': ConstantScheduler,
    }
    
    scheduler_class = schedulers.get(scheduler_type, CosineScheduler)
    return scheduler_class(config)


class CurriculumScheduler:
    def __init__(
        self,
        stage_configs: Optional[dict] = None,
    ):
        self.stage_configs = stage_configs or {
            '0': {
                'max_steps': 2000,
                'accuracy_threshold': 0.98,
                'patience': 3,
            },
            '1': {
                'max_steps': 3000,
                'accuracy_threshold': 0.90,
                'patience': 3,
            },
            '2': {
                'max_steps': 5000,
                'accuracy_threshold': 0.85,
                'patience': 3,
            },
        }
        
        self.accuracy_history = {'0': [], '1': [], '2': []}
    
    def should_advance(
        self,
        stage: str,
        step: int,
        recent_accuracy: float,
    ) -> bool:
        config = self.stage_configs.get(stage, self.stage_configs['0'])
        
        self.accuracy_history[stage].append(recent_accuracy)
        
        history = self.accuracy_history[stage]
        patience = config['patience']
        
        if len(history) >= patience:
            recent = history[-patience:]
            if all(acc >= config['accuracy_threshold'] for acc in recent):
                return True
        
        if step >= config['max_steps']:
            avg_recent = sum(history[-patience:]) / min(len(history), patience)
            if avg_recent >= config['accuracy_threshold'] * 0.95:
                return True
        
        return False
    
    def next_stage(self, current_stage: str) -> Optional[str]:
        stage_order = ['0', '1', '2']
        
        try:
            idx = stage_order.index(current_stage)
            if idx < len(stage_order) - 1:
                return stage_order[idx + 1]
        except ValueError:
            pass
        
        return None
