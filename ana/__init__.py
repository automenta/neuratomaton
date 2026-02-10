import warnings
from .config_v2 import ANAv2Config
from .model_v3 import ANAv2Model
from .bio_ana import BioANAModel, create_bio_ana

__all__ = [
    # Current (Recommended)
    'ANAConfig',
    'TrainingConfig', 
    'DataConfig',
    'ANAModel',
    'BaselineSSM',
    'LinearRecurrentUnit',
    'HyperController',
    'HoloLink',
    'AssociativeRecallDataset',
    'TextDataset',
    'CopyTaskDataset',
    'ReverseTaskDataset',
    'AdditionTaskDataset',
    'SortTaskDataset',
    'MultiQueryARDataset',
    'InductionHeadDataset',
    'LongContextARDataset',
    'run_eval_task',
    'run_all_evals',
    'run_training',
    'evaluate',
    'BenchmarkEvaluator',
    # v3 (Current)
    'ANAConfig',
    'ANAv2Config',
    'ANAConfig',
    'ANAv2Model',
    # Bio-ANA (New)
    'BioANAModel',
    'create_bio_ana',
]

# For backward compatibility (deprecated)
try:
    from .config import ANAConfig
    from .models import ANAModel, BaselineSSM, LinearRecurrentUnit, HyperController, HoloLink
    from .data import AssociativeRecallDataset, TextDataset
    from .eval import CopyTaskDataset, ReverseTaskDataset, AdditionTaskDataset, SortTaskDataset, run_eval_task, run_all_evals
    from .train import run_training, evaluate
    from .benchmarks import BenchmarkEvaluator, MultiQueryARDataset, InductionHeadDataset, LongContextARDataset
except ImportError:
    pass

from .data import AssociativeRecallDataset, TextDataset
from .eval import CopyTaskDataset, ReverseTaskDataset, AdditionTaskDataset, SortTaskDataset, run_eval_task, run_all_evals
from .train import run_training, evaluate
from .benchmarks import BenchmarkEvaluator
