#!/usr/bin/env python3
"""
ANA v2 Package

Adaptive Neural Automaton - The Beast
"""

from .core import (
    ANAConfig,
    GumbelSoftmax,
    HolographicMemory,
    ProgramStack,
    Interpreter,
    LinearRecurrentTrack,
    ANALayer,
    ANAModel
)

from .train import (
    Trainer,
    SimpleDataset
)

from .tasks import (
    Task,
    generate_copy_task,
    generate_reverse_task,
    generate_associative_recall_task,
    generate_arithmetic_task,
    generate_sorting_task,
    evaluate_task,
    get_all_tasks
)

__version__ = "2.0.0"
__all__ = [
    "ANAConfig",
    "GumbelSoftmax",
    "HolographicMemory",
    "ProgramStack",
    "Interpreter",
    "LinearRecurrentTrack",
    "ANALayer",
    "ANAModel",
    "Trainer",
    "SimpleDataset",
    "Task",
    "generate_copy_task",
    "generate_reverse_task",
    "generate_associative_recall_task",
    "generate_arithmetic_task",
    "generate_sorting_task",
    "evaluate_task",
    "get_all_tasks"
]
