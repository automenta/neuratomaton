import os
import json
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from datetime import datetime
from ana.config import ANAConfig
import argparse
import inspect

class ExperimentRegistry:
    _registry: Dict[str, Dict[str, Type["ExperimentBase"]]] = {}

    @classmethod
    def register(cls, phase: int, name: str):
        def decorator(experiment_cls):
            if phase not in cls._registry:
                cls._registry[phase] = {}
            cls._registry[phase][name] = experiment_cls
            return experiment_cls
        return decorator

    @classmethod
    def get(cls, phase: int, name: str) -> Optional[Type["ExperimentBase"]]:
        return cls._registry.get(phase, {}).get(name)

    @classmethod
    def list_experiments(cls):
        return cls._registry

class ResultManager:
    def __init__(self, experiment_name: str, phase: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"results/phase{phase}_{experiment_name}/{timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)
        self.log_file = os.path.join(self.base_dir, "experiment.log")
        self.metrics = {}

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        with open(self.log_file, "a") as f:
            f.write(formatted_message + "\n")

    def save_json(self, filename: str, data: Dict[str, Any]):
        path = os.path.join(self.base_dir, filename)
        # Convert non-serializable objects to strings
        def default_converter(o):
            if isinstance(o, (torch.Tensor, np.ndarray)):
                return o.tolist()
            return str(o)

        with open(path, "w") as f:
            json.dump(data, f, indent=4, default=default_converter)
        self.log(f"Saved JSON to {path}")

    def save_plot(self, filename: str, figure=None):
        path = os.path.join(self.base_dir, filename)
        if figure:
            figure.savefig(path)
        else:
            plt.savefig(path)
        plt.close()
        self.log(f"Saved plot to {path}")

    def save_report(self, filename: str, content: str):
        path = os.path.join(self.base_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        self.log(f"Saved report to {path}")

    def get_path(self, filename: str) -> str:
        return os.path.join(self.base_dir, filename)

class ExperimentBase(ABC):
    def __init__(self, config: ANAConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.results = ResultManager(self.name, self.phase)
        self.set_seed(42)  # Default seed

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def phase(self) -> int:
        pass

    def set_seed(self, seed: int):
        self.results.log(f"Setting seed to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup(self):
        """Override to perform setup actions before execution."""
        pass

    @abstractmethod
    def execute(self):
        """Main experiment logic."""
        pass

    def teardown(self):
        """Override to perform cleanup after execution."""
        pass

    def run(self):
        self.results.log(f"Starting experiment: {self.name} (Phase {self.phase})")
        self.results.log(f"Config: {self.config}")

        try:
            self.setup()
            self.execute()
        except Exception as e:
            self.results.log(f"Error during execution: {str(e)}")
            raise e
        finally:
            self.teardown()
            self.results.log("Experiment completed.")

def load_config_overrides(base_config: ANAConfig, override_str: str) -> ANAConfig:
    """
    Parses a string like "d_model=128,dropout=0.1" and updates the config.
    """
    if not override_str:
        return base_config

    overrides = override_str.split(",")
    for override in overrides:
        key, value = override.split("=")
        key = key.strip()
        value = value.strip()

        if hasattr(base_config, key):
            # Infer type from default value
            default_val = getattr(base_config, key)
            target_type = type(default_val)

            try:
                if target_type == bool:
                    # Handle boolean explicitly
                    if value.lower() in ('true', '1', 'yes'):
                        parsed_val = True
                    elif value.lower() in ('false', '0', 'no'):
                        parsed_val = False
                    else:
                        raise ValueError(f"Invalid boolean value: {value}")
                else:
                    parsed_val = target_type(value)

                setattr(base_config, key, parsed_val)
            except ValueError:
                print(f"Warning: Could not convert '{value}' to {target_type} for key '{key}'. Ignoring.")
        else:
            print(f"Warning: Config key '{key}' not found in ANAConfig. Ignoring.")

    return base_config
