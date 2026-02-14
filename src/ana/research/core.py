import os
import logging
import json
import shutil
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional

class ResultManager:
    """Manages experiment results, logging, and artifacts."""
    def __init__(self, experiment_name: str, phase: int):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join("results", f"phase{phase}_{experiment_name}_{self.timestamp}")
        os.makedirs(self.base_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"ANA_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.base_dir, "experiment.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # Also log to console if not already doing so
        if not self.logger.hasHandlers() or len(self.logger.handlers) == 1:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(ch)

    def log(self, message: str):
        self.logger.info(message)

    def save_json(self, filename: str, data: Dict[str, Any]):
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self.log(f"Saved JSON: {filepath}")

    def save_report(self, filename: str, content: str):
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        self.log(f"Saved Report: {filepath}")

    @property
    def output_dir(self) -> str:
        return self.base_dir


class ExperimentBase(ABC):
    """Base class for all experiments."""
    def __init__(self, config: Any = None):
        self.config = config
        self._results = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def phase(self) -> int:
        pass

    @property
    def results(self) -> ResultManager:
        if self._results is None:
            self._results = ResultManager(self.name, self.phase)
        return self._results

    def setup(self):
        """Override for setup logic."""
        self.results.log(f"Setting up experiment: {self.name}")

    @abstractmethod
    def execute(self, **kwargs):
        """Core experiment logic."""
        pass

    def teardown(self):
        """Override for cleanup logic."""
        self.results.log(f"Tearing down experiment: {self.name}")

    def run(self, **kwargs):
        """Run the full experiment lifecycle."""
        try:
            self.setup()
            self.execute(**kwargs)
        except Exception as e:
            if self._results:
                self.results.logger.error(f"Experiment failed: {e}", exc_info=True)
            raise e
        finally:
            self.teardown()


class ExperimentRegistry:
    """Registry for discovering and running experiments."""
    _experiments: Dict[int, Dict[str, Type[ExperimentBase]]] = {}

    @classmethod
    def register(cls, phase: int, name: str):
        def wrapper(experiment_cls: Type[ExperimentBase]):
            if phase not in cls._experiments:
                cls._experiments[phase] = {}
            cls._experiments[phase][name] = experiment_cls
            return experiment_cls
        return wrapper

    @classmethod
    def get(cls, phase: int, name: str) -> Optional[Type[ExperimentBase]]:
        return cls._experiments.get(phase, {}).get(name)

    @classmethod
    def list_phases(cls):
        return sorted(cls._experiments.keys())

    @classmethod
    def list_experiments(cls, phase: int):
        return sorted(cls._experiments.get(phase, {}).keys())


def load_config_overrides(config: Any, overrides_str: str) -> Any:
    """
    Parses a string like "d_model=128,dropout=0.5" and updates the config object.
    Returns a new config object (or the modified one).
    """
    if not overrides_str:
        return config

    pairs = overrides_str.split(',')
    for pair in pairs:
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Simple type inference
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        else:
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass # Keep as string

        if hasattr(config, key):
            setattr(config, key, value)

    return config
