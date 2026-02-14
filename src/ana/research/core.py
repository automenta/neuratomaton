import os
import logging
import json
import shutil
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional

class ResultManager:
    """Manages experiment results, logging, and artifacts."""
    def __init__(self, experiment_name: str, phase: int, study_name: str = "main"):
        # Directory structure: results/{study_name}/phase{phase}_{experiment_name}
        # This removes timestamps from directory names, enabling accumulation and resumption.
        self.base_dir = os.path.join("results", study_name, f"phase{phase}_{experiment_name}")
        os.makedirs(self.base_dir, exist_ok=True)

        # Setup logging - Append mode to keep history
        self.logger = logging.getLogger(f"ANA_{experiment_name}")
        self.logger.setLevel(logging.INFO)

        # Ensure proper file handler exists for the current base_dir
        log_path = os.path.join(self.base_dir, "experiment.log")
        has_file_handler = False

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                # Check if it points to the same file (normalized path)
                if os.path.normpath(handler.baseFilename) == os.path.normpath(os.path.abspath(log_path)):
                    has_file_handler = True
                    break

        if not has_file_handler:
            fh = logging.FileHandler(log_path, mode='a')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

        # Ensure console handler exists
        has_console_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        if not has_console_handler:
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

    def has_file(self, filename: str) -> bool:
        return os.path.exists(os.path.join(self.base_dir, filename))

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        filepath = os.path.join(self.base_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load JSON {filename}: {e}")
                return None
        return None

    @property
    def output_dir(self) -> str:
        return self.base_dir


class ExperimentBase(ABC):
    """Base class for all experiments."""
    def __init__(self, config: Any = None):
        self.config = config
        self._results = None
        self._study_name = "main"

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
            self._results = ResultManager(self.name, self.phase, self._study_name)
        return self._results

    def set_study_name(self, study_name: str):
        """Set the study name for persistent results."""
        self._study_name = study_name
        # Reset results manager so it re-initializes with new study name
        self._results = None

    def setup(self):
        """Override for setup logic."""
        self.results.log(f"Setting up experiment: {self.name} (Study: {self._study_name})")

    @abstractmethod
    def execute(self, **kwargs):
        """Core experiment logic."""
        pass

    def teardown(self):
        """Override for cleanup logic."""
        self.results.log(f"Tearing down experiment: {self.name}")

    def run(self, study_name: str = "main", **kwargs):
        """Run the full experiment lifecycle."""
        self.set_study_name(study_name)
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
