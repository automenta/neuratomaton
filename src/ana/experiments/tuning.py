import optuna
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from ..models.config import ANAConfig
from ..models.core import ANAModel
from ..utils.datasets import AssociativeRecallDataset, CopyTask
from .comprehensive import ComparisonRunner

class HyperparameterTuner:
    """
    Hyperparameter Tuning for ANA Architecture using Optuna.
    """
    def __init__(self, runner: ComparisonRunner):
        self.runner = runner
        self.logger = runner.logger
        self.best_config = None
        self.best_value = float('-inf')

    def tune(self, task_name: str, n_trials: int = 20, quick: bool = False) -> Optional[ANAConfig]:
        """
        Run hyperparameter tuning for a specific task.
        """
        self.logger.info(f"Starting Hyperparameter Tuning for: {task_name} ({n_trials} trials)")

        study_name = f"ana_{task_name}_{self.runner.timestamp}"
        study = optuna.create_study(direction="maximize", study_name=study_name)

        # Define objective based on task
        if task_name == "sanity":
            objective = self._objective_sanity
        elif task_name == "scaling":
            objective = self._objective_scaling
        else:
            self.logger.error(f"Unknown tuning task: {task_name}")
            return None

        # Wrap objective to pass `quick` flag
        def wrapped_objective(trial):
            return objective(trial, quick)

        try:
            study.optimize(wrapped_objective, n_trials=n_trials)
        except Exception as e:
            self.logger.error(f"Tuning failed with error: {e}")
            return None

        self.logger.info("Tuning Complete.")
        self.logger.info(f"Best Trial: {study.best_trial.number}")
        self.logger.info(f"Best Value: {study.best_value}")
        self.logger.info(f"Best Params: {study.best_params}")

        # Construct best config
        best_params = study.best_params
        best_config = self._params_to_config(best_params)

        self.best_config = best_config
        self.best_value = study.best_value

        return best_config

    def _params_to_config(self, params: Dict[str, Any]) -> ANAConfig:
        """Convert optuna params to ANAConfig, filling defaults."""
        # Start with a sensible default base
        base_config = ANAConfig(
            vocab_size=40,
            use_hololink=True,
            use_controller=True,
            use_parallel_scan=True
        )

        # Update with tuned params
        for k, v in params.items():
            if hasattr(base_config, k):
                setattr(base_config, k, v)

        # Ensure dependencies (e.g. key_dim <= d_model)
        if base_config.key_dim > base_config.d_model:
            base_config.key_dim = base_config.d_model // 2

        return base_config

    def _objective_sanity(self, trial, quick: bool):
        """
        Objective: Maximize Accuracy on Associative Recall.
        Search Space: Learning Rate, Model Size, Init Scale (implicit).
        """
        # Search Space
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        d_model = trial.suggest_categorical("d_model", [32, 64])
        state_dim = trial.suggest_categorical("state_dim", [32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        track_count = trial.suggest_int("track_count", 1, 4)

        config = ANAConfig(
            d_model=d_model,
            state_dim=state_dim,
            num_layers=num_layers,
            track_count=track_count,
            key_dim=d_model // 2,
            vocab_size=40,
            use_hololink=True,
            use_controller=True,
            use_parallel_scan=True
        )

        # Task
        steps = 50 if quick else 300
        task = AssociativeRecallDataset(num_samples=1000, vocab_size=40, num_pairs=4, noise_len=32)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        # Train
        model = ANAModel(config)

        # Pruning Callback
        def prune_callback(step, loss, model):
            if step % 50 == 0:
                trial.report(loss, step) # Report loss (minimize) but we want accuracy (maximize)
                # Actually, trial.report expects the optimization metric.
                # Let's run a quick eval
                _, acc = self.runner.evaluate_model(model, val_loader)
                trial.report(acc, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        try:
            self.runner.train_model(model, train_loader, max_steps=steps, lr=lr, callback=prune_callback)
            _, acc = self.runner.evaluate_model(model, val_loader)
            return acc
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Failed run (NaNs etc)
            return 0.0

    def _objective_scaling(self, trial, quick: bool):
        """
        Objective: Maximize Accuracy on Long-Sequence Copy Task (N=512).
        Search Space: State Dim, Key Dim, Tracks.
        """
        # Search Space
        d_model = trial.suggest_categorical("d_model", [64])
        state_dim = trial.suggest_categorical("state_dim", [64, 128])
        key_dim = trial.suggest_categorical("key_dim", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 2, 3)
        track_count = trial.suggest_int("track_count", 2, 4)

        config = ANAConfig(
            d_model=d_model,
            state_dim=state_dim,
            num_layers=num_layers,
            track_count=track_count,
            key_dim=key_dim,
            vocab_size=40,
            use_hololink=True,
            use_controller=True,
            use_parallel_scan=True
        )

        # Task: Copy Length 250 (Total ~500)
        steps = 50 if quick else 500
        seq_len = 100 # Moderate length for tuning speed
        task = CopyTask(num_samples=1000, seq_len=seq_len, vocab_size=40)
        train_loader = DataLoader(task, batch_size=8, shuffle=True)
        val_loader = DataLoader(task, batch_size=8, shuffle=False)

        model = ANAModel(config)

        try:
            self.runner.train_model(model, train_loader, max_steps=steps, lr=1e-3) # Fixed LR for architecture search
            _, acc = self.runner.evaluate_model(model, val_loader)
            return acc
        except Exception:
            return 0.0
