import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..models.config import ANAConfig
from ..models.core import ANAModel, BaselineSSM
from ..models.baselines import TransformerBaseline, LSTMBaseline
from ..utils.datasets import (
    InductionHeadTask, CopyTask, PointerChainTask,
    MultiQueryAssociativeRecall, AssociativeRecallDataset
)
from .comprehensive import ComparisonRunner

class DiscoveryEngine(ComparisonRunner):
    """
    The Discovery Engine scientifically reveals ANA's potential by:
    1. Benchmarking against strong baselines (Transformer, LSTM).
    2. Optimizing hyperparameters using Bayesian search (Optuna).
    3. Analysing feature attribution (ablation).
    4. Generating a comprehensive scientific report.
    """
    def __init__(self, output_dir: str = "results/discovery"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(self.output_dir, "discovery.log"),
                            filemode='w')
        self.logger = logging.getLogger("ANA_Discovery")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initialized DiscoveryEngine on {self.device}")

        self.best_config = None
        self.baseline_results = {}
        self.tuning_results = {}
        self.attribution_results = {}

    def run_full_suite(self, quick: bool = False):
        self.logger.info("=== STARTING FULL DISCOVERY SUITE ===")
        print("\n\033[1;34m=== STARTING FULL DISCOVERY SUITE ===\033[0m")

        # 1. Baseline Comparison
        self.run_baseline_comparison(quick=quick)

        # 2. Optimization (Science: Finding the best config)
        self.run_optimization(quick=quick)

        # 3. Feature Attribution (Exploit: Understanding why)
        self.run_feature_attribution(quick=quick)

        # 4. Report
        self.generate_scientific_report()
        print(f"\n\033[1;32mDiscovery Suite Complete. Report at: {os.path.join(self.output_dir, 'DISCOVERY_REPORT.md')}\033[0m")

    def run_baseline_comparison(self, quick: bool = False):
        self.logger.info("--- Phase 1: Baseline Comparison ---")
        print("\n\033[1;36m[PHASE 1] Baseline Comparison (ANA vs Transformer vs LSTM)\033[0m")

        # Tasks
        # 1. Induction (Context Learning)
        # 2. Copy (Memory Stability)
        # 3. Reasoning (Pointer Chain)

        steps = 200 if quick else 1000

        tasks = {
            'Induction': InductionHeadTask(num_samples=2000, seq_len=64, vocab_size=40),
            'Copy': CopyTask(num_samples=2000, seq_len=32, vocab_size=40),
            'Reasoning': PointerChainTask(num_samples=2000, vocab_size=40, chain_len=4, noise_pairs=2)
        }

        # Models
        # Ensure fair parameter count? For now, fixed hyperparameters.
        config = ANAConfig(d_model=64, num_layers=2, state_dim=64, track_count=2, use_hololink=True, use_controller=True)

        models_def = {
            'ANA': (ANAModel, config),
            'Transformer': (TransformerBaseline, config),
            'LSTM': (LSTMBaseline, config)
        }

        results = {task_name: {} for task_name in tasks}

        for task_name, dataset in tasks.items():
            print(f"  > Task: {task_name}")
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

            for model_name, (ModelClass, cfg) in models_def.items():
                self.logger.info(f"Training {model_name} on {task_name}...")
                model = ModelClass(cfg).to(self.device)

                start_time = time.time()
                self.train_model(model, train_loader, max_steps=steps)
                train_time = time.time() - start_time

                loss, acc = self.evaluate_model(model, val_loader)

                self.logger.info(f"{model_name} on {task_name}: Acc={acc*100:.2f}%, Time={train_time:.2f}s")

                results[task_name][model_name] = {
                    'accuracy': acc,
                    'loss': loss,
                    'time': train_time,
                    'params': sum(p.numel() for p in model.parameters() if p.requires_grad)
                }

                # Save visualization for ANA
                if model_name == 'ANA':
                    self.save_visualization(model, dataset, "baselines", f"{task_name}_ana")

        self.baseline_results = results
        with open(os.path.join(self.output_dir, "baseline_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    def run_optimization(self, quick: bool = False):
        self.logger.info("--- Phase 2: Hyperparameter Optimization ---")
        print("\n\033[1;36m[PHASE 2] Hyperparameter Optimization (Optuna)\033[0m")

        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not installed. Skipping optimization phase.")
            print("Optuna not found. Skipping.")
            return

        # Objective: Maximize accuracy on a hard task (Multi-Query Associative Recall)
        # while keeping parameters reasonable? Just Accuracy for now.

        steps = 200 if quick else 600
        n_trials = 5 if quick else 30

        task = MultiQueryAssociativeRecall(num_samples=1000, vocab_size=40, num_pairs=8, num_queries=3)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        def objective(trial):
            # Sampling
            d_model = trial.suggest_categorical('d_model', [32, 64, 128])
            track_count = trial.suggest_int('track_count', 1, 4)
            use_hololink = trial.suggest_categorical('use_hololink', [True, False])
            use_controller = trial.suggest_categorical('use_controller', [True, False])
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            config = ANAConfig(
                d_model=d_model,
                state_dim=d_model, # Tie state dim to d_model for simplicity
                num_layers=2,
                track_count=track_count,
                use_hololink=use_hololink,
                use_controller=use_controller,
                learning_rate=lr
            )

            model = ANAModel(config).to(self.device)

            # Pruning Callback
            check_interval = max(steps // 5, 50)

            def pruning_callback(step, loss, model):
                if step > 0 and step % check_interval == 0:
                    _, acc = self.evaluate_model(model, val_loader)
                    trial.report(acc, step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            try:
                self.train_model(model, train_loader, max_steps=steps, lr=lr, callback=pruning_callback)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                raise optuna.exceptions.TrialPruned() # Treat errors as pruned? Or fail.

            # Final Eval
            _, acc = self.evaluate_model(model, val_loader)
            return acc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Best Trial: {study.best_trial.params}")
        print(f"  > Best Accuracy: {study.best_value*100:.2f}%")
        print(f"  > Best Params: {study.best_params}")

        self.best_config = study.best_params
        self.tuning_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': [{'params': t.params, 'value': t.value} for t in study.trials]
        }

        with open(os.path.join(self.output_dir, "tuning_results.json"), 'w') as f:
            json.dump(self.tuning_results, f, indent=2)

    def run_feature_attribution(self, quick: bool = False):
        self.logger.info("--- Phase 3: Feature Attribution (Ablation) ---")
        print("\n\033[1;36m[PHASE 3] Feature Attribution\033[0m")

        # Use best config if available, else default
        base_params = self.best_config if self.best_config else {
            'd_model': 64, 'track_count': 2, 'use_hololink': True, 'use_controller': True
        }

        # Make sure we have the full config object
        # Note: best_config only has tuned params. We need defaults for others.
        def make_config(overrides):
            c = ANAConfig(d_model=64, num_layers=2) # Defaults
            # Update with base params
            for k, v in base_params.items():
                if k == 'lr': continue # skip lr for config object
                setattr(c, k, v)
            # Update with overrides
            for k, v in overrides.items():
                setattr(c, k, v)
            c.state_dim = c.d_model # Ensure tied
            return c

        steps = 200 if quick else 800
        task = PointerChainTask(num_samples=1000, vocab_size=40, chain_len=5, noise_pairs=2)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        variations = {
            'Best Found': {},
            'No HoloLink': {'use_hololink': False},
            'No Controller': {'use_controller': False},
            'Single Track': {'track_count': 1},
            'No Holo/Ctrl': {'use_hololink': False, 'use_controller': False}
        }

        results = {}

        for name, overrides in variations.items():
            config = make_config(overrides)
            self.logger.info(f"Ablation: {name}")

            model = ANAModel(config).to(self.device)
            self.train_model(model, train_loader, max_steps=steps)
            loss, acc = self.evaluate_model(model, val_loader)

            results[name] = acc
            print(f"  > {name}: {acc*100:.2f}%")

        self.attribution_results = results
        with open(os.path.join(self.output_dir, "attribution_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    def generate_scientific_report(self):
        report_path = os.path.join(self.output_dir, "DISCOVERY_REPORT.md")
        with open(report_path, 'w') as f:
            f.write("# ANA Discovery Report: Scientifically Revealing Potential\n\n")
            f.write(f"**Date:** {self.timestamp}\n\n")

            # 1. Baseline Comparison
            f.write("## 1. Baseline Comparison\n")
            f.write("Comparing ANA against conventional architectures (Transformer, LSTM) across key cognitive tasks.\n\n")

            if self.baseline_results:
                # Table header
                tasks = list(self.baseline_results.keys())
                models = list(self.baseline_results[tasks[0]].keys())

                f.write("| Task | Model | Accuracy | Params | Time (s) |\n")
                f.write("| :--- | :--- | :---: | :---: | :---: |\n")

                for task in tasks:
                    for model in models:
                        res = self.baseline_results[task].get(model)
                        if res:
                            f.write(f"| {task} | {model} | {res['accuracy']:.4f} | {res['params']} | {res['time']:.1f} |\n")
                f.write("\n")

                # Analysis
                f.write("**Key Findings:**\n")
                # Simple heuristic analysis
                ana_wins = 0
                total_tasks = len(tasks)
                for task in tasks:
                    ana_acc = self.baseline_results[task]['ANA']['accuracy']
                    trans_acc = self.baseline_results[task]['Transformer']['accuracy']
                    if ana_acc >= trans_acc:
                        ana_wins += 1

                if ana_wins == total_tasks:
                    f.write("- ANA matches or outperforms the Transformer baseline on all tested tasks.\n")
                else:
                    f.write(f"- ANA outperformed the Transformer on {ana_wins}/{total_tasks} tasks.\n")

            # 2. Optimization
            f.write("\n## 2. Hyperparameter Optimization\n")
            f.write("Searching for the ideal architectural configuration using Bayesian Optimization.\n\n")

            if self.tuning_results:
                best = self.tuning_results['best_params']
                val = self.tuning_results['best_value']
                f.write(f"**Best Configuration found (Acc: {val:.4f}):**\n")
                f.write("```json\n")
                f.write(json.dumps(best, indent=2))
                f.write("\n```\n")

                # Parameter Importance (Basic inference)
                f.write("\n**Parameter Sensitivity:**\n")
                trials = self.tuning_results['trials']
                # Just listing top 3
                sorted_trials = sorted(trials, key=lambda x: x['value'], reverse=True)
                f.write("Top 3 Configurations:\n")
                for i, t in enumerate(sorted_trials[:3]):
                    f.write(f"{i+1}. Acc={t['value']:.4f}, Params={t['params']}\n")
            else:
                f.write("Optimization was skipped or failed.\n")

            # 3. Feature Attribution
            f.write("\n## 3. Feature Attribution (Ablation)\n")
            f.write("Quantifying the contribution of ANA's unique components (HoloLink, Controller).\n\n")

            if self.attribution_results:
                f.write("| Configuration | Accuracy | Drop |\n")
                f.write("| :--- | :---: | :---: |\n")

                best_acc = self.attribution_results.get('Best Found', 0)

                for name, acc in self.attribution_results.items():
                    drop = best_acc - acc
                    f.write(f"| {name} | {acc:.4f} | -{drop:.4f} |\n")

            f.write("\n## 4. Conclusion & Recommendations\n")
            f.write("Based on the experiments above, the following 'Recipe' is recommended for training ANA models:\n\n")

            rec_params = self.best_config if self.best_config else {}
            f.write("```python\n")
            f.write("config = ANAConfig(\n")
            for k, v in rec_params.items():
                if k == 'lr': continue
                f.write(f"    {k}={v},\n")
            f.write("    # ... standard params\n")
            f.write(")\n```\n")

        self.logger.info(f"Report written to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    engine = DiscoveryEngine()
    engine.run_full_suite(quick=args.quick)
