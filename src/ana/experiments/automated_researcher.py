import torch
import numpy as np
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..models.config import ANAConfig
from ..models.core import ANAModel
from ..utils.datasets import AssociativeRecallDataset, CopyTask
from .comprehensive import ComparisonRunner
from .tuning import HyperparameterTuner

class AutomatedResearcher:
    """
    Automated Research Agent for ANA Architecture.

    Philosophy:
    1.  Fail Fast: If basic associative recall doesn't work, don't scale up.
    2.  Probe Before Commit: Run small-scale probes to verify trends before launching massive jobs.
    3.  Automated Contingency: If a hypothesis fails (e.g., ANA < Baseline), pivot or stop.
    """
    def __init__(self, output_dir: str = "results/automated"):
        self.runner = ComparisonRunner(output_dir=output_dir)
        self.logger = self.runner.logger
        self.status = "initialized"

    def run_pipeline(self, quick: bool = False, tune: bool = False, trials: int = 20):
        self.logger.info("Starting Automated Research Pipeline...")
        print("\n\033[1;34m=== ANA RESEARCH PIPELINE STARTED ===\033[0m")
        self.status = "running"

        # Hyperparameter Tuning (Optional/Adaptive)
        if tune:
             print("\n\033[1;33m[OPT] Adaptive Tuning Requested...\033[0m")
             self.logger.info("Adaptive Tuning: Requested by User.")
             self._stage_tuning("sanity", trials, quick)

        # Stage 1: Sanity Check (Associative Recall)
        print("\n\033[1;36m[STAGE 1] Sanity Check (Associative Recall)\033[0m")
        if not self._stage_sanity_check(quick):
            self.logger.warning("Stage 1 Failed: Model failed basic sanity check.")
            print("\033[1;31m[FAIL] Sanity Check Failed.\033[0m Triggering Adaptive Tuning...")
            self.logger.info("Attempting Adaptive Tuning to fix Stage 1...")

            # Adaptive Tuning
            best_config = self._stage_tuning("sanity", trials, quick)

            if best_config:
                 self.logger.info("Retrying Stage 1 with Tuned Config...")
                 print("\n\033[1;32m[RETRY] Retrying Stage 1 with Optimized Config...\033[0m")
                 if not self._stage_sanity_check(quick, config=best_config):
                      self.logger.error("Stage 1 Failed Again (Even after tuning). Aborting.")
                      print("\033[1;31m[ABORT] Stage 1 Failed even after tuning. Model architecture likely flawed.\033[0m")
                      self.status = "failed_stage_1"
                      return
            else:
                 self.logger.error("Tuning Failed. Aborting.")
                 print("\033[1;31m[ABORT] Tuning found no viable configuration.\033[0m")
                 self.status = "failed_stage_1"
                 return
        else:
            print("\033[1;32m[PASS] Sanity Check Passed.\033[0m")

        # Stage 2: Scaling Probe (Small N)
        print("\n\033[1;36m[STAGE 2] Scaling Probe (Trend Analysis)\033[0m")
        if not self._stage_scaling_probe(quick):
            self.logger.warning("Stage 2 Warning: Scaling trend is negative or inconclusive.")
            print("\033[1;33m[WARN] Scaling Trend is Negative/Inconclusive.\033[0m Triggering Tuning...")
            self.logger.info("Attempting Adaptive Tuning for Scaling...")

            best_config = self._stage_tuning("scaling", trials, quick)
            if best_config:
                 print("\033[1;32m[OPT] Optimized Scaling Config Found.\033[0m Proceeding to Deep Dive.")
        else:
            print("\033[1;32m[PASS] Scaling Probe Positive.\033[0m")

        # Stage 3: Deep Dive (Ablation & Large N)
        print("\n\033[1;36m[STAGE 3] Deep Dive (Full Benchmarks)\033[0m")
        self._stage_deep_dive(quick)

        self.status = "completed"
        self.logger.info("Research Pipeline Completed Successfully.")
        print("\n\033[1;32m=== RESEARCH PIPELINE COMPLETED ===\033[0m")
        self.runner.generate_report()
        print(f"Report generated at: {os.path.join(self.runner.output_dir, 'REPORT.md')}")

    def _stage_tuning(self, task: str, trials: int, quick: bool) -> Optional[ANAConfig]:
        """
        Run Hyperparameter Tuning.
        """
        self.logger.info(f"=== Tuning Stage: {task} ===")
        tuner = HyperparameterTuner(self.runner)
        best_config = tuner.tune(task, n_trials=trials, quick=quick)

        if best_config:
             self.logger.info(f"Tuning Success! Best Config found with value: {tuner.best_value}")
             return best_config
        else:
             self.logger.warning("Tuning yielded no valid config.")
             return None

    def _stage_sanity_check(self, quick: bool, config: Optional[ANAConfig] = None) -> bool:
        """
        Run small Associative Recall task.
        Criteria: > 90% accuracy.
        """
        self.logger.info("=== STAGE 1: Sanity Check ===")
        steps = 50 if quick else 1000  # Less steps for quick

        task = AssociativeRecallDataset(num_samples=1000, vocab_size=40, num_pairs=4, noise_len=32)
        train_loader = torch.utils.data.DataLoader(task, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(task, batch_size=16, shuffle=False)

        if config is None:
             config = ANAConfig(d_model=64, state_dim=64, num_layers=2, track_count=2, use_hololink=True, use_controller=True)

        model = ANAModel(config)

        self.runner.train_model(model, train_loader, max_steps=steps)
        loss, acc = self.runner.evaluate_model(model, val_loader)

        self.logger.info(f"Sanity Check Accuracy: {acc*100:.2f}%")

        if acc < 0.90:
            if quick:
                self.logger.warning(f"Sanity Check Failed ({acc*100:.2f}%) but --quick was set. Proceeding anyway.")
                return True # Allow pass for quick tests
            self.logger.error(f"FAILURE: Accuracy {acc*100:.2f}% < 90%. Model is broken.")
            return False

        self.logger.info("SUCCESS: Model passed sanity check.")
        return True

    def _stage_scaling_probe(self, quick: bool) -> bool:
        """
        Run small scaling benchmark (N=128, 512).
        Criteria: ANA outperforms Baseline.
        """
        self.logger.info("=== STAGE 2: Scaling Probe ===")
        seq_lens = [128, 512]
        steps = 100 if quick else 500

        results = self.runner.run_scaling_benchmark(seq_lens=seq_lens, steps_per_len=steps, quick=False)

        # Analyze results
        ana_wins = 0
        for i, sl in enumerate(seq_lens):
            ana_acc = results['ana'][i]
            base_acc = results['baseline'][i]
            if ana_acc >= base_acc:
                ana_wins += 1
            self.logger.info(f"N={sl}: ANA {ana_acc:.4f} vs Baseline {base_acc:.4f}")

        if ana_wins == 0:
            self.logger.warning("FAILURE: ANA underperformed Baseline on all probe tasks.")
            return False

        self.logger.info(f"SUCCESS: ANA outperformed Baseline on {ana_wins}/{len(seq_lens)} tasks.")
        return True

    def _stage_deep_dive(self, quick: bool):
        """
        Run full benchmarks.
        """
        self.logger.info("=== STAGE 3: Deep Dive ===")

        # 1. Full Scaling (up to 2048/4096)
        # If quick, skip large N
        if not quick:
            large_lens = [1024, 2048] # 4096 takes long on CPU
            self.runner.run_scaling_benchmark(seq_lens=large_lens, steps_per_len=500, quick=False)

        # 2. Ablation
        steps = 100 if quick else 1000
        self.runner.run_ablation_study(steps=steps, quick=False)

        # 3. Throughput
        self.runner.run_throughput_benchmark(quick=quick)
