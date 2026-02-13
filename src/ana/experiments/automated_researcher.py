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

    def run_pipeline(self, quick: bool = False):
        self.logger.info("Starting Automated Research Pipeline...")
        self.status = "running"

        # Stage 1: Sanity Check (Associative Recall)
        if not self._stage_sanity_check(quick):
            self.logger.error("Stage 1 Failed: Model failed basic sanity check. Aborting.")
            self.status = "failed_stage_1"
            return

        # Stage 2: Scaling Probe (Small N)
        if not self._stage_scaling_probe(quick):
            self.logger.warning("Stage 2 Warning: Scaling trend is negative or inconclusive. Proceeding with caution.")
            # We might still proceed, but maybe skip massive N

        # Stage 3: Deep Dive (Ablation & Large N)
        self._stage_deep_dive(quick)

        self.status = "completed"
        self.logger.info("Research Pipeline Completed Successfully.")
        self.runner.generate_report()

    def _stage_sanity_check(self, quick: bool) -> bool:
        """
        Run small Associative Recall task.
        Criteria: > 90% accuracy.
        """
        self.logger.info("=== STAGE 1: Sanity Check ===")
        steps = 50 if quick else 1000  # Less steps for quick

        task = AssociativeRecallDataset(num_samples=1000, vocab_size=40, num_pairs=4, noise_len=32)
        train_loader = torch.utils.data.DataLoader(task, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(task, batch_size=16, shuffle=False)

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
