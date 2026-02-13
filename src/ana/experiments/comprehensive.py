import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm

from ..models.config import ANAConfig
from ..models.core import ANAModel, BaselineSSM
from ..utils.datasets import CopyTask, AssociativeRecallDataset
from ..utils.plotting import plot_all

class ComparisonRunner:
    def __init__(self, output_dir: str = "results/comprehensive"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(self.output_dir, "experiment.log"),
                            filemode='w')
        self.logger = logging.getLogger("ANA_Comparison")
        # Add console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Running on device: {self.device}")

    def train_model(self, model: nn.Module, train_loader: DataLoader, max_steps: int = 1000, lr: float = 1e-3,
                    callback: Optional[Callable[[int, float, nn.Module], None]] = None) -> List[float]:
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []

        iterator = iter(train_loader)

        pbar = tqdm(range(max_steps), desc="Training")

        for step in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            # Unpack batch (x, y, mask)
            x, y, mask = batch
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

            optimizer.zero_grad()
            logits, _ = model(x)

            # Loss only on masked positions
            active_pos = mask.bool()
            if active_pos.any():
                loss = F.cross_entropy(logits[active_pos], y[active_pos])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                current_loss = loss.item()
                losses.append(current_loss)
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})
            else:
                losses.append(0.0)

            if step % 100 == 0:
                self.logger.info(f"Step {step}/{max_steps}: Loss {np.mean(losses[-100:] if len(losses) > 0 else [0]):.4f}")

            if callback:
                callback(step, losses[-1] if losses else 0.0, model)

        return losses

    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
        model.to(self.device)
        model.eval()
        total_loss = 0
        total_active = 0
        correct = 0
        total_predictions = 0

        for batch in val_loader:
            x, y, mask = batch
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

            logits, info_log = model(x, return_info=True) # Get info log just in case, but ignore for metrics

            active_pos = mask.bool()
            if active_pos.any():
                active_logits = logits[active_pos]
                active_targets = y[active_pos]

                loss = F.cross_entropy(active_logits, active_targets, reduction='sum')
                total_loss += loss.item()
                total_active += active_targets.numel()

                preds = active_logits.argmax(dim=-1)
                correct += (preds == active_targets).sum().item()
                total_predictions += active_targets.numel()

        avg_loss = total_loss / total_active if total_active > 0 else float('inf')
        accuracy = correct / total_predictions if total_predictions > 0 else 0.0

        return avg_loss, accuracy

    def save_visualization(self, model: nn.Module, dataset, task_name: str, tag: str):
        """Runs one batch and saves visualization plots."""
        model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        x, y, mask = next(iter(loader))
        x = x.to(self.device)

        logits, info_log = model(x, return_info=True)

        plot_dir = os.path.join(self.output_dir, "plots", task_name)
        os.makedirs(plot_dir, exist_ok=True)

        plot_all(info_log, plot_dir, prefix=f"{tag}_")
        self.logger.info(f"Saved visualizations to {plot_dir}")

    def run_scaling_benchmark(self, seq_lens: List[int] = [128, 512, 1024, 2048, 4096], steps_per_len: int = 500, quick: bool = False):
        self.logger.info("Starting Scaling Benchmark...")
        if quick:
            seq_lens = seq_lens[:2] # Only run first two
            steps_per_len = 50

        results = {'seq_lens': [], 'ana': [], 'baseline': []}

        base_config = ANAConfig(d_model=64, state_dim=64, num_layers=2, track_count=2, use_hololink=True, use_controller=True)

        for sl in seq_lens:
            results['seq_lens'].append(sl)
            self.logger.info(f"Testing Sequence Length: {sl}")

            # Create Task (Associative Recall as proxy for memory capacity)
            # Use smaller vocab for stability, fewer pairs for speed? No, standard settings.
            # Scaling benchmark usually implies same task difficulty, just longer sequence (more noise).
            # For CopyTask: length = sl // 2 (approx).
            # Let's use CopyTask as it's cleaner for scaling memory length directly.

            copy_len = (sl - 2) // 2
            task = CopyTask(num_samples=1000, seq_len=copy_len, vocab_size=40)
            train_loader = DataLoader(task, batch_size=8, shuffle=True)
            val_loader = DataLoader(task, batch_size=8, shuffle=False)

            # Train ANA
            ana = ANAModel(base_config)
            self.train_model(ana, train_loader, max_steps=steps_per_len)
            _, ana_acc = self.evaluate_model(ana, val_loader)

            # Train Baseline
            baseline = BaselineSSM(base_config)
            self.train_model(baseline, train_loader, max_steps=steps_per_len)
            _, base_acc = self.evaluate_model(baseline, val_loader)

            self.logger.info(f"SeqLen {sl}: ANA Acc={ana_acc:.4f}, Baseline Acc={base_acc:.4f}")
            results['ana'].append(ana_acc)
            results['baseline'].append(base_acc)

            # Save visualization for ANA at max length
            if sl == seq_lens[-1]:
                self.save_visualization(ana, task, "scaling", f"len{sl}")

        with open(os.path.join(self.output_dir, "scaling_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_ablation_study(self, steps: int = 1000, quick: bool = False):
        self.logger.info("Starting Ablation Study...")
        if quick: steps = 100

        configs = {
            'Full ANA': ANAConfig(use_hololink=True, use_controller=True, use_parallel_scan=True),
            'No HoloLink': ANAConfig(use_hololink=False, use_controller=True, use_parallel_scan=True),
            'No Controller': ANAConfig(use_hololink=True, use_controller=False, use_parallel_scan=True),
            'Sequential': ANAConfig(use_hololink=True, use_controller=True, use_parallel_scan=False, max_thinking_steps=1)
        }

        # Use Associative Recall for ablation (needs memory + reasoning)
        task = AssociativeRecallDataset(num_samples=1000, vocab_size=40, num_pairs=8, noise_len=64)
        train_loader = DataLoader(task, batch_size=8, shuffle=True)
        val_loader = DataLoader(task, batch_size=8, shuffle=False)

        results = {}

        for name, config in configs.items():
            self.logger.info(f"Testing Config: {name}")
            model = ANAModel(config)
            self.train_model(model, train_loader, max_steps=steps)
            loss, acc = self.evaluate_model(model, val_loader)
            results[name] = {'loss': loss, 'accuracy': acc}
            self.logger.info(f"{name}: Acc={acc:.4f}")

            if name == 'Full ANA':
                self.save_visualization(model, task, "ablation", "full_ana")

        with open(os.path.join(self.output_dir, "ablation_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_throughput_benchmark(self, seq_lens: List[int] = [128, 1024, 4096], quick: bool = False):
        self.logger.info("Starting Throughput Benchmark...")
        if quick: seq_lens = seq_lens[:1]

        config = ANAConfig(d_model=64, state_dim=64, num_layers=2)
        model = ANAModel(config).to(self.device)
        model.eval()

        results = {}

        for sl in seq_lens:
            x = torch.randint(0, 40, (1, sl)).to(self.device) # Batch 1

            # Warmup
            for _ in range(5):
                _ = model(x)

            # Measure
            start_time = time.time()
            num_iters = 20
            with torch.no_grad():
                for _ in range(num_iters):
                    _ = model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            duration = end_time - start_time
            tok_per_sec = (sl * num_iters) / duration

            # Memory (approximate peak if cuda)
            peak_mem = 0
            if self.device == "cuda":
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2) # MB

            self.logger.info(f"SeqLen {sl}: {tok_per_sec:.2f} tok/s, Peak Mem: {peak_mem:.2f} MB")
            results[sl] = {'tok_per_sec': tok_per_sec, 'peak_mem_mb': peak_mem}

        with open(os.path.join(self.output_dir, "throughput_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def generate_report(self):
        """Generates a Markdown report from results."""
        report_path = os.path.join(self.output_dir, "REPORT.md")
        with open(report_path, 'w') as f:
            f.write(f"# ANA Comprehensive Analysis Report\n")
            f.write(f"Date: {self.timestamp}\n\n")

            # Scaling
            scaling_path = os.path.join(self.output_dir, "scaling_results.json")
            if os.path.exists(scaling_path):
                with open(scaling_path, 'r') as j:
                    res = json.load(j)
                f.write("## Scaling Benchmark (Copy Task)\n")
                f.write("| Seq Len | ANA Acc | Baseline Acc |\n|---|---|---|\n")
                for sl, ana, base in zip(res['seq_lens'], res['ana'], res['baseline']):
                    f.write(f"| {sl} | {ana:.4f} | {base:.4f} |\n")
                f.write("\n")

            # Ablation
            ablation_path = os.path.join(self.output_dir, "ablation_results.json")
            if os.path.exists(ablation_path):
                with open(ablation_path, 'r') as j:
                    res = json.load(j)
                f.write("## Ablation Study (Associative Recall)\n")
                f.write("| Config | Accuracy | Loss |\n|---|---|---|\n")
                for name, metrics in res.items():
                    f.write(f"| {name} | {metrics['accuracy']:.4f} | {metrics['loss']:.4f} |\n")
                f.write("\n")

            # Throughput
            tp_path = os.path.join(self.output_dir, "throughput_results.json")
            if os.path.exists(tp_path):
                with open(tp_path, 'r') as j:
                    res = json.load(j)
                f.write("## Throughput Benchmark\n")
                f.write("| Seq Len | Tokens/Sec | Peak Mem (MB) |\n|---|---|---|\n")
                for sl, metrics in res.items():
                    f.write(f"| {sl} | {metrics['tok_per_sec']:.2f} | {metrics['peak_mem_mb']:.2f} |\n")

        self.logger.info(f"Report generated at {report_path}")
