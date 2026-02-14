import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import logging
import argparse
from datetime import datetime

from ..models.config import ANAConfig
from ..models.core import ANAModel
from ..utils.datasets import InductionHeadTask, MultiQueryAssociativeRecall, CopyTask, PointerChainTask, AssociativeRecallDataset
from .comprehensive import ComparisonRunner

class PotentialRevealer(ComparisonRunner):
    """
    Advanced experiments to reveal the true potential of ANA.
    Focuses on:
    1. Induction Capability (In-Context Learning)
    2. Length Generalization (Algorithmic Stability)
    3. Cognitive State Dynamics (Thinking vs Remembering)
    4. Reasoning Depth (Adaptive Computation)
    """
    def __init__(self, output_dir: str = "results/potential"):
        super().__init__(output_dir=output_dir)
        self.logger.info("Initialized PotentialRevealer")

    def run_induction_head_experiment(self, quick: bool = False):
        self.logger.info("=== EXPERIMENT: Induction Head Capability ===")

        # Configuration
        config = ANAConfig(
            d_model=64,
            state_dim=64,
            num_layers=2,
            track_count=2,
            use_hololink=True,
            use_controller=True
        )

        # Task
        seq_len = 64
        train_steps = 200 if quick else 1000

        task = InductionHeadTask(num_samples=2000, seq_len=seq_len, vocab_size=40)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        model = ANAModel(config)

        # Train
        self.logger.info("Training on Induction Head Task...")
        self.train_model(model, train_loader, max_steps=train_steps)

        # Evaluate
        avg_loss, acc = self.evaluate_model(model, val_loader)
        self.logger.info(f"Induction Head Accuracy: {acc*100:.2f}%")

        # Visualize
        self.save_visualization(model, task, "induction", "final")

        # Save results
        results = {
            "task": "Induction Head",
            "accuracy": acc,
            "loss": avg_loss,
            "config": str(config)
        }
        with open(os.path.join(self.output_dir, "induction_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_length_generalization_experiment(self, quick: bool = False):
        self.logger.info("=== EXPERIMENT: Length Generalization ===")

        # Train on short, test on long.
        train_len = 64
        test_lens = [64, 128, 256]
        if quick: test_lens = [64, 128]

        steps = 200 if quick else 1000

        # Use CopyTask for length generalization (simplest algorithmic task)
        # Note: CopyTask seq_len is length of *sequence to copy*.
        # Total length is 2 * seq_len + 1.
        # So for train_len=64, we want total sequence length around 64.
        # seq_len = (64 - 1) / 2 approx 31.

        train_seq_len_param = (train_len - 1) // 2
        train_task = CopyTask(num_samples=2000, seq_len=train_seq_len_param, vocab_size=40)
        train_loader = DataLoader(train_task, batch_size=16, shuffle=True)

        config = ANAConfig(
            d_model=64,
            state_dim=64,
            num_layers=2,
            track_count=2,
            use_hololink=True,
            use_controller=True
        )

        model = ANAModel(config)
        self.logger.info(f"Training on Length ~{train_len} (Copy Len {train_seq_len_param})...")
        self.train_model(model, train_loader, max_steps=steps)

        results = {}
        for l in test_lens:
            copy_len = (l - 1) // 2
            test_task = CopyTask(num_samples=500, seq_len=copy_len, vocab_size=40)
            test_loader = DataLoader(test_task, batch_size=16, shuffle=False)

            loss, acc = self.evaluate_model(model, test_loader)
            self.logger.info(f"Testing Length ~{l} (Copy Len {copy_len}): Accuracy {acc*100:.2f}%")
            results[l] = acc

            # Visualize extrapolation
            if l > train_len:
                self.save_visualization(model, test_task, "generalization", f"len{l}")

        with open(os.path.join(self.output_dir, "generalization_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_multi_query_experiment(self, quick: bool = False):
        self.logger.info("=== EXPERIMENT: Multi-Query Associative Recall ===")

        steps = 200 if quick else 1000

        task = MultiQueryAssociativeRecall(num_samples=2000, vocab_size=40, num_pairs=8, num_queries=3)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        config = ANAConfig(
            d_model=64, state_dim=64, num_layers=2, track_count=2,
            use_hololink=True, use_controller=True
        )

        model = ANAModel(config)
        self.logger.info("Training on Multi-Query AR...")
        self.train_model(model, train_loader, max_steps=steps)

        loss, acc = self.evaluate_model(model, val_loader)
        self.logger.info(f"Multi-Query Accuracy: {acc*100:.2f}%")

        self.save_visualization(model, task, "multiquery", "final")

        results = {"accuracy": acc, "loss": loss}
        with open(os.path.join(self.output_dir, "multiquery_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_reasoning_experiment(self, quick: bool = False):
        self.logger.info("=== EXPERIMENT: Reasoning (Thinking Steps) ===")

        steps = 200 if quick else 1000
        chain_len = 5 # Hard task

        task = PointerChainTask(num_samples=2000, vocab_size=40, chain_len=chain_len, noise_pairs=2)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        configs = {
            'No Thinking': ANAConfig(max_thinking_steps=0, d_model=64, num_layers=2),
            'Thinking (K=4)': ANAConfig(max_thinking_steps=4, d_model=64, num_layers=2)
        }

        results = {}

        for name, config in configs.items():
            self.logger.info(f"Training Config: {name}")
            model = ANAModel(config)

            # Use smaller LR for thinking steps?
            self.train_model(model, train_loader, max_steps=steps)
            loss, acc = self.evaluate_model(model, val_loader)

            self.logger.info(f"{name} Accuracy: {acc*100:.2f}%")
            results[name] = acc

            tag = "thinking" if config.max_thinking_steps > 0 else "nothinking"
            self.save_visualization(model, task, "reasoning", tag)

        with open(os.path.join(self.output_dir, "reasoning_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_noise_robustness_experiment(self, quick: bool = False):
        self.logger.info("=== EXPERIMENT: Noise Robustness ===")

        steps = 200 if quick else 1000

        # Train on Clean, Test on Noisy
        clean_task = CopyTask(num_samples=2000, seq_len=32, vocab_size=40)
        # Noisy task: Insert random tokens?
        # Standard CopyTask doesn't support noise injection easily via params unless we subclass.
        # But AssociativeRecall supports noise_len.

        # Let's use AssociativeRecall with varying noise.
        # Train on noise=8, Test on noise=32.

        train_task = MultiQueryAssociativeRecall(num_samples=2000, vocab_size=40, num_pairs=4, num_queries=2)
        # We need a Noisy version of MultiQuery.
        # Actually, let's just use PointerChainTask with varying noise_pairs.

        train_task = PointerChainTask(num_samples=2000, vocab_size=40, chain_len=3, noise_pairs=0)
        test_task = PointerChainTask(num_samples=500, vocab_size=40, chain_len=3, noise_pairs=4)

        train_loader = DataLoader(train_task, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_task, batch_size=16, shuffle=False)

        config = ANAConfig(
            d_model=64, state_dim=64, num_layers=2, track_count=2,
            use_hololink=True, use_controller=True
        )

        model = ANAModel(config)
        self.logger.info("Training on Clean Data (Noise=0)...")
        self.train_model(model, train_loader, max_steps=steps)

        _, clean_acc = self.evaluate_model(model, DataLoader(train_task, batch_size=16, shuffle=False))
        _, noisy_acc = self.evaluate_model(model, test_loader)

        self.logger.info(f"Clean Accuracy: {clean_acc*100:.2f}%")
        self.logger.info(f"Noisy Accuracy (Noise=4 pairs): {noisy_acc*100:.2f}%")

        self.save_visualization(model, test_task, "noise", "high_noise")

        results = {"clean": clean_acc, "noisy": noisy_acc}
        with open(os.path.join(self.output_dir, "noise_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_curriculum_experiment(self, quick: bool = False):
        self.logger.info("=== EXPERIMENT: Curriculum Learning ===")

        # Goal: Train model on increasing difficulty.
        # Task: Associative Recall with increasing number of pairs.

        # Standard: Train directly on Hard (8 pairs).
        # Curriculum: Train on 4 pairs -> 6 pairs -> 8 pairs.

        steps_per_stage = 100 if quick else 400

        config = ANAConfig(d_model=64, state_dim=64, num_layers=2, track_count=2, use_hololink=True, use_controller=True)

        # 1. Baseline: Direct Training
        self.logger.info("Baseline: Training directly on Hard Task (8 pairs)...")
        model_base = ANAModel(config)
        task_hard = AssociativeRecallDataset(num_samples=2000, vocab_size=40, num_pairs=8, noise_len=16)
        train_loader_hard = DataLoader(task_hard, batch_size=16, shuffle=True)
        val_loader_hard = DataLoader(task_hard, batch_size=16, shuffle=False)

        self.train_model(model_base, train_loader_hard, max_steps=steps_per_stage * 3)
        _, acc_base = self.evaluate_model(model_base, val_loader_hard)
        self.logger.info(f"Baseline Accuracy: {acc_base*100:.2f}%")

        # 2. Curriculum
        self.logger.info("Curriculum: Training on Easy -> Medium -> Hard...")
        model_curr = ANAModel(config)

        difficulties = [4, 6, 8]
        for diff in difficulties:
            self.logger.info(f"Curriculum Stage: {diff} pairs")
            task_curr = AssociativeRecallDataset(num_samples=2000, vocab_size=40, num_pairs=diff, noise_len=16)
            loader_curr = DataLoader(task_curr, batch_size=16, shuffle=True)
            self.train_model(model_curr, loader_curr, max_steps=steps_per_stage)

        _, acc_curr = self.evaluate_model(model_curr, val_loader_hard)
        self.logger.info(f"Curriculum Accuracy: {acc_curr*100:.2f}%")

        results = {"baseline": acc_base, "curriculum": acc_curr}
        with open(os.path.join(self.output_dir, "curriculum_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_sensitivity_experiment(self, quick: bool = False):
        self.logger.info("=== EXPERIMENT: Hyperparameter Sensitivity ===")

        # Grid Search on key params
        # d_model vs track_count

        d_models = [32, 64]
        track_counts = [1, 2, 4]

        if quick:
            d_models = [32]
            track_counts = [1, 2]

        results = {}
        steps = 100 if quick else 300

        task = AssociativeRecallDataset(num_samples=1000, vocab_size=40, num_pairs=4, noise_len=16)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        for dm in d_models:
            for tc in track_counts:
                name = f"d{dm}_t{tc}"
                self.logger.info(f"Testing Config: {name}")

                config = ANAConfig(
                    d_model=dm, state_dim=dm, num_layers=2, track_count=tc,
                    key_dim=dm//2, use_hololink=True, use_controller=True
                )

                model = ANAModel(config)
                self.train_model(model, train_loader, max_steps=steps)
                _, acc = self.evaluate_model(model, val_loader)

                results[name] = acc
                self.logger.info(f"{name}: {acc*100:.2f}%")

        with open(os.path.join(self.output_dir, "sensitivity_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def generate_potential_report(self):
        report_path = os.path.join(self.output_dir, "POTENTIAL_REPORT.md")
        with open(report_path, 'w') as f:
            f.write("# Scientifically Revealing ANA's Potential\n\n")
            f.write(f"**Date:** {self.timestamp}\n\n")

            # Induction
            ind_path = os.path.join(self.output_dir, "induction_results.json")
            if os.path.exists(ind_path):
                with open(ind_path) as j: res = json.load(j)
                f.write("## 1. Induction Head Capability\n")
                f.write("The ability to perform in-context learning by completing patterns like `A ... B ... A -> B`.\n\n")
                f.write(f"- **Accuracy:** {res['accuracy']*100:.2f}%\n")
                if res['accuracy'] > 0.95:
                    f.write("**Result:** ANA successfully implements induction heads.\n")
                else:
                    f.write("**Result:** ANA struggles with induction heads.\n")
                f.write("\n")

            # Generalization
            gen_path = os.path.join(self.output_dir, "generalization_results.json")
            if os.path.exists(gen_path):
                with open(gen_path) as j: res = json.load(j)
                f.write("## 2. Length Generalization\n")
                f.write("Testing model performance on sequences longer than training data.\n\n")
                f.write("| Length | Accuracy |\n| :--- | :---: |\n")
                train_len = 64 # Hardcoded based on experiment
                for l, acc in res.items():
                    tag = "(Train)" if int(l) <= train_len else "(Extrapolate)"
                    f.write(f"| {l} {tag} | {acc:.4f} |\n")
                f.write("\n")

            # Multi-Query
            mq_path = os.path.join(self.output_dir, "multiquery_results.json")
            if os.path.exists(mq_path):
                with open(mq_path) as j: res = json.load(j)
                f.write("## 3. Multi-Query Associative Recall\n")
                f.write("Testing dense retrieval capacity.\n\n")
                f.write(f"- **Accuracy:** {res['accuracy']*100:.2f}%\n\n")

            # Reasoning
            reas_path = os.path.join(self.output_dir, "reasoning_results.json")
            if os.path.exists(reas_path):
                with open(reas_path) as j: res = json.load(j)
                f.write("## 4. Reasoning & Thinking Steps\n")
                f.write("Comparing standard processing vs Adaptive Computation Time (Thinking Steps).\n\n")
                f.write("| Configuration | Accuracy |\n| :--- | :---: |\n")
                for name, acc in res.items():
                    f.write(f"| {name} | {acc:.4f} |\n")
                f.write("\n")

            # Noise
            noise_path = os.path.join(self.output_dir, "noise_results.json")
            if os.path.exists(noise_path):
                with open(noise_path) as j: res = json.load(j)
                f.write("## 5. Noise Robustness\n")
                f.write("Training on clean data, testing on noisy data.\n\n")
                f.write(f"- **Clean Accuracy:** {res['clean']*100:.2f}%\n")
                f.write(f"- **Noisy Accuracy:** {res['noisy']*100:.2f}%\n\n")

            # Curriculum
            curr_path = os.path.join(self.output_dir, "curriculum_results.json")
            if os.path.exists(curr_path):
                with open(curr_path) as j: res = json.load(j)
                f.write("## 6. Curriculum Learning\n")
                f.write("Benefits of gradual difficulty increase.\n\n")
                f.write(f"- **Baseline:** {res['baseline']*100:.2f}%\n")
                f.write(f"- **Curriculum:** {res['curriculum']*100:.2f}%\n\n")

            # Sensitivity
            sens_path = os.path.join(self.output_dir, "sensitivity_results.json")
            if os.path.exists(sens_path):
                with open(sens_path) as j: res = json.load(j)
                f.write("## 7. Hyperparameter Sensitivity\n")
                f.write("Stability across configurations.\n\n")
                f.write("| Config | Accuracy |\n| :--- | :---: |\n")
                for name, acc in res.items():
                    f.write(f"| {name} | {acc:.4f} |\n")
                f.write("\n")

            f.write("## 8. Visual Analysis\n")
            f.write("Check the `plots/` directory for 'Cognitive State' visualizations showing how ANA dynamically allocates attention between HoloLink (Memory) and Recurrent Tracks (Reasoning).\n")

        self.logger.info(f"Report generated at {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick smoketest")
    parser.add_argument("--output_dir", type=str, default="results/potential")

    # Flags for individual experiments
    parser.add_argument("--induction", action="store_true")
    parser.add_argument("--generalization", action="store_true")
    parser.add_argument("--multiquery", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all experiments")

    args = parser.parse_args()

    # Default to all if no specific flag set
    if not (args.induction or args.generalization or args.multiquery or
            args.reasoning or args.noise or args.curriculum or args.sensitivity):
        args.all = True

    revealer = PotentialRevealer(output_dir=args.output_dir)

    if args.all or args.induction:
        print("=== Running Induction Experiment ===")
        revealer.run_induction_head_experiment(quick=args.quick)

    if args.all or args.generalization:
        print("=== Running Length Generalization Experiment ===")
        revealer.run_length_generalization_experiment(quick=args.quick)

    if args.all or args.multiquery:
        print("=== Running Multi-Query Experiment ===")
        revealer.run_multi_query_experiment(quick=args.quick)

    if args.all or args.reasoning:
        print("=== Running Reasoning Experiment ===")
        revealer.run_reasoning_experiment(quick=args.quick)

    if args.all or args.noise:
        print("=== Running Noise Robustness Experiment ===")
        revealer.run_noise_robustness_experiment(quick=args.quick)

    if args.all or args.curriculum:
        print("=== Running Curriculum Experiment ===")
        revealer.run_curriculum_experiment(quick=args.quick)

    if args.all or args.sensitivity:
        print("=== Running Sensitivity Experiment ===")
        revealer.run_sensitivity_experiment(quick=args.quick)

    revealer.generate_potential_report()
    print(f"Done. Results in {revealer.output_dir}")

if __name__ == "__main__":
    main()
