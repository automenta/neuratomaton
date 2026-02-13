import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ana.models import ANAModel, BaselineSSM
from ana.config import ANAConfig
from ana.tasks import TASK_REGISTRY
from ana.research.core import ExperimentBase, ExperimentRegistry
import json
import os
import copy

def collate_with_mask(batch):
    """Collate function that handles variable-length sequences and masks."""
    if len(batch[0]) == 2:
        xs, ys = zip(*batch)
        masks = None
    else:
        xs, ys, masks = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    xs_pad = torch.stack([F.pad(x, (0, max_len - x.size(0))) for x in xs])
    # Pad target with -100 (ignore index)
    ys_pad = torch.stack([F.pad(y, (0, max_len - y.size(0)), value=-100) for y in ys])

    if masks is not None:
        # Pad masks with 0
        masks_pad = torch.stack([F.pad(m, (0, max_len - m.size(0))) for m in masks])
        return xs_pad, ys_pad, masks_pad
    return xs_pad, ys_pad, None

@ExperimentRegistry.register(phase=1, name="baseline_comparison")
class BaselineComparisonExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "baseline_comparison"

    @property
    def phase(self) -> int:
        return 1

    def create_dataset(self, task_name, num_samples, seq_len, vocab_size):
        TaskClass = TASK_REGISTRY[task_name]

        if task_name == 'associative_recall':
            # Map seq_len to noise_len
            # Seq = KEY k VAL v [noise] QUERY k
            # Fixed parts: 4 + 2 = 6 tokens.
            # noise_len = seq_len - 6
            if seq_len <= 6:
                noise_len = 1
            else:
                noise_len = seq_len - 6
            return TaskClass(num_samples=num_samples, vocab_size=vocab_size,
                             min_noise=noise_len, max_noise=noise_len)
        elif task_name == 'add':
             return TaskClass(num_samples=num_samples, max_val=20)
        else:
            return TaskClass(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)

    def evaluate_generalization(
        self,
        model,
        task_name,
        train_lengths,
        test_lengths,
        vocab_size=30,
        steps_per_length=200,
        lr=5e-3
    ):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        results = {'train': {}, 'test': {}}

        # Training Loop
        self.results.log(f"Training on lengths {train_lengths}...")
        model.train()

        for L in train_lengths:
            self.results.log(f"  Training Length: {L}")
            dataset = self.create_dataset(task_name, num_samples=steps_per_length * 32, seq_len=L, vocab_size=vocab_size)
            loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_with_mask)

            running_loss = 0.0
            steps = 0
            for x, y, mask in loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                optimizer.zero_grad()
                logits, _ = model(x)

                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)

                loss_raw = criterion(logits_flat, y_flat).view(y.size())

                if mask is not None:
                    # Masked loss
                    # We only care about positions where mask == 1.0
                    loss = (loss_raw * mask).sum() / (mask.sum() + 1e-8)
                else:
                    loss = loss_raw.mean()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                steps += 1

            self.results.log(f"    Avg Loss: {running_loss/steps:.4f}")
            results['train'][L] = running_loss/steps

        # Evaluation Loop
        self.results.log("Evaluating...")
        model.eval()
        with torch.no_grad():
            # Test on train lengths (Verification)
            for L in train_lengths:
                acc = self.compute_accuracy(model, task_name, L, vocab_size)
                # Store accuracy instead of loss for train results final report
                results['train'][L] = acc

            # Test on test lengths (Generalization)
            max_train = max(train_lengths)
            for L in test_lengths:
                acc = self.compute_accuracy(model, task_name, L, vocab_size)
                results['test'][L] = {
                    'accuracy': acc,
                    'k_ratio': L / max_train
                }

        return results

    def compute_accuracy(self, model, task_name, seq_len, vocab_size):
        dataset = self.create_dataset(task_name, num_samples=100, seq_len=seq_len, vocab_size=vocab_size)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_with_mask)

        correct, total = 0, 0
        for x, y, mask in loader:
            x, y = x.to(self.device), y.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            logits, _ = model(x)
            preds = logits.argmax(-1)

            valid = (y != -100)
            if mask is not None:
                valid = valid & (mask > 0.5)

            if valid.sum() == 0:
                continue

            correct += (preds[valid] == y[valid]).sum().item()
            total += valid.sum().item()

        return correct / total if total > 0 else 0

    def execute(self):
        tasks = ['copy', 'reverse', 'associative_recall']

        # Configuration
        # Need enough vocab for associative recall
        vocab_size = 40
        self.config.vocab_size = vocab_size

        train_lengths = [10, 20]
        test_lengths = [30, 40, 50]

        all_results = {}

        # Define configurations to test
        configs_to_test = [
            ("BaselineSSM", BaselineSSM, self.config),
            ("ANA_Full", ANAModel, self.config),
        ]

        # Ablation Config
        config_no_holo = copy.deepcopy(self.config)
        config_no_holo.use_hololink = False
        configs_to_test.append(("ANA_NoHolo", ANAModel, config_no_holo))

        for task_name in tasks:
            self.results.log(f"\n{'='*50}")
            self.results.log(f"Task: {task_name}")
            self.results.log(f"{'='*50}")

            task_results = {}

            # Reduce lengths for associative recall if needed?
            # AR uses len as noise len, so 50 is huge.
            # But let's try.

            for model_name, ModelClass, cfg in configs_to_test:
                self.results.log(f"\n--- {model_name} ---")

                # Init model
                model = ModelClass(cfg).to(self.device)
                params = sum(p.numel() for p in model.parameters())
                self.results.log(f"Parameters: {params:,}")

                res = self.evaluate_generalization(
                    model, task_name, train_lengths, test_lengths,
                    vocab_size=vocab_size,
                    steps_per_length=50, # Reduced for speed, increase for rigor
                    lr=5e-3
                )

                task_results[model_name] = res
                self.results.log(f"Train Accuracy: {res['train']}")
                self.results.log(f"Test Accuracy: {res['test']}")

                # Save Best Model
                if res['train'][max(train_lengths)] > 0.9:
                    save_path = self.results.get_path(f"{task_name}_{model_name}_best.pt")
                    torch.save(model.state_dict(), save_path)
                    self.results.log(f"Saved model to {save_path}")

            all_results[task_name] = task_results

        self.results.save_json("comparison_results.json", all_results)
        self.generate_comparison_report(all_results)

    def generate_comparison_report(self, results):
        content = "# ANA vs Baseline Comparison\n\n"
        content += "Analysis of ANA's performance against BaselineSSM and Ablations.\n\n"

        for task, data in results.items():
            content += f"## Task: {task.capitalize()}\n\n"
            content += "| Model | Train Acc (Max Len) | Gen Acc (Avg) | Gen Acc (Max Len) |\n"
            content += "|---|---|---|---|\n"

            for model, res in data.items():
                train_acc = res['train'][max(res['train'].keys())]
                test_accs = [r['accuracy'] for r in res['test'].values()]
                test_avg = sum(test_accs) / len(test_accs)
                test_max_len = res['test'][max(res['test'].keys())]['accuracy']

                content += f"| {model} | {train_acc*100:.1f}% | {test_avg*100:.1f}% | {test_max_len*100:.1f}% |\n"
            content += "\n"

        self.results.save_report("comparison_report.md", content)

if __name__ == "__main__":
    config = ANAConfig(d_model=64, vocab_size=40, state_dim=64, num_layers=2)
    # Ensure device is set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = BaselineComparisonExperiment(config, device=device)
    exp.run()
