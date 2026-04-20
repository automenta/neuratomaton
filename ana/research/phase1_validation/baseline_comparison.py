import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ana.models import ANAModel, BaselineSSM
from ana.config import ANAConfig
from ana.tasks import TASK_REGISTRY
from ana.research.core import ExperimentBase, ExperimentRegistry
import json

def collate_with_mask(batch):
    """Collate function that handles variable-length sequences and masks."""
    if len(batch[0]) == 2:
        xs, ys = zip(*batch)
        masks = None
    else:
        xs, ys, masks = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    xs_pad = torch.stack([F.pad(x, (0, max_len - x.size(0))) for x in xs])
    ys_pad = torch.stack([F.pad(y, (0, max_len - y.size(0)), value=-100) for y in ys])

    if masks is not None:
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
                noise_len = 1 # Minimum noise
            else:
                noise_len = seq_len - 6

            # Using same min/max noise for simplicity to target approximate length
            return TaskClass(num_samples=num_samples, vocab_size=vocab_size,
                             min_noise=noise_len, max_noise=noise_len)
        elif task_name == 'add':
             # AddTask doesn't take seq_len either, it depends on max_val.
             # Seq len is fixed at 4 (num + num = num).
             # So we ignore seq_len for AddTask or skip it for length generalization if seq_len != 4.
             # For now, let's just pass default if possible, but AddTask init is:
             # __init__(self, num_samples=500, max_val=20)
             return TaskClass(num_samples=num_samples, max_val=20)
        else:
            return TaskClass(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)

    def evaluate_generalization(
        self,
        model,
        task_name,
        train_lengths,
        test_lengths,
        vocab_size=20,
        steps_per_length=50,
        lr=1e-2
    ):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        results = {'train': {}, 'test': {}}

        # Training
        self.results.log(f"Training on lengths {train_lengths}...")
        model.train()
        for L in train_lengths:
            dataset = self.create_dataset(task_name, num_samples=steps_per_length * 16, seq_len=L, vocab_size=vocab_size)
            loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_with_mask)

            for x, y, mask in loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                optimizer.zero_grad()
                logits, _ = model(x)

                # Flatten
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)

                loss_raw = criterion(logits_flat, y_flat).view(y.size())

                if mask is not None:
                    loss = (loss_raw * mask).sum() / (mask.sum() + 1e-8)
                else:
                    loss = loss_raw.mean()

                loss.backward()
                optimizer.step()

        # Evaluation
        self.results.log("Evaluating...")
        model.eval()
        with torch.no_grad():
            # Train accuracy
            for L in train_lengths:
                dataset = self.create_dataset(task_name, num_samples=100, seq_len=L, vocab_size=vocab_size)
                loader = DataLoader(dataset, batch_size=16, collate_fn=collate_with_mask)

                correct, total = 0, 0
                for x, y, mask in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits, _ = model(x)
                    preds = logits.argmax(-1)

                    valid = (y != -100)
                    if mask is not None:
                        # Only count masked positions if mask exists?
                        # Usually mask indicates target positions.
                        # ana/benchmark.py logic: valid = (y != -100)
                        # But AssociativeRecallTask sets mask[-1]=1 and others 0.
                        # If we use y!=-100, we count all non-padded tokens.
                        # Let's stick to simple valid check for now as per benchmark.py
                        pass

                    correct += (preds[valid] == y[valid]).sum().item()
                    total += valid.sum().item()

                acc = correct / total if total > 0 else 0
                results['train'][L] = acc

            # Test accuracy (generalization)
            max_train = max(train_lengths)
            for L in test_lengths:
                dataset = self.create_dataset(task_name, num_samples=100, seq_len=L, vocab_size=vocab_size)
                loader = DataLoader(dataset, batch_size=16, collate_fn=collate_with_mask)

                correct, total = 0, 0
                for x, y, mask in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits, _ = model(x)
                    preds = logits.argmax(-1)

                    valid = (y != -100)
                    correct += (preds[valid] == y[valid]).sum().item()
                    total += valid.sum().item()

                acc = correct / total if total > 0 else 0
                results['test'][L] = {
                    'accuracy': acc,
                    'k_ratio': L / max_train
                }

        return results

    def execute(self):
        tasks = ['copy', 'reverse'] # Default tasks
        # Can override via config if needed, but keeping simple for now

        train_lengths = [10, 20]
        test_lengths = [30, 40]

        # Override lengths for speed if just testing
        # train_lengths = [5, 10]
        # test_lengths = [15]

        all_results = {}

        for task_name in tasks:
            self.results.log(f"\n{'='*50}")
            self.results.log(f"Task: {task_name}")
            self.results.log(f"{'='*50}")

            task_results = {}

            for model_name, ModelClass in [('ANA', ANAModel), ('Baseline', BaselineSSM)]:
                self.results.log(f"\n--- {model_name} ---")

                # Re-init model
                model = ModelClass(self.config).to(self.device)
                params = sum(p.numel() for p in model.parameters())
                self.results.log(f"Parameters: {params:,}")

                res = self.evaluate_generalization(
                    model, task_name, train_lengths, test_lengths,
                    vocab_size=self.config.vocab_size,
                    steps_per_length=20, # Reduced for speed in this demo
                    lr=1e-2
                )

                task_results[model_name] = res

                self.results.log(f"Train Accuracy: {res['train']}")
                self.results.log(f"Test Accuracy: {res['test']}")

            all_results[task_name] = task_results

        self.results.save_json("comparison_results.json", all_results)
        self.generate_comparison_report(all_results)

    def generate_comparison_report(self, results):
        content = "# ANA vs Baseline Comparison\n\n"

        for task, data in results.items():
            content += f"## Task: {task.capitalize()}\n\n"
            content += "| Model | Train Acc (Avg) | Gen Acc (Avg) |\n"
            content += "|---|---|---|\n"

            for model, res in data.items():
                train_avg = sum(res['train'].values()) / len(res['train'])
                test_avg = sum(r['accuracy'] for r in res['test'].values()) / len(res['test'])
                content += f"| {model} | {train_avg*100:.1f}% | {test_avg*100:.1f}% |\n"
            content += "\n"

        self.results.save_report("comparison_report.md", content)

if __name__ == "__main__":
    config = ANAConfig(d_model=32, vocab_size=20, state_dim=32)
    exp = BaselineComparisonExperiment(config)
    exp.run()
