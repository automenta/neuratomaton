"""
Benchmark Suite - Systematic evaluation of model capabilities

Defines standard tasks for shallow model space exploration:
- Single-KV associative recall
- Multi-KV associative recall (capacity test)
- Copy task
- Reverse task
- Arithmetic task (simple computation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import random


# ============================================================================
# DATASETS
# ============================================================================

class SingleKVDataset(Dataset):
    """Single key-value associative recall (needle-in-haystack)."""

    def __init__(self, size=500, vocab_size=30, min_noise=5, max_noise=20):
        self.TOK_KEY, self.TOK_VAL, self.TOK_QUERY = 1, 2, 3
        self.content = list(range(4, vocab_size))
        self.data = []

        for _ in range(size):
            key = random.choice(self.content)
            value = random.choice(self.content)

            seq = [self.TOK_KEY, key, self.TOK_VAL, value]
            noise_len = random.randint(min_noise, max_noise)
            seq.extend([random.choice(self.content) for _ in range(noise_len)])
            seq.extend([self.TOK_QUERY, key, value])

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0

            self.data.append((x, y, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiKVDataset(Dataset):
    """Multi-KV associative recall - tests capacity."""

    def __init__(self, size=500, vocab_size=30, num_kv_pairs=2, min_noise=3, max_noise=10):
        self.TOK_KEY, self.TOK_VAL, self.TOK_QUERY = 1, 2, 3
        self.content = list(range(4, vocab_size))
        self.data = []

        for _ in range(size):
            kv_pairs = [(random.choice(self.content), random.choice(self.content)) for _ in range(num_kv_pairs)]

            seq = []
            for k, v in kv_pairs:
                seq.extend([self.TOK_KEY, k, self.TOK_VAL, v])

            noise_len = random.randint(min_noise, max_noise)
            seq.extend([random.choice(self.content) for _ in range(noise_len)])

            target_idx = random.randint(0, num_kv_pairs - 1)
            seq.extend([self.TOK_QUERY, kv_pairs[target_idx][0], kv_pairs[target_idx][1]])

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0

            self.data.append((x, y, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CopyDataset(Dataset):
    """Copy task - reproduce sequence verbatim."""

    def __init__(self, size=500, vocab_size=30, seq_len=8):
        self.TOK_COPY, self.TOK_END = 1, 2
        self.content = list(range(3, vocab_size))
        self.data = []

        for _ in range(size):
            content = [random.choice(self.content) for _ in range(seq_len)]
            seq = [self.TOK_COPY] + content + [self.TOK_END] + content

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            mask = torch.ones_like(y, dtype=torch.float)  # All positions matter

            self.data.append((x, y, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ReverseDataset(Dataset):
    """Reverse task - output reverse of input."""

    def __init__(self, size=500, vocab_size=30, seq_len=6):
        self.TOK_REVERSE, self.TOK_END = 1, 2
        self.content = list(range(3, vocab_size))
        self.data = []

        for _ in range(size):
            content = [random.choice(self.content) for _ in range(seq_len)]
            seq = [self.TOK_REVERSE] + content + [self.TOK_END] + list(reversed(content))

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            mask = torch.ones_like(y, dtype=torch.float)

            self.data.append((x, y, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ArithmeticDataset(Dataset):
    """Simple arithmetic task: learn to add small numbers."""

    def __init__(self, size=500, vocab_size=30, max_operand=10):
        self.TOK_ADD, self.TOK_EQ = 1, 2
        # Numbers are encoded as 3 + n
        self.data = []

        for _ in range(size):
            a = random.randint(0, max_operand)
            b = random.randint(0, max_operand)
            c = a + b

            seq = [3 + a, self.TOK_ADD, 3 + b, self.TOK_EQ, 3 + c]

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0

            self.data.append((x, y, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

TASK_DATASETS = {
    "single_kv": SingleKVDataset,
    "multi_kv": MultiKVDataset,
    "copy": CopyDataset,
    "reverse": ReverseDataset,
    "arithmetic": ArithmeticDataset,
}


def collate_fn(batch):
    """Pad sequences to same length."""
    xs, ys, masks = zip(*batch)
    max_len = max(t.size(0) for t in xs)

    x_pad = torch.stack([F.pad(t, (0, max_len - t.size(0))) for t in xs])
    y_pad = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=-100) for t in ys])
    # Mask needs to be adjusted - the target position should remain at the same index
    m_pad = torch.stack([F.pad(t, (0, max_len - t.size(0))) for t in masks])

    return x_pad, y_pad, m_pad


# ============================================================================
# EVALUATION PROTOCOLS
# ============================================================================

def evaluate_task(model: nn.Module, task_name: str, device='cpu', epochs=10, **dataset_args) -> Dict:
    """
    Evaluate a model on a task.

    Returns:
        results dict with:
            - final_accuracy: final test accuracy
            - best_accuracy: best test accuracy during training
            - final_loss: final training loss
            - history: training history
    """
    # Create dataset
    dataset_class = TASK_DATASETS[task_name]
    train_ds = dataset_class(size=400, **dataset_args)
    test_ds = dataset_class(size=100, **dataset_args)

    # Use collate_fn that tracks lengths
    def collate_with_lengths(batch):
        xs, ys, masks = zip(*batch)
        max_len = max(t.size(0) for t in xs)
        lengths = torch.tensor([t.size(0) for t in xs])
        x_pad = torch.stack([F.pad(t, (0, max_len - t.size(0))) for t in xs])
        y_pad = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=-100) for t in ys])
        m_pad = torch.stack([F.pad(t, (0, max_len - t.size(0))) for t in masks])
        return x_pad, y_pad, m_pad, lengths

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_with_lengths)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_with_lengths)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    history = {'loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x, y, mask, lengths in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)

            loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size())
            loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, mask, lengths in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = logits.argmax(dim=-1)
                # Check prediction at position (length-1) for each sequence
                for i, length in enumerate(lengths):
                    pred = preds[i, length-1].item()
                    target = y[i, length-1].item()
                    if pred == target:
                        correct += 1
                    total += 1

        history['loss'].append(epoch_loss / len(train_loader))
        history['test_acc'].append(correct / total)

    return {
        'final_accuracy': history['test_acc'][-1],
        'best_accuracy': max(history['test_acc']),
        'final_loss': history['loss'][-1],
        'history': history,
    }


# ============================================================================
# BENCHMARK SUITE
# ============================================================================

class BenchmarkSuite:
    """Run a comprehensive benchmark on a model."""

    def __init__(self, device='cpu'):
        self.device = device
        self.tasks = list(TASK_DATASETS.keys())

    def run_full_benchmark(self, model: nn.Module, epochs_per_task=10) -> Dict[str, Dict]:
        """Run all benchmarks."""
        results = {}

        for task_name in self.tasks:
            print(f"  Running task: {task_name}...")
            try:
                result = evaluate_task(model, task_name, self.device, epochs=epochs_per_task)
                results[task_name] = result
            except Exception as e:
                print(f"    Error on {task_name}: {e}")
                results[task_name] = {'error': str(e)}

        return results

    def run_quick_benchmark(self, model: nn.Module) -> Dict[str, float]:
        """Quick benchmark with few epochs."""
        results = {}

        for task_name in self.tasks:
            try:
                result = evaluate_task(model, task_name, self.device, epochs=5)
                results[task_name] = result['best_accuracy']
            except Exception as e:
                results[task_name] = 0.0

        return results


# ============================================================================
# MULTI-KV CAPACITY TEST
# ============================================================================

def test_capacity(model_factory, device='cpu', max_kv_pairs=8) -> Dict[int, float]:
    """
    Test model capacity on multi-KV task.

    Returns dict of {num_kv_pairs: accuracy}.
    """
    results = {}

    for num_kv in [1, 2, 3, 4, 6, 8]:
        if num_kv > max_kv_pairs:
            break

        # Create fresh model for each test
        model = model_factory()
        model = model.to(device)

        print(f"  Testing {num_kv} KV pairs...")
        result = evaluate_task(model, 'multi_kv', device, epochs=12, num_kv_pairs=num_kv)
        results[num_kv] = result['best_accuracy']

    return results
