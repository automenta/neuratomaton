
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

class CopyTaskDataset(Dataset):
    """
    Task: Repeat the input sequence.
    Input: [START] A B C [SEP]
    Target: A B C [END]
    """
    def __init__(self, size=1000, vocab_size=20, seq_len=10):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.START = 1
        self.SEP = 2
        self.END = 3
        self.content_start = 4

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq = [random.randint(self.content_start, self.vocab_size-1) for _ in range(self.seq_len)]

        # Input: START + SEQ + SEP
        # Target: SEQ + END (Shifted relative to input is complex for causal LM)

        # Causal LM:
        # Input:  [START] A B C [SEP] A B C
        # Target:    A    B C [SEP] A B C [END]

        # Actually standard copy task for causal LM:
        # Prompt: [START] A B C [SEP]
        # Completion: A B C [END]

        full_seq = [self.START] + seq + [self.SEP] + seq + [self.END]

        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)

        # Mask: Only calculate loss on the completion part
        mask = torch.zeros_like(y, dtype=torch.float)
        # Length of prompt is 1 + seq_len + 1 = seq_len + 2
        # So we mask until index seq_len + 1
        mask[self.seq_len + 1:] = 1.0

        return x, y, mask

class ReverseTaskDataset(Dataset):
    """
    Task: Reverse the input sequence.
    Input: [START] A B C [SEP]
    Target: C B A [END]
    """
    def __init__(self, size=1000, vocab_size=20, seq_len=10):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.START = 1
        self.SEP = 2
        self.END = 3
        self.content_start = 4

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq = [random.randint(self.content_start, self.vocab_size-1) for _ in range(self.seq_len)]
        rev_seq = seq[::-1]

        full_seq = [self.START] + seq + [self.SEP] + rev_seq + [self.END]

        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)

        mask = torch.zeros_like(y, dtype=torch.float)
        mask[self.seq_len + 1:] = 1.0

        return x, y, mask

def run_eval_task(model, dataset, device, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # No collate needed if fixed size
    model.eval()

    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)

            # Accuracy on masked region only
            correct = (preds == y).float() * mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    return total_correct / total_tokens if total_tokens > 0 else 0.0
