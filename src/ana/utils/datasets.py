"""
Dataset utilities for ANA experiments
"""

import torch
from torch.utils.data import Dataset
import random
import os
import numpy as np
from typing import Tuple, List, Optional


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling
    """
    def __init__(self, text: str, seq_len: int = 128, vocab_size: Optional[int] = None):
        chars = sorted(list(set(text)))
        if vocab_size and len(chars) < vocab_size:
            # Add padding characters if needed
            extra_chars = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in chars]
            chars.extend(extra_chars[:vocab_size - len(chars)])
        
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, (len(self.data) - self.seq_len) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        if end > len(self.data):
            # Pad if needed
            seq = self.data[start:]
            padded = torch.full((self.seq_len + 1,), 0, dtype=torch.long)
            padded[:len(seq)] = seq
            seq = padded
        else:
            seq = self.data[start:end]
        return seq[:-1], seq[1:]


class AssociativeRecallDataset(Dataset):
    """
    Dataset for associative recall tasks
    """
    def __init__(self, num_samples: int = 1000, vocab_size: int = 40, num_pairs: int = 4, noise_len: int = 8):
        self.samples = []
        self.TOK_KEY, self.TOK_VAL, self.TOK_QUERY = 1, 2, 3
        content_range = list(range(4, vocab_size))
        
        for _ in range(num_samples):
            # Select unique keys and values
            keys = np.random.choice(content_range, size=num_pairs, replace=False)
            vals = np.random.choice([x for x in content_range if x not in keys], size=num_pairs, replace=False)
            
            # Create KV pairs
            kv_seq = []
            for k, v in zip(keys, vals):
                kv_seq.extend([self.TOK_KEY, k, self.TOK_VAL, v])
            
            # Add noise
            noise = np.random.choice(content_range, size=noise_len)
            kv_seq.extend(noise)
            
            # Add query
            query_idx = np.random.randint(0, num_pairs)
            query_key = keys[query_idx]
            target_val = vals[query_idx]
            
            kv_seq.extend([self.TOK_QUERY, query_key, target_val])
            
            # Convert to tensor
            x = torch.tensor(kv_seq[:-1], dtype=torch.long)
            y = torch.tensor(kv_seq[1:], dtype=torch.long)
            
            # Mask - only care about predicting the final value
            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0
            
            self.samples.append((x, y, mask))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class CopyTask(Dataset):
    """Copy input sequence to output: [SEQ] [SEP] [SEQ]"""
    def __init__(self, num_samples=500, seq_len=10, vocab_size=20):
        # 0: PAD, 1: SEP, 2..: Content
        # Content tokens: range(2, vocab_size)
        self.data = []
        TOK_PAD, TOK_SEP = 0, 1
        content_range = list(range(2, vocab_size))

        for _ in range(num_samples):
            # Input: [SEQ] [SEP] [SEQ]
            # Target is masked on first SEQ and SEP

            seq = [random.choice(content_range) for _ in range(seq_len)]

            # Construct input/target
            full_seq = seq + [TOK_SEP] + seq

            x = torch.tensor(full_seq[:-1], dtype=torch.long)
            y = torch.tensor(full_seq[1:], dtype=torch.long)

            mask = torch.zeros_like(y, dtype=torch.float)
            # Mask should be 1 for the second SEQ part.
            # SEQ starts at index seq_len in y (target for SEP)
            mask[seq_len:] = 1.0

            self.data.append((x, y, mask))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class ReverseTask(Dataset):
    """Reverse input sequence: [SEQ] [SEP] [REV_SEQ]"""
    def __init__(self, num_samples=500, seq_len=10, vocab_size=20):
        self.data = []
        TOK_PAD, TOK_SEP = 0, 1
        content_range = list(range(2, vocab_size))

        for _ in range(num_samples):
            seq = [random.choice(content_range) for _ in range(seq_len)]
            rev_seq = seq[::-1]

            full_seq = seq + [TOK_SEP] + rev_seq

            x = torch.tensor(full_seq[:-1], dtype=torch.long)
            y = torch.tensor(full_seq[1:], dtype=torch.long)

            mask = torch.zeros_like(y, dtype=torch.float)
            # Mask for rev_seq part
            mask[seq_len:] = 1.0

            self.data.append((x, y, mask))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class ShiftTask(Dataset):
    """Shift sequence by k positions (with padding)."""
    def __init__(self, num_samples=500, seq_len=10, vocab_size=20, shift=1):
        self.data = []
        for _ in range(num_samples):
            seq = torch.randint(1, vocab_size, (seq_len,))
            target = torch.cat([torch.zeros(shift, dtype=torch.long), seq[:-shift]])

            x = seq
            y = target
            mask = torch.ones_like(y, dtype=torch.float)

            self.data.append((x, y, mask))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class SortTask(Dataset):
    """Sort sequence (limited vocab for tractability)."""
    def __init__(self, num_samples=500, seq_len=6, vocab_size=10):
        self.data = []
        for _ in range(num_samples):
            seq = torch.randint(1, vocab_size, (seq_len,))
            sorted_seq, _ = torch.sort(seq)

            x = seq
            y = sorted_seq
            mask = torch.ones_like(y, dtype=torch.float)
            self.data.append((x, y, mask))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class AddTask(Dataset):
    """Add two numbers (encoded as tokens)."""
    def __init__(self, num_samples=500, max_val=20):
        self.data = []
        TOK_PLUS, TOK_EQ = 1, 2
        # Numbers start from 3

        for _ in range(num_samples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            c = a + b

            # A + B = C
            seq = [3 + a, TOK_PLUS, 3 + b, TOK_EQ, 3 + c]

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)

            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0

            self.data.append((x, y, mask))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class TextGenerationTask(Dataset):
    """
    Character-level text generation from 'data/input.txt'.
    """
    def __init__(self, num_samples=2000, seq_len=128, vocab_size=None, filepath="data/input.txt"):
        self.block_size = seq_len # Map seq_len to block_size
        self.filepath = filepath
        self.num_samples = num_samples

        if not os.path.exists(filepath):
            # Fallback if file not found (e.g. CI/CD environment)
            self.text = "This is a dummy text fallback because the actual data was not found. " * 100
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.text = f.read()

        # Build Vocabulary
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode
        self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Ignore idx, just sample randomly
        if len(self.data) <= self.block_size + 1:
            # Handle too short data
            x = torch.zeros(self.block_size, dtype=torch.long)
            y = torch.zeros(self.block_size, dtype=torch.long)
            mask = torch.zeros(self.block_size, dtype=torch.float)
            return x, y, mask

        ix = random.randint(0, len(self.data) - self.block_size - 1)
        x = self.data[ix : ix + self.block_size]
        y = self.data[ix + 1 : ix + self.block_size + 1]

        # Mask is all ones for causal LM
        mask = torch.ones_like(y, dtype=torch.float)

        return x, y, mask


TASK_REGISTRY = {
    'copy': CopyTask,
    'reverse': ReverseTask,
    'associative_recall': AssociativeRecallDataset,
    'shift': ShiftTask,
    'sort': SortTask,
    'add': AddTask,
    'text_generation': TextGenerationTask,
}