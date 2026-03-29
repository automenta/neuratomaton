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


class PointerChainTask(Dataset):
    """
    Pointer Chain Execution: Given pairs (A->B, B->C, ...), predict the end of the chain starting from a Query.
    Input: [A] [B] [B] [C] [C] [D] ... [Q] [A]
    Target: ... [B] [C] [D] (Sequential prediction of the path)

    Or simpler: Just predict the final node after K hops?
    For sequence modeling, it's better to predict the full path or just the next token.
    Here we define it as: Given random pairs, and a query, predict the next K tokens in the chain.
    """
    def __init__(self, num_samples=1000, vocab_size=40, chain_len=3, noise_pairs=0):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content_range = list(range(4, vocab_size))

        for _ in range(num_samples):
            # Generate a chain: n1 -> n2 -> n3 -> ... -> nk
            nodes = np.random.choice(content_range, size=chain_len + 1, replace=False)

            pairs = []
            for i in range(chain_len):
                pairs.append((nodes[i], nodes[i+1]))

            # Shuffle pairs in input
            random.shuffle(pairs)

            # Construct input sequence: [K1 V1 K2 V2 ...]
            seq = []
            for k, v in pairs:
                seq.extend([TOK_KEY, k, TOK_VAL, v])

            # Add noise pairs
            if noise_pairs > 0:
                noise_nodes = np.random.choice(content_range, size=noise_pairs*2, replace=True)
                for i in range(noise_pairs):
                    seq.extend([TOK_KEY, noise_nodes[2*i], TOK_VAL, noise_nodes[2*i+1]])

            # Query: Start of the chain
            start_node = nodes[0]
            seq.extend([TOK_QUERY, start_node])

            # Target sequence: The rest of the chain [n2, n3, ..., nk]
            # But we need to align x and y.
            # x: ... [Q] [n1] -> Predict [n2]
            # Then feed [n2] -> Predict [n3]?
            # Standard AR setup:
            # Input: ... [Q] [n1] [n2] ... [nk-1]
            # Target: ...      [n2] [n3] ... [nk]

            chain_rest = nodes[1:].tolist()

            # Full sequence for training
            # We provide the chain tokens in input so model learns to follow?
            # Or we only provide [Q] [n1] and expect it to output [n2] ... [nk] purely from memory?
            # "Pure Pointer Execution" means retrieving from memory.

            # If we just put [Q] [n1], and target is [n2], that's 1 hop.
            # To do multi-hop generation, we feed the predicted token back.
            # For training, we use teacher forcing:
            # Input: ... [Q] [n1] [n2] ... [nk-1]
            # Target: ...     [n2] [n3] ... [nk]

            suffix_input = chain_rest[:-1] # [n2, ..., nk-1]
            suffix_target = chain_rest     # [n2, ..., nk]

            # Input x includes the prompt and the chain steps (except last)
            # prompt: [Q] [n1]

            full_input = seq + suffix_input
            full_target = seq[1:] + [TOK_QUERY, start_node] + suffix_target
            # Wait, y must be shifted x.
            # x: [Seq] [Q] [n1] [n2] ... [nk-1]
            # y:       [Q] [n1] [n2] ... [nk]
            # The part [Seq] [Q] [n1] is context.
            # We want to predict [n2] given [n1] and Context.

            x_seq = seq + suffix_input
            y_seq = seq[1:] + [suffix_target[0]] # Aligning is tricky with lists
            # Let's rebuild carefully.

            # Context: [K V ... K V]
            # Prompt: [Q] [n1]
            # Completion: [n2] [n3] ... [nk]

            # Full X: Context + [Q] + [n1] + [n2] ... + [nk-1]
            # Full Y: ...     + [n1] + [n2] + [n3] ... + [nk]

            x = torch.tensor(seq + chain_rest[:-1], dtype=torch.long)

            # Construct Y by shifting X left and appending last target
            # But 'seq' part of Y is just shifting seq.
            # The prediction starts after [Q] [n1].

            # Let's just construct the full sequence and shift.
            full_seq = seq + chain_rest

            x_t = torch.tensor(full_seq[:-1], dtype=torch.long)
            y_t = torch.tensor(full_seq[1:], dtype=torch.long)

            mask = torch.zeros_like(y_t, dtype=torch.float)

            # Mask the chain prediction part
            # The chain part starts after `len(seq)`.
            # `seq` ends with [Q] [n1].
            # So `len(seq)` is index of first prediction [n2] in y_t?
            # len(seq) in full_seq is the index of [n2].
            # y_t has length len(full_seq) - 1.
            # y_t[i] corresponds to prediction at step i.
            # step i=len(seq)-1 is input [n1] (last of seq), target [n2].

            start_pred_idx = len(seq) - 1
            if start_pred_idx < len(mask):
                mask[start_pred_idx:] = 1.0

            self.data.append((x_t, y_t, mask))

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


class InductionHeadTask(Dataset):
    """
    Induction Head Task: Predict B given ... A B ... A.
    Format: [SEQ] [A] [B] [SEQ] [A] -> Target [B]
    """
    def __init__(self, num_samples=1000, seq_len=64, vocab_size=40):
        self.data = []
        content_range = list(range(4, vocab_size))

        for _ in range(num_samples):
            # 1. Generate random sequence
            seq = np.random.choice(content_range, size=seq_len, replace=True).tolist()

            # 2. Pick a random pair (A, B)
            a, b = np.random.choice(content_range, size=2, replace=False)

            # 3. Insert (A, B) at a random position in the first half
            idx_AB = np.random.randint(0, seq_len // 2)
            seq[idx_AB] = a
            seq[idx_AB + 1] = b

            # 4. Insert A at a random position in the second half
            idx_A_trigger = np.random.randint(seq_len // 2 + 1, seq_len - 1)
            seq[idx_A_trigger] = a
            # We enforce the NEXT token in input to be B so the target is B?
            # No, we want the MODEL to predict B.
            # The target corresponding to input at idx_A_trigger is B.
            # So y[idx_A_trigger] = B.
            # Which means seq[idx_A_trigger + 1] must be B.
            seq[idx_A_trigger + 1] = b

            # Recreate tensors
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y_target = torch.tensor(seq[1:], dtype=torch.long)

            # Mask: only predict B after the second A
            mask = torch.zeros_like(y_target, dtype=torch.float)
            mask[idx_A_trigger] = 1.0

            self.data.append((x, y_target, mask))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class MultiQueryAssociativeRecall(Dataset):
    """
    Associative Recall with multiple queries.
    Input: [K1] [V1] [K2] [V2] ... [Q_K1] [Q_K2]
    Target: ... [V1] [V2]
    """
    def __init__(self, num_samples=1000, vocab_size=40, num_pairs=8, num_queries=3):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content_range = list(range(4, vocab_size))

        for _ in range(num_samples):
            keys = np.random.choice(content_range, size=num_pairs, replace=False)
            vals = np.random.choice([x for x in content_range if x not in keys], size=num_pairs, replace=False)

            kv_seq = []
            for k, v in zip(keys, vals):
                kv_seq.extend([TOK_KEY, k, TOK_VAL, v])

            # Queries
            query_indices = np.random.choice(range(num_pairs), size=num_queries, replace=False)
            query_keys = keys[query_indices]
            query_vals = vals[query_indices]

            base = len(kv_seq)
            for qk, qv in zip(query_keys, query_vals):
                kv_seq.extend([TOK_QUERY, qk, qv])

            x = torch.tensor(kv_seq[:-1], dtype=torch.long)
            y = torch.tensor(kv_seq[1:], dtype=torch.long)

            mask = torch.zeros_like(y, dtype=torch.float)

            # Mask the values corresponding to queries
            # Query block: [Q, K, V]. We want to predict V given K.
            # x indices for block i (after base):
            # base + i*3 + 0: Q
            # base + i*3 + 1: K -> Predict V (at y[base + i*3 + 1])
            # base + i*3 + 2: V (not in x if last, but in y)

            # x length is len(kv_seq) - 1.
            # base is start of query part in kv_seq.

            # Adjust base for x indexing (x matches kv_seq up to -1)
            # Query part starts at 'base' index in kv_seq.
            # So in x, it starts at 'base'.

            for i in range(num_queries):
                # We want to mask the position where input is K (so we predict V)
                # K is at index `base + i*3 + 1` in kv_seq.
                # So input x index is `base + i*3 + 1`.

                mask_idx = base + i * 3 + 1
                if mask_idx < len(mask):
                    mask[mask_idx] = 1.0

            self.data.append((x, y, mask))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


TASK_REGISTRY = {
    'copy': CopyTask,
    'reverse': ReverseTask,
    'associative_recall': AssociativeRecallDataset,
    'induction_head': InductionHeadTask,
    'multi_query_ar': MultiQueryAssociativeRecall,
    'pointer_chain': PointerChainTask,
    'shift': ShiftTask,
    'sort': SortTask,
    'add': AddTask,
    'text_generation': TextGenerationTask,
}