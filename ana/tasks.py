"""
ANA Task Suite - Algorithmic reasoning benchmarks
"""
import torch
from torch.utils.data import Dataset
import random


class CopyTask(Dataset):
    """Copy input sequence to output."""
    def __init__(self, num_samples=500, seq_len=10, vocab_size=20):
        self.data = []
        for _ in range(num_samples):
            seq = torch.randint(1, vocab_size, (seq_len,))
            self.data.append((seq, seq.clone()))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class ReverseTask(Dataset):
    """Reverse input sequence."""
    def __init__(self, num_samples=500, seq_len=10, vocab_size=20):
        self.data = []
        for _ in range(num_samples):
            seq = torch.randint(1, vocab_size, (seq_len,))
            self.data.append((seq, seq.flip(dims=[0])))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class AssociativeRecallTask(Dataset):
    """Key-value associative recall (needle in haystack)."""
    def __init__(self, num_samples=500, vocab_size=30, min_noise=10, max_noise=30):
        self.data = []
        TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
        content = list(range(4, vocab_size))
        
        for _ in range(num_samples):
            key = random.choice(content)
            val = random.choice([c for c in content if c != key])
            
            seq = [TOK_KEY, key, TOK_VAL, val]
            noise_len = random.randint(min_noise, max_noise)
            seq.extend([random.choice(content) for _ in range(noise_len)])
            seq.extend([TOK_QUERY, key])
            
            x = torch.tensor(seq, dtype=torch.long)
            y = torch.tensor(seq[1:] + [val], dtype=torch.long)
            
            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0
            
            self.data.append((x, y, mask))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class ShiftTask(Dataset):
    """Shift sequence by k positions (with padding)."""
    def __init__(self, num_samples=500, seq_len=10, vocab_size=20, shift=1):
        self.data = []
        for _ in range(num_samples):
            seq = torch.randint(1, vocab_size, (seq_len,))
            # Shift right, pad with zeros
            target = torch.cat([torch.zeros(shift, dtype=torch.long), seq[:-shift]])
            self.data.append((seq, target))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class SortTask(Dataset):
    """Sort sequence (limited vocab for tractability)."""
    def __init__(self, num_samples=500, seq_len=6, vocab_size=10):
        self.data = []
        for _ in range(num_samples):
            seq = torch.randint(1, vocab_size, (seq_len,))
            sorted_seq, _ = torch.sort(seq)
            self.data.append((seq, sorted_seq))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class AddTask(Dataset):
    """Add two numbers (encoded as tokens)."""
    def __init__(self, num_samples=500, max_val=20):
        self.data = []
        TOK_PLUS, TOK_EQ = 1, 2
        
        for _ in range(num_samples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            c = a + b
            
            seq = [3 + a, TOK_PLUS, 3 + b, TOK_EQ]
            x = torch.tensor(seq, dtype=torch.long)
            y = torch.tensor(seq[1:] + [3 + c], dtype=torch.long)
            
            mask = torch.zeros_like(y, dtype=torch.float)
            mask[-1] = 1.0
            
            self.data.append((x, y, mask))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


TASK_REGISTRY = {
    'copy': CopyTask,
    'reverse': ReverseTask,
    'associative_recall': AssociativeRecallTask,
    'shift': ShiftTask,
    'sort': SortTask,
    'add': AddTask,
}
