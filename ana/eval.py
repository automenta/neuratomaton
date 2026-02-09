import torch
from torch.utils.data import Dataset, DataLoader
import random

class CopyTaskDataset(Dataset):
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
        full_seq = [self.START] + seq + [self.SEP] + seq + [self.END]
        
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[self.seq_len + 1:] = 1.0
        
        return x, y, mask

class ReverseTaskDataset(Dataset):
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

class AdditionTaskDataset(Dataset):
    def __init__(self, size=1000, max_digits=3, base=10):
        self.size = size
        self.max_digits = max_digits
        self.base = base
        self.DIGIT_START = 1
        self.SEP = base + 1
        self.END = base + 2
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        max_num = self.base ** self.max_digits - 1
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        result = a + b
        
        def num_to_digits(n, pad_len=None):
            if n == 0:
                digits = [0]
            else:
                digits = []
                while n > 0:
                    digits.append(n % self.base)
                    n //= self.base
            if pad_len and len(digits) < pad_len:
                digits = digits + [0] * (pad_len - len(digits))
            return digits
        
        a_digits = num_to_digits(a, self.max_digits)
        b_digits = num_to_digits(b, self.max_digits)
        max_result_digits = self.max_digits + 1
        result_digits = num_to_digits(result, max_result_digits)
        
        seq = a_digits + [self.SEP] + b_digits + [self.SEP]
        full_seq = seq + result_digits + [self.END]
        
        x = torch.tensor([d + self.DIGIT_START for d in full_seq[:-1]], dtype=torch.long)
        y = torch.tensor([d + self.DIGIT_START for d in full_seq[1:]], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[len(seq):] = 1.0
        
        return x, y, mask

class SortTaskDataset(Dataset):
    def __init__(self, size=1000, vocab_size=20, seq_len=5):
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
        sorted_seq = sorted(seq)
        full_seq = [self.START] + seq + [self.SEP] + sorted_seq + [self.END]
        
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)
        
        mask = torch.zeros_like(y, dtype=torch.float)
        mask[self.seq_len + 1:] = 1.0
        
        return x, y, mask

def run_eval_task(model, dataset, device, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            
            correct = (preds == y).float() * mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    return total_correct / total_tokens if total_tokens > 0 else 0.0

def run_all_evals(model, device, vocab_size=40):
    results = {}
    
    copy_ds = CopyTaskDataset(size=200, vocab_size=vocab_size, seq_len=10)
    results['copy'] = run_eval_task(model, copy_ds, device)
    
    rev_ds = ReverseTaskDataset(size=200, vocab_size=vocab_size, seq_len=10)
    results['reverse'] = run_eval_task(model, rev_ds, device)
    
    add_ds = AdditionTaskDataset(size=200, max_digits=2, base=10)
    vocab_size_add = 10 + 3
    results['addition'] = run_eval_task(model, add_ds, device)
    
    sort_ds = SortTaskDataset(size=200, vocab_size=vocab_size, seq_len=5)
    results['sort'] = run_eval_task(model, sort_ds, device)
    
    return results
