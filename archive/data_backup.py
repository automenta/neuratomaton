
import torch
from torch.utils.data import Dataset
import random
import string

class AssociativeRecallDataset(Dataset):
    """
    Generates 'Needle-in-a-Haystack' sequences.
    Structure:
    [KEY_MARK] [KEY_VAL] [VAL_MARK] [VAL_VAL] ... [NOISE] ... [QUERY_MARK] [KEY_VAL] [TARGET] [VAL_VAL]
    
    Vocabulary:
    0: PAD
    1: KEY_MARK
    2: VAL_MARK
    3: QUERY_MARK
    4: END_MARK (Target is next)
    5-25: Digits/Chars (Content)
    
    Task:
    Given: KEY A VAL B ... [Noise] ... QUERY A
    Predict: B
    """
    def __init__(self, size=1000, vocab_size=30, min_noise=10, max_noise=50):
        self.size = size
        self.vocab_size = vocab_size
        self.min_noise = min_noise
        self.max_noise = max_noise
        
        self.TOK_KEY = 1
        self.TOK_VAL = 2
        self.TOK_QUERY = 3
        self.TOK_TARGET = 4 # Might not be explicit in input, but useful concept
        
        self.content_tokens = list(range(5, vocab_size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. Select Key and Value
        key_token = random.choice(self.content_tokens)
        val_token = random.choice(self.content_tokens)
        
        # 2. Key-Value Pair Definition
        # [KEY] [K] [VAL] [V]
        kv_pair = [self.TOK_KEY, key_token, self.TOK_VAL, val_token]
        
        # 3. Noise
        noise_len = random.randint(self.min_noise, self.max_noise)
        noise = [random.choice(self.content_tokens) for _ in range(noise_len)]
        
        # 4. Query
        # [QUERY] [K]
        query = [self.TOK_QUERY, key_token]
        
        # Full Sequence
        # Input: KV_PAIR + NOISE + QUERY
        # Target: Shifted by 1. Last token of Target should be VAL_TOKEN.
        
        # We want to train it as causal LM.
        # Context: ... [QUERY] [K] -> Predict [V]
        
        input_seq = kv_pair + noise + query
        target_seq = kv_pair[1:] + noise + query + [val_token]
        
        # Truncate if needed (not expected with these params)
        # Pad? Not needed if batch_size=1 or collate handles it.
        # But standard DataLoader requires equal length tensors if not using custom collate.
        # We fixed noise for now or padding?
        # Let's simple fix max length.
        
        # Only training on the final prediction matters most, but Causal LM trains on all.
        # The loss on noise is irrelevant/easy. The loss on V is critical.
        
        x = torch.tensor(input_seq, dtype=torch.long)
        y = torch.tensor(target_seq[len(input_seq)-len(query)-len(noise)-len(kv_pair):], dtype=torch.long) # Wrong logic
        
        # Correct Causal LM:
        # Input:  [A, B, C]
        # Target: [B, C, D]
        
        full_seq = kv_pair + noise + query + [val_token]
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)
        
        # Masking? We can iterate loss over everything.
        # The model should learn the KV association.
        
        return x, y

class TextDataset(Dataset):
    """
    Character-level text dataset for Stage 2B Warmup.
    Reads text files from a directory.
    """
    def __init__(self, file_path, seq_len=64):
        self.seq_len = seq_len
        
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()
            
        # Vocab: ASCII 0-127 usually.
        # Simple mapping: Ord(c) % 256
        self.vocab_size = 256
        self.data = [ord(c) % 256 for c in self.text]
        
        # Chunk into sequences
        self.num_samples = len(self.data) // (seq_len + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        
        # If too short (shouldn't happen with integer division logic but strictly)
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y
