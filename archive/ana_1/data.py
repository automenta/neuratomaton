import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import itertools

class SlimPajamaStream(IterableDataset):
    def __init__(self, tokenizer, seq_len=2048, batch_size=8):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        # Load Streaming
        self.dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
        
    def __iter__(self):
        # Generator
        buffer = []
        for sample in self.dataset:
            text = sample['text']
            # Tokenize
            ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0]
            
            # Chunking
            # Simple approach: Accumulate and slice
            buffer.extend(ids.tolist())
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1] # x + target
                buffer = buffer[self.seq_len:] # Overlap? Or non-overlapping?
                # Standard is non-overlapping sliding window usually.
                # Let's do stride = seq_len
                 
                yield torch.tensor(chunk)

                yield torch.tensor(chunk)

def get_dataloader(batch_size=8, seq_len=2048, dataset_name="slimpajama"):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if dataset_name == "wikitext":
        # Load Wikitext-2 (Small)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Tokenize all
        def tokenize_function(examples):
            return tokenizer(examples["text"])
        
        tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
        # Concat
        block_size = seq_len + 1
        def group_texts(examples):
            concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        lm_ds = tokenized_ds.map(group_texts, batched=True)
        # Convert to iterator for consistency with interface or just return DL
        # Since it's small, map style DL is fine.
        from torch.utils.data import TensorDataset
        # But we need tensors.
        lm_ds.set_format(type='torch', columns=['input_ids'])
        return DataLoader(lm_ds, batch_size=batch_size, shuffle=True)
        
    ds = SlimPajamaStream(tokenizer, seq_len, batch_size)
    return DataLoader(ds, batch_size=batch_size)
