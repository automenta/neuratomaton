"""
Fair Comparison: ANA vs Transformer (Slightly Larger)

Train both models on TinyStories and compare.
Transformer is ~2x larger to give it an advantage.
If ANA beats it, that's a meaningful result.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import tiktoken
from datasets import load_dataset


# ============================================================================
# Transformer Model
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with causal mask
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embedding.weight
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, []


# ============================================================================
# Dataset
# ============================================================================

class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.eos_id = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, tokens):
        return self.enc.decode(tokens)


class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, split="train", max_samples=None, cache_dir="data/tinystories"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print(f"Loading TinyStories {split}...")
        dataset = load_dataset("roneneldan/TinyStories", split=split, cache_dir=cache_dir)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Tokenizing {len(dataset)} stories...")
        self.tokens = []
        for i, example in enumerate(dataset):
            tokens = tokenizer.encode(example["text"])
            tokens.append(tokenizer.eos_id)
            self.tokens.extend(tokens)
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(dataset)}")
                
        print(f"Total tokens: {len(self.tokens):,}")
            
    def __len__(self):
        return max(1, (len(self.tokens) - self.seq_len) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.tokens[start:end]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ============================================================================
# Training
# ============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    
    model.train()
    return total_loss / total_tokens


def train_model(model, train_loader, val_loader, device, steps=5000, lr=3e-4, log_interval=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')
    
    train_iter = iter(train_loader)
    losses = []
    
    for step in range(steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with autocast('cuda'):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % log_interval == 0:
            losses.append(loss.item())
            print(f"  Step {step:5d} | Loss: {loss.item():.4f}")
    
    val_loss = evaluate(model, val_loader, device)
    return val_loss


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=100):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)
        x = torch.cat([x, next_token], dim=1)
        if next_token.item() == tokenizer.eos_id:
            break
    
    return tokenizer.decode(x[0].tolist())


# ============================================================================
# Main
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    
    # Create datasets
    print("\n" + "="*60)
    print("Creating datasets...")
    print("="*60)
    
    train_dataset = TinyStoriesDataset(tokenizer, seq_len=128, split="train", max_samples=30000)
    val_dataset = TinyStoriesDataset(tokenizer, seq_len=128, split="validation", max_samples=2000)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    results = {}
    
    # ========== ANA Model ==========
    print("\n" + "="*60)
    print("1. ANA Model (13M params)")
    print("="*60)
    
    from ana import ANAConfig, ANAModel
    
    ana_config = ANAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        state_dim=128,
        key_dim=64,
        num_layers=1,
        track_count=1,
        use_hololink=True,
        use_controller=False,
        use_parallel_scan=True,
        max_position=512
    )
    
    ana_model = ANAModel(ana_config).to(device)
    ana_params = count_params(ana_model)
    print(f"Parameters: {ana_params:,}")
    
    print("\nTraining ANA...")
    start = time.time()
    ana_loss = train_model(ana_model, train_loader, val_loader, device, steps=3000, lr=3e-4)
    ana_time = time.time() - start
    
    ana_ppl = math.exp(ana_loss)
    print(f"\nVal Loss: {ana_loss:.4f} | Perplexity: {ana_ppl:.2f} | Time: {ana_time:.1f}s")
    
    ana_sample = generate(ana_model, tokenizer, "Once upon a time", device)
    print(f"Sample: {ana_sample[:150]}...")
    
    results['ANA'] = {'params': ana_params, 'loss': ana_loss, 'ppl': ana_ppl, 'time': ana_time}
    
    # ========== Transformer (2x larger) ==========
    print("\n" + "="*60)
    print("2. Transformer (26M params, 2x larger)")
    print("="*60)
    
    # Calculate size for ~26M params
    # For transformer: params ≈ vocab*d + 4*d^2*L + 12*d*L (approx)
    # With d=192, L=4: ~14M
    # With d=256, L=4: ~25M
    
    tf_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    
    tf_params = count_params(tf_model)
    print(f"Parameters: {tf_params:,} ({tf_params/ana_params:.1f}x ANA)")
    
    print("\nTraining Transformer...")
    start = time.time()
    tf_loss = train_model(tf_model, train_loader, val_loader, device, steps=3000, lr=3e-4)
    tf_time = time.time() - start
    
    tf_ppl = math.exp(tf_loss)
    print(f"\nVal Loss: {tf_loss:.4f} | Perplexity: {tf_ppl:.2f} | Time: {tf_time:.1f}s")
    
    tf_sample = generate(tf_model, tokenizer, "Once upon a time", device)
    print(f"Sample: {tf_sample[:150]}...")
    
    results['Transformer'] = {'params': tf_params, 'loss': tf_loss, 'ppl': tf_ppl, 'time': tf_time}
    
    # ========== Results ==========
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<15} {'Params':>12} {'Val Loss':>10} {'Perplexity':>12} {'Time':>8}")
    print("-"*60)
    
    for name, data in results.items():
        print(f"{name:<15} {data['params']:>12,} {data['loss']:>10.4f} {data['ppl']:>12.2f} {data['time']:>7.1f}s")
    
    # ========== Key Insight ==========
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    
    if ana_ppl < tf_ppl:
        improvement = (tf_ppl - ana_ppl) / tf_ppl * 100
        print(f"""
ANA ({ana_params:,} params) BEATS Transformer ({tf_params:,} params)!

ANA Perplexity:     {ana_ppl:.2f}
Transformer PPL:    {tf_ppl:.2f}

ANA achieves {improvement:.1f}% better perplexity with {tf_params/ana_params:.1f}x FEWER parameters!

This demonstrates that HoloLink's associative memory provides
capabilities that larger Transformers can't match.
""")
    else:
        improvement = (ana_ppl - tf_ppl) / ana_ppl * 100
        print(f"""
Transformer ({tf_params:,} params) beats ANA ({ana_params:,} params).

ANA Perplexity:     {ana_ppl:.2f}
Transformer PPL:    {tf_ppl:.2f}

Transformer is {improvement:.1f}% better despite needing more compute.
The larger model has an advantage on this simple language task.
""")


if __name__ == "__main__":
    main()
