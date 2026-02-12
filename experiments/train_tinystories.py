"""
TinyStories Language Model Training

Train a small ANA model on TinyStories dataset to demonstrate
breakthrough performance vs larger baselines.

Target: ~5-10M param model beats GPT-2 small (117M params) on story generation.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import os
import json
import math
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import tiktoken
from datasets import load_dataset

from ana import ANAConfig, ANAModel


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LMConfig:
    # Model
    d_model: int = 128
    state_dim: int = 128
    key_dim: int = 64
    num_layers: int = 2
    track_count: int = 2
    use_hololink: bool = True
    use_controller: bool = True
    
    # Training
    batch_size: int = 16
    seq_len: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 50000
    warmup_steps: int = 500
    eval_interval: int = 500
    eval_steps: int = 100
    save_interval: int = 2000
    log_interval: int = 50
    
    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    
    # Paths
    data_dir: str = "data/tinystories"
    checkpoint_dir: str = "checkpoints/tinystories"
    samples_dir: str = "samples"
    
    # Tokenizer
    tokenizer_name: str = "gpt2"


# ============================================================================
# Tokenizer
# ============================================================================

class Tokenizer:
    """Wrapper around tiktoken for consistent interface."""
    
    def __init__(self, name: str = "gpt2"):
        self.encoder = tiktoken.get_encoding(name)
        self.vocab_size = self.encoder.n_vocab
        self.eos_token = "<|endoftext|>"
        self.eos_id = self.encoder.encode(self.eos_token, allowed_special={self.eos_token})[0]
        
    def encode(self, text: str) -> List[int]:
        return self.encoder.encode(text, allowed_special={self.eos_token})
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)
    
    def __len__(self):
        return self.vocab_size


# ============================================================================
# Dataset
# ============================================================================

class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling."""
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_len: int = 256,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: str = "data/tinystories"
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        
        print(f"Loading TinyStories {split} split...")
        
        # Load dataset
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Tokenize all stories
        print(f"Tokenizing {len(dataset)} stories...")
        self.tokens = []
        
        for i, example in enumerate(dataset):
            text = example["text"]
            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eos_id)
            self.tokens.extend(tokens)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{len(dataset)} stories, {len(self.tokens):,} tokens")
        
        print(f"Total tokens: {len(self.tokens):,}")
        
    def __len__(self):
        return (len(self.tokens) - self.seq_len) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        tokens = self.tokens[start:end]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y


def create_dataloaders(config: LMConfig, tokenizer: Tokenizer):
    """Create train and validation dataloaders."""
    
    train_dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        split="train",
        max_samples=50000,  # Start with subset for faster iteration
        cache_dir=config.data_dir
    )
    
    val_dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        split="validation",
        max_samples=2000,
        cache_dir=config.data_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# Model
# ============================================================================

def create_model(config: LMConfig, vocab_size: int) -> ANAModel:
    """Create ANA model for language modeling."""
    
    model_config = ANAConfig(
        vocab_size=vocab_size,
        d_model=config.d_model,
        state_dim=config.state_dim,
        key_dim=config.key_dim,
        num_layers=config.num_layers,
        track_count=config.track_count,
        use_hololink=config.use_hololink,
        use_controller=config.use_controller,
        use_parallel_scan=True,
        max_position=config.seq_len * 4  # Allow longer generation
    )
    
    return ANAModel(model_config)


def get_component_params(model: nn.Module):
    """Split parameters by component for two-phase training."""
    holo_params, ctl_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if 'holo' in name:
            holo_params.append(p)
        elif 'controller' in name:
            ctl_params.append(p)
        else:
            other_params.append(p)
    return holo_params, ctl_params, other_params


# ============================================================================
# Training
# ============================================================================

def get_lr(step: int, config: LMConfig):
    """Learning rate with warmup and cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def train_step(model, batch, optimizer, scheduler, scaler, device, config):
    """Single training step."""
    x, y = batch
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    
    with autocast('cuda'):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    
    if torch.isnan(loss):
        return float('nan')
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    
    model.train()
    return total_loss / total_tokens


# ============================================================================
# Generation
# ============================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cuda"
) -> str:
    """Generate text from prompt."""
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # Get logits for last position
        logits, _ = model(x)
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop on EOS
        if next_token.item() == tokenizer.eos_id:
            break
        
        x = torch.cat([x, next_token], dim=1)
    
    model.train()
    return tokenizer.decode(x[0].tolist())


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config: LMConfig):
    """Main training function."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = Tokenizer(config.tokenizer_name)
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_loader, val_loader = create_dataloaders(config, tokenizer)
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, tokenizer.vocab_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    holo_params, ctl_params, other_params = get_component_params(model)
    print(f"  HoloLink: {sum(p.numel() for p in holo_params):,}")
    print(f"  Controller: {sum(p.numel() for p in ctl_params):,}")
    print(f"  Other: {sum(p.numel() for p in other_params):,}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scaler = GradScaler('cuda')
    
    # Training loop
    print("\nStarting training...")
    print(f"Max steps: {config.max_steps:,}")
    
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    
    for step in range(config.max_steps):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Train step
        loss = train_step(model, batch, optimizer, None, scaler, device, config)
        lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if step % config.log_interval == 0:
            print(f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e}")
        
        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            ppl = math.exp(val_loss)
            print(f"\n{'='*60}")
            print(f"Step {step} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
            
            # Generate samples
            prompts = [
                "Once upon a time",
                "The little girl",
                "One day, a brave"
            ]
            
            print("\nGenerated samples:")
            for prompt in prompts:
                text = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    device=device
                )
                print(f"\n[{prompt}]")
                print(text)
            
            print(f"{'='*60}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config.__dict__
                }, os.path.join(config.checkpoint_dir, 'best.pt'))
        
        # Save checkpoint
        if step > 0 and step % config.save_interval == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__
            }, os.path.join(config.checkpoint_dir, f'checkpoint_{step}.pt'))
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    val_loss = evaluate(model, val_loader, device)
    ppl = math.exp(val_loss)
    print(f"Final Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
    
    # Generate final samples
    print("\nFinal samples:")
    for prompt in ["Once upon a time", "The little boy", "In a magical forest"]:
        text = generate(model, tokenizer, prompt, max_new_tokens=200, device=device)
        print(f"\n[{prompt}]")
        print(text)
    
    return model, tokenizer


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    config = LMConfig(
        d_model=128,
        state_dim=128,
        key_dim=64,
        num_layers=1,  # Start with single layer
        track_count=1,  # Single track
        use_hololink=True,
        use_controller=False,
        batch_size=8,
        seq_len=128,  # Shorter sequences
        max_steps=10000,
        eval_interval=500
    )
    
    model, tokenizer = train(config)
