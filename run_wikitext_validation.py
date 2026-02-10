#!/usr/bin/env python3
"""
Bio-ANA WikiText-2 Rapid Validation
Tests model on real language modeling task
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import json

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


class WikiTextDataset(Dataset):
    """Simple WikiText dataset for language modeling"""
    
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_sequences = (len(tokens) - 1) // seq_len
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        if end > len(self.tokens):
            start = len(self.tokens) - self.seq_len - 1
            end = len(self.tokens)
        
        seq = self.tokens[start:end]
        input_ids = seq[:-1]
        target_ids = seq[1:]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def tokenize_wikitext(data_path, vocab_size=10000):
    """Simple tokenizer for WikiText"""
    with open(data_path, 'r') as f:
        text = f.read()
    
    # Simple whitespace + punctuation tokenization
    tokens = []
    words = text.split()
    
    # Build vocabulary
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Top vocab_size words
    vocab = ['<pad>', '<unk>'] + sorted(word_counts.keys(), key=lambda x: -word_counts[x])[:vocab_size-2]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    
    # Tokenize
    for word in words:
        token_id = word_to_id.get(word, 1)  # 1 = <unk>
        tokens.append(token_id)
    
    return tokens, vocab


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for input_ids, target_ids in tqdm(dataloader, desc="Training"):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Reshape for loss calculation
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, V),
            target_ids.view(-1),
            ignore_index=0
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_tokens = 0
    
    for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        logits = model(input_ids)
        
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, V),
            target_ids.view(-1),
            ignore_index=0,
            reduction='sum'
        )
        
        total_loss += loss.item()
        num_tokens += (target_ids != 0).sum().item()
    
    ppl = torch.exp(torch.tensor(total_loss / num_tokens)).item()
    return ppl, total_loss / num_tokens


def run_wikitext_validation(
    variant='nano',
    data_path='data/wikitext-2/train.txt',
    vocab_size=10000,
    seq_len=128,
    batch_size=8,
    num_epochs=3,
    lr=1e-3,
    eval_every=1,
    device=None,
    output_dir=None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Bio-ANA WikiText-2 Validation")
    print("="*60)
    print(f"Variant: {variant}")
    print(f"Vocab size: {vocab_size}")
    print(f"Seq length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print()
    
    # Check for data
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"⚠ WikiText data not found at {data_path}")
        print("Using synthetic tokens for testing...")
        
        # Generate synthetic tokens
        import random
        random.seed(42)
        tokens = [random.randint(1, vocab_size-1) for _ in range(100000)]
        vocab = [f'token_{i}' for i in range(vocab_size)]
    else:
        print(f"Loading WikiText from {data_path}...")
        tokens, vocab = tokenize_wikitext(data_path, vocab_size)
    
    print(f"Dataset: {len(tokens)} tokens, vocab size: {len(vocab)}")
    
    # Create dataset
    # Use subset for quick validation
    subset_size = min(len(tokens), 200000)  # 200K tokens for quick test
    tokens = tokens[:subset_size]
    
    train_size = int(0.9 * len(tokens))
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]
    
    print(f"Train tokens: {len(train_tokens)}, Val tokens: {len(val_tokens)}")
    
    train_dataset = WikiTextDataset(train_tokens, seq_len)
    val_dataset = WikiTextDataset(val_tokens, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    config = get_bio_config(variant, vocab_size=vocab_size)
    model = create_bio_ana(variant, vocab_size=vocab_size).to(device)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print()
    print("="*60)
    print("Starting Training")
    print("="*60)
    
    start_time = time.time()
    history = []
    best_ppl = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        if (epoch + 1) % eval_every == 0:
            val_ppl, val_loss = evaluate(model, val_loader, device)
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val PPL: {val_ppl:.2f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'time_s': epoch_time,
            })
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}")
    
    total_time = time.time() - start_time
    
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Best validation perplexity: {best_ppl:.2f}")
    print(f"Final validation perplexity: {val_ppl:.2f}")
    
    # Success criteria
    target_ppl = 35
    if best_ppl < target_ppl:
        print(f"\n✅ SUCCESS: PPL {best_ppl:.2f} < target {target_ppl}")
        status = "PASS"
    elif best_ppl < target_ppl * 1.15:
        print(f"\n⚠ MODERATE: PPL {best_ppl:.2f} within 15% of target {target_ppl}")
        status = "MODERATE"
    else:
        print(f"\n❌ NEEDS WORK: PPL {best_ppl:.2f} exceeds target {target_ppl}")
        status = "FAIL"
    
    results = {
        'variant': variant,
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'best_val_ppl': best_ppl,
        'final_val_ppl': val_ppl,
        'total_time_s': total_time,
        'status': status,
        'target_ppl': target_ppl,
        'history': history,
    }
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Bio-ANA WikiText-2 Validation')
    parser.add_argument('--variant', type=str, default='nano',
                        choices=['nano', 'small', 'base', 'large'],
                        help='Model variant')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='results/wikitext2_validation',
                        help='Output directory')
    
    args = parser.parse_args()
    
    results = run_wikitext_validation(
        variant=args.variant,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output,
    )
    
    print(f"\nStatus: {results['status']}")
