import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import time
import json

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


def generate_ar_batch(batch_size, seq_len, vocab_size, device):
    """Generate AR samples with consistent patterns."""
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        key = torch.randint(1, vocab_size, (1,)).item()
        value = torch.randint(1, vocab_size, (1,)).item()
        
        input_ids[b, 0] = key
        input_ids[b, 1] = value
        
        num_noise = torch.randint(5, 15, (1,)).item()
        for j in range(2, seq_len - 1):
            if j < 2 + num_noise:
                input_ids[b, j] = torch.randint(1, vocab_size, (1,)).item()
        
        input_ids[b, -1] = key
        target_ids[b, -1] = value
    
    return input_ids, target_ids


def train_ar_task(
    num_steps=500,
    batch_size=16,
    seq_len=24,
    vocab_size=50,
    lr=1e-3,
    eval_every=50,
    device=None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("AR Task Training")
    print("=" * 60)
    print(f"Steps: {num_steps}, Batch: {batch_size}, Seq: {seq_len}")
    
    model = create_bio_ana('nano').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    start_time = time.time()
    
    for step in range(num_steps):
        input_ids, target_ids = generate_ar_batch(batch_size, seq_len, vocab_size, device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        
        loss = F.cross_entropy(logits[:, -1, :], target_ids[:, -1])
        
        loss.backward()
        optimizer.step()
        
        if step % eval_every == 0 or step == num_steps - 1:
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == target_ids[:, -1]).float().mean()
            elapsed = time.time() - start_time
            print(f"Step {step:4d}: loss={loss.item():.4f}, acc={acc.item():.0%}, time={elapsed:.1f}s")
            
            if acc.item() > 0.98:
                print(f"\nReached 98% accuracy at step {step}!")
                break
    
    print(f"\nFinal eval...")
    model.eval()
    
    val_input, val_target = generate_ar_batch(100, seq_len, vocab_size, device)
    with torch.no_grad():
        logits = model(val_input)
        pred = logits[:, -1, :].argmax(dim=-1)
        acc = (pred == val_target[:, -1]).float().mean()
    
    print(f"Validation accuracy: {acc.item():.0%}")
    
    return {
        'steps': step + 1,
        'final_loss': loss.item(),
        'final_acc': acc.item(),
        'time_s': time.time() - start_time,
    }


if __name__ == "__main__":
    results = train_ar_task(num_steps=300, batch_size=16)
    
    with open('results/phase3_ar_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults: {results}")
