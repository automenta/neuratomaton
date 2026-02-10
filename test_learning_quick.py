import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import time

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


def test_forward_only():
    print("=" * 60)
    print("Forward Pass Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = create_bio_ana('nano').to(device)
    model.eval()
    
    vocab_size = 50
    seq_len = 16
    batch_size = 4
    
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(input_ids)
        elapsed = time.perf_counter() - start
    
    print(f"Forward pass: {elapsed*1000:.1f}ms")
    print(f"Output shape: {logits.shape}")
    print(f"Time per token: {elapsed*1000/(batch_size*seq_len):.2f}ms")


def test_single_step():
    print("\n" + "=" * 60)
    print("Single Training Step Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_bio_ana('nano').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    vocab_size = 50
    seq_len = 8
    batch_size = 2
    
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Input shape: {input_ids.shape}")
    
    model.train()
    optimizer.zero_grad()
    
    start = time.perf_counter()
    logits = model(input_ids)
    forward_time = time.perf_counter() - start
    
    loss = model.compute_loss(logits, target_ids)
    print(f"Loss: {loss['total'].item():.4f}")
    
    start = time.perf_counter()
    loss['total'].backward()
    backward_time = time.perf_counter() - start
    
    start = time.perf_counter()
    optimizer.step()
    optimizer_time = time.perf_counter() - start
    
    print(f"\nTiming:")
    print(f"  Forward:  {forward_time*1000:.1f}ms")
    print(f"  Backward: {backward_time*1000:.1f}ms")
    print(f"  Optimizer: {optimizer_time*1000:.1f}ms")
    print(f"  Total:    {(forward_time+backward_time+optimizer_time)*1000:.1f}ms")


def test_quick_learning():
    print("\n" + "=" * 60)
    print("Quick Learning Test (10 steps)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_bio_ana('nano').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    key, value = 10, 20
    input_ids = torch.zeros(2, 8, dtype=torch.long, device=device)
    input_ids[:, 0] = key
    input_ids[:, 1] = value
    input_ids[:, 7] = key
    
    target_ids = torch.zeros(2, 8, dtype=torch.long, device=device)
    target_ids[:, 7] = value
    
    print(f"Training: key={key}, value={value}, query at pos 7")
    
    model.train()
    start_time = time.perf_counter()
    
    for step in range(10):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits[:, 7, :], target_ids[:, 7])
        loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            with torch.no_grad():
                pred = logits[:, 7, :].argmax(dim=-1)
                acc = (pred == value).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.0%}")
    
    elapsed = time.perf_counter() - start_time
    print(f"\nTime: {elapsed:.1f}s ({10/elapsed:.1f} steps/sec)")


if __name__ == "__main__":
    test_forward_only()
    test_single_step()
    test_quick_learning()
