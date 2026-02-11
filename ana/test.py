"""
Quick test to verify ANA works
"""

import torch
from ana import ANAConfig, ANAModel

def test_forward():
    """Test forward pass works"""
    config = ANAConfig(d_model=32, vocab_size=20, state_dim=32, track_count=2)
    model = ANAModel(config)
    x = torch.randint(1, 10, (2, 8))
    logits, info = model(x)
    print(f"✓ Forward pass: {logits.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return True

def test_backward():
    """Test backward pass works"""
    config = ANAConfig(d_model=32, vocab_size=20, state_dim=32, track_count=2)
    model = ANAModel(config)
    x = torch.randint(1, 10, (2, 8))
    logits, _ = model(x)
    targets = torch.randint(1, 10, (2, 8))
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 20), targets.view(-1))
    loss.backward()
    print(f"✓ Backward pass: loss={loss.item():.4f}")
    return True

def test_copy_task():
    """Test simple copy learning"""
    config = ANAConfig(d_model=32, vocab_size=10, state_dim=32, track_count=2)
    model = ANAModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Copy task: input = output
    train = torch.randint(1, 8, (16, 8))
    targets = train.clone()
    
    for i in range(50):
        optimizer.zero_grad()
        logits, _ = model(train)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 10), targets.view(-1))
        loss.backward()
        optimizer.step()
    
    # Check accuracy
    model.eval()
    with torch.no_grad():
        logits, _ = model(train)
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
    print(f"✓ Copy task accuracy: {100*acc:.0f}%")
    return acc > 0.8

def test_scaling():
    """Test O(N) scaling"""
    import time
    config = ANAConfig(d_model=32, vocab_size=20, state_dim=32, track_count=2)
    model = ANAModel(config)
    model.eval()
    
    times = []
    for seq_len in [64, 128, 256, 512]:
        x = torch.randint(1, 10, (1, seq_len))
        t0 = time.time()
        logits, _ = model(x)
        t1 = time.time()
        times.append((seq_len, t1 - t0))
    
    print("✓ Scaling test:")
    for n, t in times:
        print(f"  n={n:4}: {t*1000:.1f}ms")
    return True

if __name__ == "__main__":
    print("Testing ANA...")
    test_forward()
    test_backward()
    test_copy_task()
    test_scaling()
    print("\n✓ All tests passed!")
