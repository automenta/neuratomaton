import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import time

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


def test_simple_learning():
    print("=" * 60)
    print("Simple Learning Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = get_bio_config('nano')
    model = create_bio_ana('nano').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    vocab_size = config.vocab_size
    seq_len = 16
    
    key = 10
    value = 20
    query = key
    
    input_ids = torch.zeros(4, seq_len, dtype=torch.long, device=device)
    target_ids = torch.zeros(4, seq_len, dtype=torch.long, device=device)
    
    input_ids[:, 0] = key
    input_ids[:, 1] = value
    input_ids[:, 7] = query
    target_ids[:, 7] = value
    
    print(f"Input pattern: key={key}, value={value}, query={query}")
    print(f"Target at position 7: {value}")
    
    print("\nTraining for 100 steps...")
    for step in range(100):
        optimizer.zero_grad()
        
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, 7, :],
            target_ids[:, 7],
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            with torch.no_grad():
                pred = logits[:, 7, :].argmax(dim=-1)
                acc = (pred == target_ids[:, 7]).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.2%}")
    
    print("\nFinal evaluation:")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        pred = logits[:, 7, :].argmax(dim=-1)
        prob = F.softmax(logits[:, 7, :], dim=-1)
        top_probs, top_ids = prob[0].topk(5)
        
        print(f"  Predictions: {top_ids.tolist()}")
        print(f"  Probabilities: {top_probs.tolist()}")
        print(f"  Target: {value}")
        print(f"  Correct: {pred[0].item() == value}")


def test_ar_task():
    print("\n" + "=" * 60)
    print("Associative Recall Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_bio_ana('nano').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    vocab_size = 50
    seq_len = 32
    num_samples = 100
    
    print(f"Generating {num_samples} training samples...")
    
    inputs = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)
    query_positions = []
    
    for i in range(num_samples):
        key = torch.randint(1, vocab_size, (1,)).item()
        value = torch.randint(1, vocab_size, (1,)).item()
        
        inputs[i, 0] = key
        inputs[i, 1] = value
        
        for j in range(2, seq_len - 1):
            if torch.rand(1).item() > 0.7:
                inputs[i, j] = torch.randint(1, vocab_size, (1,)).item()
        
        query_pos = seq_len - 1
        inputs[i, query_pos] = key
        targets[i, query_pos] = value
        query_positions.append(query_pos)
    
    print(f"Training for 200 steps...")
    start_time = time.time()
    
    for step in range(200):
        optimizer.zero_grad()
        
        logits = model(inputs)
        
        loss = F.cross_entropy(
            logits[:, -1, :],
            targets[:, -1],
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets[:, -1]).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.2%}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f}s ({200/elapsed:.1f} steps/sec)")
    
    print("\nFinal evaluation:")
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        pred = logits[:, -1, :].argmax(dim=-1)
        acc = (pred == targets[:, -1]).float().mean()
        print(f"  Accuracy: {acc.item():.2%}")


if __name__ == "__main__":
    test_simple_learning()
    test_ar_task()
