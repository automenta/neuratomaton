"""
E2: Memory Capacity Test - Find HoloLink limits
Tests scaling from 1-32 KV pairs with HoloLink only
"""
import torch
import torch.nn.functional as F
import random
from ana import ANAConfig, ANAModel


def generate_kv_task(batch_size, num_pairs, vocab_size, noise_len=10):
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    content = list(range(4, vocab_size))
    
    inputs, targets = [], []
    for _ in range(batch_size):
        keys = random.sample(content, num_pairs)
        vals = random.sample([t for t in content if t not in keys], num_pairs)
        
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([TOK_KEY, k, TOK_VAL, v])
        
        seq.extend(random.choices(content, k=noise_len))
        
        q_idx = random.randint(0, num_pairs - 1)
        seq.extend([TOK_QUERY, keys[q_idx]])
        
        inputs.append(seq)
        targets.append(vals[q_idx])
    
    max_len = max(len(s) for s in inputs)
    x = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, s in enumerate(inputs):
        x[i, :len(s)] = torch.tensor(s)
    
    return x, torch.tensor(targets)


def train_hololink(config, device, curriculum):
    model = ANAModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    vocab_size = config.vocab_size
    
    for num_pairs, steps in curriculum:
        for step in range(steps):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_model(model, num_pairs, vocab_size, device, n_eval=200):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n_eval // 32):
            x, y = generate_kv_task(32, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits[:, -1].argmax(-1)
            correct += (pred == y).sum().item()
    return correct / n_eval


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    vocab_size = 60
    
    config = ANAConfig(
        d_model=64,
        vocab_size=vocab_size,
        state_dim=64,
        key_dim=64,
        use_hololink=True,
        use_controller=False,
        use_parallel_scan=True
    )
    
    print("\n" + "="*60)
    print("HOLOLINK MEMORY CAPACITY TEST")
    print("="*60)
    
    kv_pairs_to_test = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
    results = {}
    
    for target_pairs in kv_pairs_to_test:
        print(f"\n--- Training for {target_pairs} KV pairs ---")
        
        curriculum = []
        for n in [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]:
            if n <= target_pairs:
                steps = 1000 if n == target_pairs else 800
                curriculum.append((n, steps))
        
        model = train_hololink(config, device, curriculum)
        
        acc = evaluate_model(model, target_pairs, vocab_size, device)
        results[target_pairs] = acc
        print(f"  {target_pairs} KV pairs: {100*acc:.1f}%")
        
        if acc < 0.60 and target_pairs > 8:
            print(f"  ⚠️ Early stop: capacity limit detected")
            break
        
        del model
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("MEMORY CAPACITY RESULTS")
    print("="*60)
    print(f"{'KV Pairs':<12} {'Accuracy':<12} {'Status'}")
    print("-"*40)
    for n, acc in sorted(results.items()):
        if acc >= 0.95:
            status = "✅"
        elif acc >= 0.80:
            status = "⚠️"
        else:
            status = "❌"
        print(f"{n:<12} {100*acc:<11.1f}% {status}")
    
    valid = [n for n, acc in results.items() if acc >= 0.80]
    if valid:
        max_capacity = max(valid)
        print(f"\n  Max capacity (≥80%): {max_capacity} KV pairs")
    else:
        print("\n  No capacity >= 80% achieved")


if __name__ == "__main__":
    main()