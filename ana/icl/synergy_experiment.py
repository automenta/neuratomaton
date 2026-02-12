"""
E1: Synergy Experiment with Curriculum Training
"""
import torch
import torch.nn.functional as F
import random
from ana import ANAConfig, ANAModel


def generate_kv_task(batch_size, num_pairs, vocab_size, noise_range=(5, 15)):
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    content = list(range(4, vocab_size))
    
    inputs, targets = [], []
    for _ in range(batch_size):
        keys = random.sample(content, num_pairs)
        vals = random.sample([t for t in content if t not in keys], num_pairs)
        
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([TOK_KEY, k, TOK_VAL, v])
        
        noise_len = random.randint(*noise_range)
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


def train_with_curriculum(config, device, curriculum=[(4, 200), (6, 200), (8, 200), (10, 200), (12, 300)]):
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
    vocab_size = 100
    
    configs = {
        'Full ANA': ANAConfig(d_model=64, vocab_size=vocab_size, state_dim=64, 
                              use_hololink=True, use_controller=True, use_parallel_scan=True),
        'Controller Only': ANAConfig(d_model=64, vocab_size=vocab_size, state_dim=64,
                                     use_hololink=False, use_controller=True, use_parallel_scan=True),
        'HoloLink Only': ANAConfig(d_model=64, vocab_size=vocab_size, state_dim=64,
                                   use_hololink=True, use_controller=False, use_parallel_scan=True),
    }
    
    results = {}
    
    for name, cfg in configs.items():
        print(f"\n--- {name} ---")
        model = train_with_curriculum(cfg, device)
        
        acc_12 = evaluate_model(model, 12, vocab_size, device)
        results[name] = acc_12
        print(f"  12 KV pairs: {100*acc_12:.1f}%")
    
    full = results['Full ANA']
    best_ablation = max(results['Controller Only'], results['HoloLink Only'])
    synergy = full - best_ablation
    
    print("\n" + "="*60)
    print("SYNERGY RESULTS")
    print("="*60)
    for name, acc in results.items():
        print(f"  {name}: {100*acc:.1f}%")
    print(f"\n  Synergy: {100*synergy:.1f}%")
    
    if synergy > 0.10:
        print("  ✅ SUCCESS")
        return True
    else:
        print("  ❌ FAIL")
        return False


if __name__ == "__main__":
    main()
