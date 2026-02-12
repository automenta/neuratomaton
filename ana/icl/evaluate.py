"""
ICL Tasks for ANA Evaluation

Focus on associative recall and pattern completion (NOT copy/reverse).
"""
import torch
import torch.nn.functional as F
import random


def generate_associative_recall_task(batch_size, num_kv_pairs=1, vocab_size=30, min_noise=10, max_noise=30):
    """
    Generate Associative Recall task matching paper format.
    
    Format: [TOK_KEY K TOK_VAL V]×n + noise + [TOK_QUERY K]
    Goal: Predict the value associated with the query key.
    
    Special tokens:
        TOK_KEY = 1
        TOK_VAL = 2  
        TOK_QUERY = 3
        Content tokens = 4 to vocab_size-1
    """
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    content_tokens = list(range(4, vocab_size))
    
    input_ids = []
    target_ids = []
    
    for _ in range(batch_size):
        keys = random.sample(content_tokens, num_kv_pairs)
        vals = random.sample([t for t in content_tokens if t not in keys], num_kv_pairs)
        
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([TOK_KEY, k, TOK_VAL, v])
        
        noise_len = random.randint(min_noise, max_noise)
        noise = [random.choice(content_tokens) for _ in range(noise_len)]
        seq.extend(noise)
        
        query_idx = random.randint(0, num_kv_pairs - 1)
        query_key = keys[query_idx]
        target_val = vals[query_idx]
        seq.extend([TOK_QUERY, query_key])
        
        input_ids.append(seq)
        target_ids.append(target_val)
    
    max_len = max(len(s) for s in input_ids)
    input_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(input_ids):
        input_tensor[i, :len(seq)] = torch.tensor(seq)
    
    target_tensor = torch.tensor(target_ids, dtype=torch.long)
    
    return input_tensor, target_tensor


def evaluate_kv_recall(model, num_kv_pairs, batch_size=32, vocab_size=30, num_eval=100):
    """Evaluate KV recall accuracy."""
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _ in range(num_eval // batch_size):
            inputs, targets = generate_associative_recall_task(
                batch_size=batch_size,
                num_kv_pairs=num_kv_pairs,
                vocab_size=vocab_size
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits, _ = model(inputs)
            pred = logits[:, -1, :].argmax(dim=-1)
            correct += (pred == targets).sum().item()
            total += batch_size
    
    return correct / total


def evaluate_synergy(model_class, config, num_kv_pairs=12, steps=1000, verbose=True):
    """
    E1: Evaluate HoloLink synergy effect.
    
    Compares:
    - Full ANA (Controller + HoloLink)
    - Controller only (HoloLink disabled)
    - HoloLink only (Controller disabled)
    
    Success: Full ANA > max(ablations) by >10%
    """
    from ana import ANAConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    vocab_size = 30 + num_kv_pairs * 2
    
    configs = {
        'Full ANA': ANAConfig(
            d_model=config.d_model,
            vocab_size=vocab_size,
            state_dim=config.state_dim,
            use_hololink=True,
            use_controller=True,
            use_parallel_scan=True
        ),
        'Controller Only': ANAConfig(
            d_model=config.d_model,
            vocab_size=vocab_size,
            state_dim=config.state_dim,
            use_hololink=False,
            use_controller=True,
            use_parallel_scan=True
        ),
        'HoloLink Only': ANAConfig(
            d_model=config.d_model,
            vocab_size=vocab_size,
            state_dim=config.state_dim,
            use_hololink=True,
            use_controller=False,
            use_parallel_scan=True
        ),
    }
    
    for name, cfg in configs.items():
        if verbose:
            print(f"\n--- {name} ---")
        
        model = model_class(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for step in range(steps):
            inputs, targets = generate_associative_recall_task(
                batch_size=32,
                num_kv_pairs=num_kv_pairs,
                vocab_size=vocab_size,
                min_noise=5,
                max_noise=15
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            optimizer.step()
            
            if verbose and (step + 1) % 200 == 0:
                acc = evaluate_kv_recall(model, num_kv_pairs, batch_size=32, vocab_size=vocab_size, num_eval=64)
                print(f"  Step {step+1}: loss={loss.item():.4f}, acc={100*acc:.1f}%")
        
        final_acc = evaluate_kv_recall(model, num_kv_pairs, batch_size=32, vocab_size=vocab_size, num_eval=200)
        results[name] = final_acc
        
        if verbose:
            print(f"  Final: {100*final_acc:.1f}%")
    
    full_acc = results['Full ANA']
    best_ablation = max(results['Controller Only'], results['HoloLink Only'])
    synergy = full_acc - best_ablation
    
    print("\n" + "="*60)
    print("SYNERGY RESULTS")
    print("="*60)
    for name, acc in results.items():
        print(f"  {name}: {100*acc:.1f}%")
    print(f"\n  Synergy: {100*synergy:.1f}%")
    
    if synergy > 0.10:
        print(f"  ✅ SUCCESS: >10% synergy achieved")
        return True, results
    else:
        print(f"  ❌ FAIL: <10% synergy")
        return False, results


def run_scaling_experiment(model_class, config, num_pairs_list=[1, 2, 4, 6, 8, 10, 12], steps=800, verbose=True):
    """
    E2: Run scaling experiment to find KV capacity.
    
    Success: >80% at 16 pairs, >60% at 8 pairs
    """
    from ana import ANAConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    
    for num_pairs in num_pairs_list:
        if verbose:
            print(f"\n--- {num_pairs} KV Pairs ---")
        
        vocab_size = 30 + num_pairs * 2
        cfg = ANAConfig(
            d_model=config.d_model,
            vocab_size=vocab_size,
            state_dim=config.state_dim,
            use_hololink=True,
            use_controller=True,
            use_parallel_scan=True
        )
        
        model = model_class(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for step in range(steps):
            inputs, targets = generate_associative_recall_task(
                batch_size=32,
                num_kv_pairs=num_pairs,
                vocab_size=vocab_size,
                min_noise=5,
                max_noise=15
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            optimizer.step()
        
        final_acc = evaluate_kv_recall(model, num_pairs, batch_size=32, vocab_size=vocab_size, num_eval=200)
        results[num_pairs] = final_acc
        
        if verbose:
            print(f"  Final accuracy: {100*final_acc:.1f}%")
        
        if final_acc < 0.6 and num_pairs < 16:
            print(f"  Early stop: capacity limit reached")
            break
    
    print("\n" + "="*60)
    print("SCALING RESULTS")
    print("="*60)
    for n, acc in results.items():
        status = "✅" if acc > 0.8 else ("⚠️" if acc > 0.6 else "❌")
        print(f"  {n} pairs: {100*acc:.1f}% {status}")
    
    return results


if __name__ == "__main__":
    from ana import ANAConfig, ANAModel
    
    print("="*60)
    print("E1: SYNERGY EXPERIMENT (12 KV pairs)")
    print("="*60)
    success, results = evaluate_synergy(
        ANAModel,
        ANAConfig(d_model=64, state_dim=64),
        num_kv_pairs=12,
        steps=800
    )
