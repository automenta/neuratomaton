import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import random
import time
import json

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


def run_stage0_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Stage 0 Validation: Simple Associative Recall")
    print("=" * 60)
    
    config = get_bio_config('nano')
    model = create_bio_ana('nano').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    vocab_size = 50
    
    random.seed(42)
    train_kv_pairs = [(i, i + 10) for i in range(1, 16)]  # values: 11-25
    val_kv_pairs = [(i, i + 10) for i in range(26, 36)]    # keys: 26-35, values: 36-45
    
    print(f"Train pairs: {len(train_kv_pairs)}")
    print(f"Val pairs: {len(val_kv_pairs)} (held out)")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    start_time = time.time()
    best_val_acc = 0.0
    history = []
    
    for step in range(200):
        batch_size = 16
        seq_len = 20
        
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            key, value = random.choice(train_kv_pairs)
            input_ids[b, 0] = key
            input_ids[b, 1] = value
            
            for j in range(2, seq_len - 1):
                if random.random() < 0.4:
                    input_ids[b, j] = random.randint(40, 49)  # noise tokens in upper range
            
            input_ids[b, -1] = key
            target_ids[b, -1] = value
        
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits[:, -1, :], target_ids[:, -1])
        loss.backward()
        optimizer.step()
        
        if step % 25 == 0:
            with torch.no_grad():
                train_pred = logits[:, -1, :].argmax(-1)
                train_acc = (train_pred == target_ids[:, -1]).float().mean().item()
            
            val_correct = 0
            val_total = 0
            model.eval()
            for key, value in val_kv_pairs:
                for _ in range(5):
                    seq_len_v = random.randint(12, 25)
                    inp = torch.zeros(1, seq_len_v, dtype=torch.long, device=device)
                    inp[0, 0] = key
                    inp[0, 1] = value
                    for j in range(2, seq_len_v - 1):
                        if random.random() < 0.4:
                            inp[0, j] = random.randint(40, 49)  # noise tokens
                    inp[0, -1] = key
                    
                    with torch.no_grad():
                        pred = model(inp)[0, -1, :].argmax().item()
                    if pred == value:
                        val_correct += 1
                    val_total += 1
            model.train()
            
            val_acc = val_correct / val_total
            elapsed = time.time() - start_time
            
            print(f"Step {step:3d}: loss={loss.item():.4f}, "
                  f"train={train_acc:.0%}, val={val_acc:.0%}, time={elapsed:.1f}s")
            
            history.append({
                'step': step,
                'loss': loss.item(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if val_acc >= 0.98:
                print(f"\n✓ Stage 0 PASSED at step {step}!")
                break
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Steps: {step + 1}")
    print(f"Best val accuracy: {best_val_acc:.0%}")
    print(f"Final val accuracy: {val_acc:.0%}")
    print(f"Time: {time.time() - start_time:.1f}s")
    print(f"Status: {'PASS' if val_acc >= 0.98 else 'NEEDS_MORE_TRAINING'}")
    
    results = {
        'stage': 0,
        'task': 'Simple Associative Recall',
        'train_kv_pairs': len(train_kv_pairs),
        'val_kv_pairs': len(val_kv_pairs),
        'steps': step + 1,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'passed': val_acc >= 0.98,
        'time_s': time.time() - start_time,
        'history': history,
    }
    
    output_dir = Path("results/phase3")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "stage0_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'stage0_results.json'}")
    
    return results


if __name__ == "__main__":
    results = run_stage0_validation()
