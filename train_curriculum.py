import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import time
import json
import random
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


class ARCurriculum:
    def __init__(self, vocab_size=50, num_kv_pairs=10, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        
        self.kv_pairs = []
        key_space = list(range(1, vocab_size // 2))
        value_space = list(range(vocab_size // 2, vocab_size))
        random.shuffle(key_space)
        random.shuffle(value_space)
        
        for i in range(min(num_kv_pairs, len(key_space), len(value_space))):
            self.kv_pairs.append((key_space[i], value_space[i]))
        
        self.key_to_value = {k: v for k, v in self.kv_pairs}
    
    def generate_batch(self, batch_size, seq_len, device, stage=0):
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        kv_start = 0
        noise_start = 2
        
        if stage == 0:
            num_noise_range = (3, 8)
        elif stage == 1:
            num_noise_range = (8, 15)
        else:
            num_noise_range = (15, 25)
        
        for b in range(batch_size):
            key, value = random.choice(self.kv_pairs[:4]) if stage == 0 else random.choice(self.kv_pairs)
            
            input_ids[b, kv_start] = key
            input_ids[b, kv_start + 1] = value
            
            num_noise = random.randint(*num_noise_range)
            for j in range(noise_start, min(noise_start + num_noise, seq_len - 1)):
                input_ids[b, j] = random.randint(1, self.vocab_size - 1)
            
            query_pos = seq_len - 1
            input_ids[b, query_pos] = key
            target_ids[b, query_pos] = value
        
        return input_ids, target_ids


def train_with_curriculum(
    variant='nano',
    num_steps=500,
    batch_size=16,
    seq_len=16,
    lr=1e-3,
    eval_every=50,
    target_accuracy=0.98,
    output_dir=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Bio-ANA Curriculum Training")
    print("=" * 60)
    print(f"Variant: {variant}")
    print(f"Steps: {num_steps}, Batch: {batch_size}, Seq: {seq_len}")
    print(f"Target accuracy: {target_accuracy:.0%}")
    print()
    
    config = get_bio_config(variant)
    model = create_bio_ana(variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    curriculum = ARCurriculum(vocab_size=config.vocab_size, num_kv_pairs=10)
    
    print(f"KV pairs: {curriculum.kv_pairs[:5]}...")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    start_time = time.time()
    history = []
    best_acc = 0.0
    
    for step in range(num_steps):
        input_ids, target_ids = curriculum.generate_batch(
            batch_size, seq_len, device, stage=0
        )
        
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits[:, -1, :], target_ids[:, -1])
        loss.backward()
        optimizer.step()
        
        if step % eval_every == 0 or step == num_steps - 1:
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == target_ids[:, -1]).float().mean().item()
            
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * batch_size * seq_len / elapsed
            
            print(f"Step {step:4d}: loss={loss.item():.4f}, acc={acc:.0%}, "
                  f"{tokens_per_sec:.0f} tok/s")
            
            history.append({
                'step': step,
                'loss': loss.item(),
                'accuracy': acc,
                'tokens_per_sec': tokens_per_sec,
            })
            
            if acc > best_acc:
                best_acc = acc
                if output_dir:
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
            
            if acc >= target_accuracy:
                print(f"\n✓ Reached target accuracy at step {step}!")
                break
    
    print("\nValidation...")
    model.eval()
    
    val_input, val_target = curriculum.generate_batch(100, seq_len, device, stage=1)
    with torch.no_grad():
        logits = model(val_input)
        pred = logits[:, -1, :].argmax(dim=-1)
        val_acc = (pred == val_target[:, -1]).float().mean().item()
    
    print(f"Validation accuracy (stage 1): {val_acc:.0%}")
    
    results = {
        'variant': variant,
        'num_steps': step + 1,
        'best_train_acc': best_acc,
        'val_acc': val_acc,
        'time_s': time.time() - start_time,
        'history': history,
    }
    
    if output_dir:
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    results = train_with_curriculum(
        variant='nano',
        num_steps=300,
        batch_size=16,
        seq_len=16,
        lr=1e-3,
        eval_every=25,
        target_accuracy=0.98,
        output_dir=Path("results/experiments/curriculum_test"),
    )
    
    print(f"\nFinal results: {results['best_train_acc']:.0%} train, {results['val_acc']:.0%} val")
