"""
Training script for reversal task with curriculum and data augmentation
"""

import torch
import torch.nn.functional as F
from ana import ANAConfig, ANAModel
import random

def generate_reversal_task(length, vocab_size=10):
    data = []
    for _ in range(32):
        seq = [random.randint(1, vocab_size-1) for _ in range(length)]
        data.append(seq)
    return torch.tensor(data)

def train_with_augmentation():
    config = ANAConfig(
        d_model=128,
        vocab_size=10,
        state_dim=128,
        num_layers=2,
        track_count=2,
        use_hololink=True
    )
    
    model = ANAModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.001)
    
    print('Training with reversal augmentation:')
    print('=' * 60)
    
    for epoch in range(10):
        total_loss = 0.0
        steps = 0
        
        for L in [2, 3, 4, 5, 6]:
            for _ in range(20):
                seq = generate_reversal_task(L)
                
                # Train on both forward and backward
                optimizer.zero_grad()
                
                # Forward direction
                logits, _ = model(seq)
                loss = F.cross_entropy(logits.view(-1, 10), seq.flip(dims=[1]).view(-1))
                loss.backward()
                
                # Backward direction
                seq_rev = seq.flip(dims=[1])
                logits_rev, _ = model(seq_rev)
                loss_rev = F.cross_entropy(logits_rev.view(-1, 10), seq.view(-1))
                loss_rev.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += (loss.item() + loss_rev.item())
                steps += 2
        
        avg_loss = total_loss / steps
        
        if (epoch + 1) % 5 == 0:
            print(f'epoch {epoch+1:3d}: loss={avg_loss:.3f}')
    
    model.eval()
    print()
    print('Evaluation:')
    print('=' * 60)
    
    with torch.no_grad():
        for L_test in [7, 8, 10, 12]:
            accs = []
            for _ in range(50):
                test = generate_reversal_task(L_test)
                logits, _ = model(test)
                pred = logits.argmax(-1)
                acc = (pred == test.flip(dims=[1])).float().mean()
                accs.append(acc.item())
            
            print(f'  Length {L_test:2d}: {100*sum(accs)/len(accs):.1f}%')
    
    return model

if __name__ == "__main__":
    model = train_with_augmentation()
