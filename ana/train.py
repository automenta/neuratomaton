"""
Simple training loop for ANA
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ana import ANAConfig, ANAModel

def train_copy(steps=50, lr=1e-2):
    """Train on copy task."""
    config = ANAConfig(d_model=32, vocab_size=10, state_dim=32, track_count=2)
    model = ANAModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Simple copy data
    train = torch.randint(1, 10, (64, 6))
    
    for i in range(steps):
        optimizer.zero_grad()
        logits, _ = model(train)
        loss = F.cross_entropy(logits.view(-1, 10), train.view(-1), ignore_index=0)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == train).float().mean()
            print(f"  Step {i+1}: loss={loss.item():.4f}, acc={100*acc:.0f}%")
    
    return model

if __name__ == "__main__":
    train_copy()
