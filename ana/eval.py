"""Evaluate ANA"""
import torch
import torch.nn.functional as F
from ana import ANAConfig, ANAModel

def eval_reverse():
    """Evaluate on reverse task."""
    config = ANAConfig(d_model=32, vocab_size=10, state_dim=32, track_count=2)
    model = ANAModel(config)
    
    # Train first
    train = torch.randint(1, 8, (32, 5))
    targ = train.flip(dims=[1])
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for i in range(30):
        opt.zero_grad()
        logits, _ = model(train)
        F.cross_entropy(logits.view(-1, 10), targ.view(-1)).backward()
        opt.step()
    
    # Test on longer
    test = torch.randint(1, 8, (1, 8))
    test_targ = test.flip(dims=[1])
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(test)
        acc = (logits.argmax(-1) == test_targ).float().mean()
        print(f"Reverse accuracy: {100*acc:.0f}%")

if __name__ == "__main__":
    eval_reverse()
