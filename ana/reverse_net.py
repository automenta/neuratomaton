"""
A simple network specifically designed for sequence reversal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseNet(nn.Module):
    """
    A network that is explicitly designed to learn sequence reversal
    by combining RNN with explicit position information
    """
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_encoding = nn.Embedding(100, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        
        batch, seq_len = x.shape
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        pos_encoding = self.position_encoding(pos_ids)
        
        x_combined = torch.cat([emb, pos_encoding], dim=-1)
        
        output, _ = self.rnn(x_combined)
        
        logits = self.output(output)
        return logits, []

def test_reverse_net():
    model = ReverseNet(vocab_size=10, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print('Testing ReverseNet on reversal task:')
    print('=' * 60)
    
    # Train
    for epoch in range(5):
        total_loss = 0.0
        steps = 0
        
        for L in [2, 3, 4, 5, 6]:
            for _ in range(50):
                seq = torch.randint(1, 9, (16, L))
                targ = seq.flip(dims=[1])
                
                optimizer.zero_grad()
                logits, _ = model(seq)
                loss = F.cross_entropy(logits.view(-1, 10), targ.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                steps += 1
        
        avg_loss = total_loss / steps
        print(f'epoch {epoch+1}: loss={avg_loss:.3f}')
    
    # Test
    model.eval()
    print()
    print('Evaluation:')
    print('=' * 60)
    
    with torch.no_grad():
        for L_test in [7, 8, 10, 12]:
            accs = []
            for _ in range(50):
                test = torch.randint(1, 9, (16, L_test))
                logits, _ = model(test)
                pred = logits.argmax(-1)
                acc = (pred == test.flip(dims=[1])).float().mean()
                accs.append(acc.item())
            
            print(f'  Length {L_test:2d}: {100*sum(accs)/len(accs):.1f}%')
    
    # Test specific patterns
    print()
    print('Pattern tests:')
    print('=' * 60)
    
    test_patterns = [[1,2,3,4], [4,3,2,1], [1,3,5,7], [2,4,6,8], [1,2,3,4,5]]
    model.eval()
    with torch.no_grad():
        for pattern in test_patterns:
            test = torch.tensor([pattern])
            logits, _ = model(test)
            pred = logits.argmax(-1)
            print(f'{pattern} → {pred[0].tolist()}, targ={list(reversed(pattern))}')

if __name__ == "__main__":
    test_reverse_net()
