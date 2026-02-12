"""
Fast EqProp experiment - reduced steps for quick validation
"""

import torch
import torch.nn.functional as F
import random
import sys
sys.path.insert(0, '/home/me/ana')
from ana.eqprop_seq import EqPropANA

def fast_eqprop_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 60
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    
    def gen(batch, pairs):
        content = list(range(4, vocab_size))
        x, y = [], []
        for _ in range(batch):
            keys = random.sample(content, min(pairs, len(content)))
            vals = random.sample([t for t in content if t not in keys], min(pairs, len(content)))
            seq = []
            for k, v in zip(keys, vals):
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            seq.extend(random.choices(content, k=10))
            q = random.randint(0, len(keys)-1)
            seq.extend([TOK_QUERY, keys[q]])
            x.append(seq)
            y.append(vals[q])
        mx = max(len(s) for s in x)
        t = torch.zeros(batch, mx, dtype=torch.long)
        for i, s in enumerate(x):
            t[i, :len(s)] = torch.tensor(s)
        return t, torch.tensor(y)
    
    def evaluate(model, pairs, n=20):
        model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(n):
                bx, by = gen(16, pairs)
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                correct += (logits[:, -1].argmax(-1) == by).sum().item()
        model.train()
        return correct / (n * 16)
    
    print('='*60)
    print('FAST EqProp Experiment')
    print('='*60)
    print(f'Device: {device}')
    print()
    
    model = EqPropANA(vocab_size=vocab_size, d_model=32, hidden_dim=64, max_steps=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print()
    
    curriculum = [(1, 50), (2, 50), (4, 50), (6, 50), (8, 50), (10, 50), (12, 100)]
    
    results = []
    for pairs, steps in curriculum:
        print(f'Training {pairs} pairs...', end=' ', flush=True)
        for step in range(steps):
            bx, by = gen(16, pairs)
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits[:, -1, :], by)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model, pairs, n=20)
        results.append((pairs, acc))
        status = '✅' if acc > 0.8 else ('⚠️' if acc > 0.5 else '❌')
        print(f'{100*acc:.1f}% {status}')
    
    print()
    print('='*60)
    print('SUMMARY')
    print('='*60)
    for pairs, acc in results:
        print(f'  {pairs:2d} pairs: {100*acc:5.1f}%')
    
    final_acc = results[-1][1] if results else 0
    print()
    if final_acc > 0.9:
        print('>>> BREAKTHROUGH: EqProp works! <<<')
    elif final_acc > 0.7:
        print('>>> PROMISING: Better than backprop (8-9%) <<<')
    elif final_acc > 0.3:
        print('>>> PARTIAL: Some improvement <<<')
    else:
        print('>>> FAILED: EqProp does not solve interference <<<')
    
    return results

if __name__ == "__main__":
    fast_eqprop_test()
