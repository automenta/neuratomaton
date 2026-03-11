#!/usr/bin/env python
"""
EqProp for Modular Architecture Training

Tests whether Equilibrium Propagation's local learning can solve
the gradient interference problem in modular architectures.

Uses proper bioplausible API with ModelConfig.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from bioplausible.models.base import ModelConfig
from bioplausible.models import StandardEqProp, LoopedMLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3

def gen(batch, pairs, vocab_size=60):
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

def evaluate_eqprop(model, pairs, vocab_size=60, n=50):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n):
            bx, by = gen(32, pairs, vocab_size)
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            correct += (out.argmax(1) == by).sum().item()
            total += by.size(0)
    model.train()
    return correct / total

print('\n' + '='*60)
print('EqProp for Associative Memory Task')
print('='*60)

vocab_size = 60
d_model = 64

# Create EqProp model with proper config
config = ModelConfig(
    name='eqprop_assoc',
    input_dim=d_model,
    output_dim=vocab_size,
    hidden_dims=[128, 128],
    beta=0.5,
    equilibrium_steps=15,
    learning_rate=0.01,
)

class EqPropAssocModel(nn.Module):
    """Associative memory model with EqProp core."""
    def __init__(self, vocab_size, d_model, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.eqprop_core = StandardEqProp(config)
        self.vocab_size = vocab_size
        self.d_model = d_model
        
    def forward(self, x):
        # x: [batch, seq]
        emb = self.embedding(x)  # [batch, seq, d_model]
        # Use last position for classification
        last = emb[:, -1, :]  # [batch, d_model]
        return self.eqprop_core(last)
    
    def train_step(self, x, y):
        """Custom train step that combines embedding gradient with EqProp."""
        # Get embeddings
        emb = self.embedding(x)
        last = emb[:, -1, :]
        
        # EqProp contrastive training
        target = torch.zeros(y.size(0), self.vocab_size, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        
        # Free phase
        with torch.no_grad():
            self.eqprop_core.forward(last, beta=0.0)
            free_activations = self.eqprop_core._last_activations
        
        # Nudged phase
        with torch.no_grad():
            self.eqprop_core.forward(last, beta=self.eqprop_core.beta, target=target)
            nudged_activations = self.eqprop_core._last_activations
        
        # Update EqProp weights (contrastive Hebbian)
        self.eqprop_core.optimizer.zero_grad()
        with torch.no_grad():
            for i, layer in enumerate(self.eqprop_core.layers):
                h_prev_free = free_activations[i]
                h_post_free = free_activations[i + 1]
                h_prev_nudged = nudged_activations[i]
                h_post_nudged = nudged_activations[i + 1]
                
                prod_nudged = torch.matmul(h_post_nudged.T, h_prev_nudged)
                prod_free = torch.matmul(h_post_free.T, h_prev_free)
                dW = (prod_nudged - prod_free) / self.eqprop_core.beta / last.size(0)
                
                # Set gradient
                if hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, 'weight'):
                    w_param = layer.parametrizations.weight.original
                else:
                    w_param = layer.weight
                
                if w_param.grad is None:
                    w_param.grad = -dW
                else:
                    w_param.grad += -dW
                
                if layer.bias is not None:
                    db = (h_post_nudged - h_post_free).sum(0) / self.eqprop_core.beta / last.size(0)
                    if layer.bias.grad is None:
                        layer.bias.grad = -db
                    else:
                        layer.bias.grad += -db
        
        self.eqprop_core.optimizer.step()
        
        # Also train embedding with regular backprop
        self.embedding.train()
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        
        # Return metrics
        pred = out.argmax(1)
        acc = (pred == y).float().mean().item()
        return {'loss': loss.item(), 'accuracy': acc}

# Create model
model = EqPropAssocModel(vocab_size, d_model, config).to(device)

# Add separate optimizer for embeddings
emb_optimizer = torch.optim.Adam(model.embedding.parameters(), lr=1e-3)

print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print(f'  Embedding: {sum(p.numel() for p in model.embedding.parameters()):,}')
print(f'  EqProp core: {sum(p.numel() for p in model.eqprop_core.parameters()):,}')

curriculum = [(1, 300), (2, 300), (4, 300), (6, 300), (8, 300), (10, 300), (12, 400)]

print('\nTraining with EqProp (local contrastive learning)...')
for pairs, steps in curriculum:
    for step in range(steps):
        bx, by = gen(32, pairs, vocab_size)
        bx, by = bx.to(device), by.to(device)
        
        emb_optimizer.zero_grad()
        metrics = model.train_step(bx, by)
        emb_optimizer.step()
    
    acc = evaluate_eqprop(model, pairs, vocab_size, n=20)
    bar = '\u2588' * int(acc * 20)
    print(f'  {pairs:2d} pairs: {100*acc:5.1f}% {bar}', flush=True)

final = evaluate_eqprop(model, 12, vocab_size, n=50)
print(f'\n>>> EqProp Final: {100*final:.1f}%')

if final > 0.85:
    print('\n✅ EqProp successfully learns associative memory!')
elif final > 0.5:
    print('\n⚠️ EqProp shows partial learning')
else:
    print('\n❌ EqProp struggles with this task')
