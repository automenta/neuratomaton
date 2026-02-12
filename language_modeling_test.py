#!/usr/bin/env python
"""
REAL-WORLD TEST: Language Modeling with ANA vs Transformer

Task: Character-level language modeling on a small text corpus.
We use a simple dataset (Shakespeare or similar) and compare:
- ANA (HoloLink) vs Transformer
- At matched parameter counts
- Measured by perplexity (lower = better)

This tests whether HoloLink helps on actual language data, not just synthetic tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# DATASET: Simple character-level text
# ============================================================================

def create_text_dataset():
    """Create a simple text dataset for character-level LM."""
    # Simple text - can be replaced with any corpus
    text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.
""" * 10  # Repeat to get more data
    
    # Build vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    
    # Encode text
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    return data, vocab_size, char_to_idx, idx_to_char

def get_batches(data, batch_size, seq_len):
    """Generate batches for language modeling."""
    n_batches = (len(data) - 1) // (batch_size * seq_len)
    data = data[:n_batches * batch_size * seq_len + 1]
    
    x = data[:-1].view(batch_size, -1)
    y = data[1:].view(batch_size, -1)
    
    batches = []
    for i in range(0, x.shape[1], seq_len):
        if i + seq_len <= x.shape[1]:
            batches.append((x[:, i:i+seq_len], y[:, i:i+seq_len]))
    
    return batches

# ============================================================================
# MODELS
# ============================================================================

class ANALayer(nn.Module):
    def __init__(self, d_model, state_dim, key_dim):
        super().__init__()
        self.in_proj = nn.Linear(d_model, state_dim)
        self.out_proj = nn.Linear(state_dim, d_model)
        self.alpha = nn.Parameter(torch.zeros(state_dim))
        self.beta = nn.Parameter(torch.zeros(state_dim))
        self.k_proj = nn.Linear(state_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(state_dim, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.bind = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        B, S, D = x.shape
        u = self.in_proj(x)
        a = torch.sigmoid(self.alpha).view(1, 1, -1)
        b = torch.sigmoid(self.beta).view(1, 1, -1)
        h = torch.zeros(B, self.alpha.shape[0], device=x.device)
        hs = []
        for t in range(S):
            h = a.squeeze(1) * h + b.squeeze(1) * u[:, t]
            hs.append(h)
        h = torch.stack(hs, dim=1)
        y = self.out_proj(h)
        k = F.normalize(self.k_proj(h), p=2, dim=-1)
        v = self.v_proj(h)
        M = torch.cumsum(F.softplus(self.bind) * k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        q = F.normalize(self.q_proj(x), p=2, dim=-1)
        retrieved = self.norm(self.out((q.unsqueeze(-2) @ M).squeeze(-2)))
        return x + y + retrieved

class ANA(nn.Module):
    def __init__(self, vocab_size, d_model=64, state_dim=64, key_dim=32, n_layers=2, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([ANALayer(d_model, state_dim, key_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq=128):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        # Causal mask
        self.register_buffer('mask', torch.triu(torch.ones(max_seq, max_seq), diagonal=1).bool())
    
    def forward(self, x):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        attn = attn.masked_fill(self.mask[:S, :S], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        x = x + self.out((attn @ v).permute(0, 2, 1, 3).reshape(B, S, D))
        return x + self.ff(self.norm2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, max_seq) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_lm(model, batches, epochs=10, lr=1e-3):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(batches)
        losses.append(avg_loss)
    
    return losses

def compute_perplexity(model, batches):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)

def generate_text(model, idx_to_char, char_to_idx, prompt="To be", length=100):
    model.eval()
    chars = list(prompt)
    x = torch.tensor([[char_to_idx.get(c, 0) for c in chars]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(length):
            logits = model(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            chars.append(idx_to_char[next_idx.item()])
            x = torch.cat([x, next_idx], dim=1)
    
    return ''.join(chars)

# ============================================================================
# MAIN
# ============================================================================

print('='*70)
print('LANGUAGE MODELING: ANA vs Transformer')
print('='*70)

# Load data
print('\nLoading text data...')
data, vocab_size, char_to_idx, idx_to_char = create_text_dataset()
print(f'Vocabulary size: {vocab_size}')
print(f'Text length: {len(data)} characters')

# Split into train/val
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# Create batches
batch_size = 16
seq_len = 64
train_batches = get_batches(train_data, batch_size, seq_len)
val_batches = get_batches(val_data, batch_size, seq_len)
print(f'Training batches: {len(train_batches)}')
print(f'Validation batches: {len(val_batches)}')

# Train ANA
print('\n' + '-'*70)
print('Training ANA (~60K params)...')
ana = ANA(vocab_size, d_model=64, state_dim=64, key_dim=32, n_layers=2, max_seq=seq_len).to(device)
ana_params = count_params(ana)
print(f'ANA parameters: {ana_params:,}')
ana_losses = train_lm(ana, train_batches, epochs=20, lr=3e-3)
ana_ppl = compute_perplexity(ana, val_batches)
print(f'ANA validation perplexity: {ana_ppl:.2f}')

# Generate sample
print('\nANA generated text:')
print(generate_text(ana, idx_to_char, char_to_idx, "To be", 50))

del ana
torch.cuda.empty_cache()

# Train Transformer (matched params)
print('\n' + '-'*70)
print('Training Transformer (~60K params)...')
trans = Transformer(vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq=seq_len).to(device)
trans_params = count_params(trans)
print(f'Transformer parameters: {trans_params:,}')
trans_losses = train_lm(trans, train_batches, epochs=20, lr=3e-3)
trans_ppl = compute_perplexity(trans, val_batches)
print(f'Transformer validation perplexity: {trans_ppl:.2f}')

# Generate sample
print('\nTransformer generated text:')
print(generate_text(trans, idx_to_char, char_to_idx, "To be", 50))

del trans
torch.cuda.empty_cache()

# Train larger Transformer
print('\n' + '-'*70)
print('Training Transformer (~200K params)...')
trans2 = Transformer(vocab_size, d_model=96, n_heads=4, n_layers=4, d_ff=256, max_seq=seq_len).to(device)
trans2_params = count_params(trans2)
print(f'Transformer parameters: {trans2_params:,}')
trans2_losses = train_lm(trans2, train_batches, epochs=20, lr=3e-3)
trans2_ppl = compute_perplexity(trans2, val_batches)
print(f'Transformer validation perplexity: {trans2_ppl:.2f}')

del trans2
torch.cuda.empty_cache()

# Summary
print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)
print(f'\n{"Model":<25} {"Params":<12} {"Perplexity":<12}')
print('-'*50)
print(f'{"ANA (HoloLink)":<25} {ana_params:<12,} {ana_ppl:.2f}')
print(f'{"Transformer (matched)":<25} {trans_params:<12,} {trans_ppl:.2f}')
print(f'{"Transformer (3x larger)":<25} {trans2_params:<12,} {trans2_ppl:.2f}')

# Verdict
print('\n' + '='*70)
print('VERDICT')
print('='*70)

if ana_ppl < trans_ppl:
    improvement = (trans_ppl - ana_ppl) / trans_ppl * 100
    print(f'\n✅ ANA WINS on language modeling!')
    print(f'   ANA perplexity: {ana_ppl:.2f}')
    print(f'   Transformer perplexity: {trans_ppl:.2f}')
    print(f'   Improvement: {improvement:.1f}% lower perplexity')
elif ana_ppl < trans2_ppl:
    print(f'\n⚠️ MIXED RESULTS')
    print(f'   ANA beats matched Transformer but not larger one')
    print(f'   ANA: {ana_ppl:.2f} vs Matched Trans: {trans_ppl:.2f}')
    print(f'   ANA: {ana_ppl:.2f} vs Larger Trans: {trans2_ppl:.2f}')
else:
    print(f'\n❌ ANA does NOT beat Transformer on language modeling')
    print(f'   HoloLink advantage may be task-specific')
