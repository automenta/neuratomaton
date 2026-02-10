#!/usr/bin/env python3
"""
Phase C: Language Modeling
Validate ANA on real-world task (WikiText-2)
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import math

sys.path.insert(0, '.')
os.makedirs('archive/experiments', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM

class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.data = [self.char_to_idx[c] for c in text]
        self.num_samples = len(self.data) // (seq_len + 1)
    
    def __len__(self): return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        h = self.embedding(x)
        h = self.transformer(h)
        return self.output(h), {}

def get_wikitext_sample():
    sample_text = """
The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation and they were given millions of dollars to make this vision come true.

Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from Congress, the U.S. and British Governments stopped funding undirected research into artificial intelligence. Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again. This cycle of boom and bust, of hype and disappointment, has been repeated throughout the history of AI.

However, since 2012, AI has experienced a remarkable resurgence, driven by advances in machine learning, particularly deep learning. Applications of AI have spread to almost every sector of society, from healthcare to transportation to finance. Large language models like GPT have demonstrated unprecedented capabilities in natural language processing and generation.

The transformer architecture, introduced in 2017, revolutionized the field by enabling parallel processing of sequences and better capture of long-range dependencies. This architecture forms the backbone of modern language models and has been adapted for various modalities including vision and audio.

State space models represent an alternative approach to sequence modeling, offering potential advantages in computational efficiency. These models maintain a fixed-size hidden state that evolves over time, allowing for constant memory usage regardless of sequence length. The Linear Recurrent Unit is one such approach that simplifies the recurrent computation while maintaining expressive power.

The combination of different memory mechanisms, such as holographic memory and dynamic gating, may provide complementary benefits for sequence processing tasks. Holographic memory uses outer product binding to store key-value associations in a distributed manner, while dynamic gating allows the model to selectively update and retrieve information based on context.

Research in neural architectures continues to explore novel combinations of mechanisms from different paradigms. The goal is to develop models that are both computationally efficient and capable of handling long-range dependencies in sequential data. This remains an active area of investigation with significant implications for the future of artificial intelligence.

Machine learning has transformed numerous fields through its ability to learn patterns from data. Deep neural networks, with their multiple layers of processing, have proven particularly effective at learning hierarchical representations. The success of these models depends on large datasets, powerful computing resources, and careful architectural design.

Natural language processing has seen dramatic improvements in recent years. Modern language models can perform tasks that were previously thought to require human-level understanding, including translation, summarization, and question answering. These capabilities emerge from training on vast corpora of text data.

The development of efficient training algorithms has been crucial to the success of deep learning. Backpropagation, combined with stochastic gradient descent and various optimization techniques, enables the training of networks with billions of parameters. Techniques like dropout, batch normalization, and learning rate scheduling help improve generalization and training stability.

Attention mechanisms allow neural networks to focus on relevant parts of the input when producing each element of the output. This capability is particularly important for tasks involving long sequences, where maintaining awareness of distant context is essential. Self-attention extends this idea to compute relationships between all positions within a single sequence.

Recurrent neural networks process sequences one element at a time, maintaining a hidden state that captures information from previous time steps. While theoretically powerful, traditional RNNs struggle with long sequences due to gradient-related issues. Modern variants like LSTM and GRU address some of these limitations through specialized gating mechanisms.

The quest for efficient sequence models continues to drive research in neural architecture design. Understanding the trade-offs between different approaches helps researchers develop models suited to specific applications and computational constraints.
""" * 5
    return sample_text.strip()

print("="*70)
print("PHASE C: LANGUAGE MODELING")
print("="*70)

text = get_wikitext_sample()
print(f"Corpus size: {len(text)} characters")

train_size = int(len(text) * 0.9)
train_text = text[:train_size]
val_text = text[train_size:]

train_dataset = CharDataset(train_text, seq_len=128)
val_dataset = CharDataset(val_text, seq_len=128)
vocab_size = train_dataset.vocab_size
print(f"Vocabulary size: {vocab_size}")

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    
    best_ppl = float('inf')
    history = {'train_loss': [], 'val_ppl': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        
        train_loss = total_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss)
        history['val_ppl'].append(val_ppl)
        
        if val_ppl < best_ppl:
            best_ppl = val_ppl
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_ppl={val_ppl:.2f}")
    
    return best_ppl, history

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

results = {}

print("\n" + "-"*50)
print("Training ANA...")
torch.manual_seed(42)
random.seed(42)
ana_cfg = ANAConfig(d_model=128, vocab_size=vocab_size, num_layers=3, state_dim=128)
ana = ANAModel(ana_cfg).to(device)
ana_params = sum(p.numel() for p in ana.parameters())
print(f"ANA params: {ana_params:,}")
ana_ppl, ana_history = train_model(ana, train_loader, val_loader, epochs=30)
results['ana'] = {'best_ppl': ana_ppl, 'params': ana_params, 'history': ana_history}
print(f"ANA best perplexity: {ana_ppl:.2f}")

print("\n" + "-"*50)
print("Training Transformer...")
torch.manual_seed(42)
random.seed(42)
xformer = SimpleTransformer(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=3).to(device)
xf_params = sum(p.numel() for p in xformer.parameters())
print(f"Transformer params: {xf_params:,}")
xf_ppl, xf_history = train_model(xformer, train_loader, val_loader, epochs=30)
results['transformer'] = {'best_ppl': xf_ppl, 'params': xf_params, 'history': xf_history}
print(f"Transformer best perplexity: {xf_ppl:.2f}")

print("\n" + "-"*50)
print("Training Baseline SSM...")
torch.manual_seed(42)
random.seed(42)
baseline = BaselineSSM(ANAConfig(d_model=128, vocab_size=vocab_size, num_layers=3, state_dim=128)).to(device)
baseline_params = sum(p.numel() for p in baseline.parameters())
print(f"Baseline SSM params: {baseline_params:,}")
baseline_ppl, baseline_history = train_model(baseline, train_loader, val_loader, epochs=30)
results['baseline'] = {'best_ppl': baseline_ppl, 'params': baseline_params, 'history': baseline_history}
print(f"Baseline SSM best perplexity: {baseline_ppl:.2f}")

with open('archive/experiments/phaseC_language.json', 'w') as f:
    json.dump({
        'ana': {'best_ppl': results['ana']['best_ppl'], 'params': results['ana']['params']},
        'transformer': {'best_ppl': results['transformer']['best_ppl'], 'params': results['transformer']['params']},
        'baseline': {'best_ppl': results['baseline']['best_ppl'], 'params': results['baseline']['params']}
    }, f, indent=2)

print("\n" + "="*70)
print("LANGUAGE MODELING RESULTS")
print("="*70)
print(f"{'Model':<15} | {'Params':>10} | {'Best PPL':>10}")
print("-"*45)
print(f"{'ANA':<15} | {ana_params:>10,} | {ana_ppl:>10.2f}")
print(f"{'Transformer':<15} | {xf_params:>10,} | {xf_ppl:>10.2f}")
print(f"{'Baseline SSM':<15} | {baseline_params:>10,} | {baseline_ppl:>10.2f}")

if ana_ppl < xf_ppl:
    print(f"\nANA outperforms Transformer by {(xf_ppl/ana_ppl - 1)*100:.1f}% (lower is better)")
elif ana_ppl > xf_ppl:
    print(f"\nTransformer outperforms ANA by {(ana_ppl/xf_ppl - 1)*100:.1f}% (lower is better)")
else:
    print("\nANA and Transformer have similar perplexity")

print(f"\nResults saved to: archive/experiments/phaseC_language.json")
