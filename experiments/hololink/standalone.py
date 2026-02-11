"""
HoloLink Standalone Experiments

Tests whether holographic memory can be used as a drop-in replacement
for attention or key-value caches in existing architectures.

Key Experiments:
1. HoloLink vs Attention comparison
2. Retrieval-Augmented Generation (RAG)
3. Memory augmentation for existing models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ana.models_v3 import FaultTraceBuffer
from ana.config_v2 import ANAv2Config


class HoloLinkMemory(nn.Module):
    def __init__(self, key_dim=64, value_dim=64, capacity=1000):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        
        # Holographic memory (outer product storage)
        self.register_buffer('memory', torch.zeros(capacity, value_dim))
        self.register_buffer('write_idx', torch.zeros(1, dtype=torch.long))
    
    def write(self, keys, values):
        """Write key-value pairs to holographic memory"""
        batch_size = keys.size(0)
        
        for b in range(batch_size):
            idx = self.write_idx.item()
            # Simple write: overwrite at current index
            self.memory[idx] = values[b]
            self.write_idx[0] = (self.write_idx + 1) % self.capacity
    
    def read(self, query):
        """Read from memory using associative retrieval"""
        # Compute similarity (dot product for simplicity)
        similarity = torch.matmul(query, self.memory.T)  # [batch, capacity]
        
        # Softmax weighting
        weights = F.softmax(similarity / np.sqrt(self.key_dim), dim=-1)
        
        # Weighted sum
        output = torch.matmul(weights, self.memory)  # [batch, value_dim]
        
        return output, weights


class AttentionMemory(nn.Module):
    """Standard attention mechanism for comparison"""
    def __init__(self, key_dim=64, value_dim=64, capacity=1000):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        
        # Key and value storage
        self.register_buffer('keys', torch.zeros(capacity, key_dim))
        self.register_buffer('values', torch.zeros(capacity, value_dim))
        self.register_buffer('write_idx', torch.zeros(1, dtype=torch.long))
    
    def write(self, keys, values):
        """Write key-value pairs"""
        batch_size = keys.size(0)
        
        for b in range(batch_size):
            idx = self.write_idx.item()
            self.keys[idx] = keys[b]
            self.values[idx] = values[b]
            self.write_idx[0] = (self.write_idx + 1) % self.capacity
    
    def read(self, query):
        """Read using attention mechanism"""
        # Compute similarity
        similarity = torch.matmul(query, self.keys.T) / np.sqrt(self.key_dim)
        
        # Attention weights
        weights = F.softmax(similarity, dim=-1)
        
        # Weighted sum
        output = torch.matmul(weights, self.values)
        
        return output, weights


class MemoryComparisonModel(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, memory_type='hololink', capacity=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.memory_type = memory_type
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Memory layer
        if memory_type == 'hololink':
            self.memory = HoloLinkMemory(key_dim=d_model, value_dim=d_model, capacity=capacity)
        else:
            self.memory = AttentionMemory(key_dim=d_model, value_dim=d_model, capacity=capacity)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True),
            num_layers=2
        )
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, memory_keys=None, memory_values=None):
        # Embed
        x = self.embedding(input_ids)  # [batch, seq, d_model]
        
        # Write to memory if provided
        if memory_keys is not None and memory_values is not None:
            self.memory.write(memory_keys, memory_values)
        
        # Process through transformer
        h = self.transformer(x)
        
        # Read from memory
        query = h.mean(dim=1)  # [batch, d_model]
        memory_output, weights = self.memory.read(query)
        
        # Combine transformer output with memory
        combined = h + memory_output.unsqueeze(1)
        
        # Output
        logits = self.output(combined)
        
        return logits, weights


class RAGDataset(Dataset):
    def __init__(self, num_samples=1000, vocab_size=100, seq_len=32, num_docs=100):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_docs = num_docs
        
        # Generate documents (memory store)
        self.documents = []
        for _ in range(num_docs):
            doc = torch.randint(1, vocab_size, (np.random.randint(10, 30),))
            self.documents.append(doc)
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Query (some tokens from a document)
            doc_idx = np.random.randint(len(self.documents))
            doc = self.documents[doc_idx]
            
            query_start = np.random.randint(max(0, len(doc) - 5))
            query = doc[query_start:query_start + 3]
            
            # Context (random tokens)
            context = torch.randint(1, self.vocab_size, (self.seq_len - len(query) - 1,))
            
            # Target (next token from document)
            if query_start + 4 < len(doc):
                target = doc[query_start + 3]
            else:
                target = torch.randint(1, self.vocab_size, (1,)).item()
            
            sequence = torch.cat([context, query, torch.tensor([target])])
            
            input_ids = sequence[:-1]
            target_id = sequence[-1]
            
            data.append((input_ids, target_id, doc_idx))
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx][:2]  # Return input, target (ignore doc_idx for now)


class RetrievalAugmentedModel(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, memory_type='hololink', num_docs=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_docs = num_docs
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Memory for documents
        if memory_type == 'hololink':
            self.memory = HoloLinkMemory(key_dim=d_model, value_dim=d_model*2, capacity=num_docs*2)
        else:
            self.memory = AttentionMemory(key_dim=d_model, value_dim=d_model*2, capacity=num_docs*2)
        
        self.query_projector = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def index_documents(self, documents):
        """Index documents into memory"""
        with torch.no_grad():
            for doc in documents:
                doc_emb = self.embedding(doc)
                doc_key = doc_emb.mean(dim=0)
                doc_value = doc_emb.mean(dim=0)
                self.memory.write(doc_key.unsqueeze(0), doc_value.unsqueeze(0))
    
    def forward(self, input_ids):
        # Embed input
        x = self.embedding(input_ids)  # [batch, seq, d_model]
        
        # Create query from input
        query = x.mean(dim=1)  # [batch, d_model]
        query = self.query_projector(query)
        
        # Retrieve from memory
        retrieved, weights = self.memory.read(query)
        
        # Combine with input
        combined = x + retrieved.unsqueeze(1)
        combined = combined.mean(dim=1)  # [batch, d_model]
        
        # Output
        logits = self.output(combined)
        
        return logits, weights


def compare_memory_mechanisms():
    print("="*80)
    print("HOLINK VS ATTENTION COMPARISON")
    print("="*80)
    
    results = {}
    
    # Create dataset
    dataset = RAGDataset(num_samples=1000, vocab_size=100, num_docs=100)
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 1. HoloLink model
    print("\n1. Training HoloLink Model...")
    hololink_model = MemoryComparisonModel(
        vocab_size=100, d_model=64, memory_type='hololink', capacity=200
    )
    optimizer = torch.optim.AdamW(hololink_model.parameters(), lr=1e-3)
    
    hololink_history = train_model(hololink_model, train_loader, val_loader, optimizer, epochs=10)
    results['hololink'] = hololink_history
    
    # 2. Attention model
    print("\n2. Training Attention Model...")
    attention_model = MemoryComparisonModel(
        vocab_size=100, d_model=64, memory_type='attention', capacity=200
    )
    optimizer = torch.optim.AdamW(attention_model.parameters(), lr=1e-3)
    
    attention_history = train_model(attention_model, train_loader, val_loader, optimizer, epochs=10)
    results['attention'] = attention_history
    
    # 3. RAG comparison
    print("\n3. Retrieval-Augmented Generation Comparison...")
    
    # Index documents
    rag_documents = [dataset.documents[i] for i in range(min(50, len(dataset.documents)))]
    
    # HoloLink RAG
    rag_hololink = RetrievalAugmentedModel(
        vocab_size=100, d_model=64, memory_type='hololink', num_docs=50
    )
    rag_hololink.index_documents(rag_documents)
    optimizer = torch.optim.AdamW(rag_hololink.parameters(), lr=1e-3)
    
    rag_hololink_history = train_model(rag_hololink, train_loader, val_loader, optimizer, epochs=10)
    results['rag_hololink'] = rag_hololink_history
    
    # Attention RAG
    rag_attention = RetrievalAugmentedModel(
        vocab_size=100, d_model=64, memory_type='attention', num_docs=50
    )
    rag_attention.index_documents(rag_documents)
    optimizer = torch.optim.AdamW(rag_attention.parameters(), lr=1e-3)
    
    rag_attention_history = train_model(rag_attention, train_loader, val_loader, optimizer, epochs=10)
    results['rag_attention'] = rag_attention_history
    
    # 4. Memory efficiency comparison
    print("\n4. Memory Efficiency Analysis...")
    results['memory_efficiency'] = analyze_memory_efficiency(hololink_model, attention_model)
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'experiments' / 'hololink'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'hololink_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, history in results.items():
        if name == 'memory_efficiency':
            continue
        
        final_acc = history['val_acc'][-1] if 'val_acc' in history else 0
        best_acc = max(history['val_acc']) if 'val_acc' in history else 0
        print(f"\n{name.upper()}:")
        print(f"  Final Accuracy: {final_acc:.2%}")
        print(f"  Best Accuracy: {best_acc:.2%}")
    
    # Compare HoloLink vs Attention
    hl_final = results['hololink']['val_acc'][-1]
    att_final = results['attention']['val_acc'][-1]
    
    print(f"\nDirect Comparison:")
    print(f"  HoloLink: {hl_final:.2%}")
    print(f"  Attention: {att_final:.2%}")
    print(f"  Difference: {(hl_final - att_final):+.2%}")
    
    if hl_final > att_final + 0.02:  # 2% advantage
        print(f"\n✓ HOLINK OUTPERFORMS ATTENTION")
        print(f"  Publication: Architecture Workshop / ICLR")
        print(f"  Value: More efficient memory mechanism")
    elif abs(hl_final - att_final) < 0.02:
        print(f"\n≈ HOLINK COMPARABLE TO ATTENTION")
        print(f"  Publication: Efficiency comparison paper")
        print(f"  Value: Alternative with different trade-offs")
    else:
        print(f"\n⚠ ATTENTION OUTPERFORMS HOLINK")
        print(f"  Publication: arXiv (comparison study)")
        print(f"  Value: Understanding memory mechanism limitations")
    
    print(f"\n✓ Results saved: {output_dir / 'hololink_comparison.json'}")
    
    return results


def train_model(model, train_loader, val_loader, optimizer, epochs=10):
    """Simple training loop"""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), \
                              targets.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            
            # Reshape for classification
            batch, seq, vocab = logits.shape
            logits = logits[:, -1, :]  # Use last token only
            loss = F.cross_entropy(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=-1)
            train_correct += (pred == targets).sum().item()
            train_total += targets.size(0)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), \
                                  targets.to('cuda' if torch.cuda.is_available() else 'cpu')
                
                logits, _ = model(inputs)
                logits = logits[:, -1, :]
                loss = F.cross_entropy(logits, targets)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=-1)
                val_correct += (pred == targets).sum().item()
                val_total += targets.size(0)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Val Acc = {history['val_acc'][-1]:.2%}")
    
    return history


def analyze_memory_efficiency(model1, model2):
    """Compare memory efficiency of two models"""
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    
    return {
        'hololink_params': params1,
        'attention_params': params2,
        'param_ratio': params2 / params1 if params1 > 0 else 0
    }


if __name__ == '__main__':
    results = compare_memory_mechanisms()
    print("\n✓ HoloLink experiments complete!")
