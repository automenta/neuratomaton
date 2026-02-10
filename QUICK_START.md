# ANA Quick Start Guide

## 5-Minute Tutorial

### Installation
```bash
# Clone and navigate
git clone https://github.com/your-repo/ana.git
cd ana

# Install dependencies
pip install torch numpy
```

### Basic Usage

```python
from ana.models import ANAModel
from ana.config import ANAConfig
import torch

# 1. Configure model
config = ANAConfig(
    d_model=64,           # Model dimension
    num_layers=2,         # Number of layers
    state_dim=64,         # LRU state dimension
    vocab_size=30,        # Vocabulary size
    use_hololink=True,    # Enable holographic memory
    use_controller=True   # Enable dynamic gating
)

# 2. Create model
model = ANAModel(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. Forward pass
x = torch.randint(0, 30, (1, 50))  # Batch=1, Seq=50
logits, info = model(x)
print(f"Output shape: {logits.shape}")
```

### Training Example

```python
from torch.utils.data import DataLoader

# 1. Create dataset (associative recall)
from ana.data import AssociativeRecallDataset
dataset = AssociativeRecallDataset(size=1000, num_kv=8)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 2. Setup training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# 3. Training loop
for epoch in range(20):
    model.train()
    for x, y, mask in loader:
        optimizer.zero_grad()
        logits, _ = model(x)
        
        # Masked loss (only care about final prediction)
        loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = (loss_raw.view(y.size()) * mask).sum() / mask.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
```

### Evaluating

```python
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for x, y, mask in loader:
        logits, _ = model(x)
        preds = torch.argmax(logits, -1)
        
        for i in range(x.size(0)):
            pos = (mask[i] > 0.5).nonzero(as_tuple=True)[0][0]
            if preds[i, pos] == y[i, pos]:
                correct += 1
            total += 1

print(f"Accuracy: {correct/total*100:.1f}%")
```

---

## Common Patterns

### 1. Ablation Study

```python
configs = {
    'baseline': {'use_hololink': False, 'use_controller': False},
    'controller': {'use_hololink': False, 'use_controller': True},
    'hololink': {'use_hololink': True, 'use_controller': False},
    'full': {'use_hololink': True, 'use_controller': True},
}

for name, flags in configs.items():
    model = ANAModel(ANAConfig(**flags))
    acc = train_and_eval(model)
    print(f"{name}: {acc*100:.1f}%")
```

### 2. Scale to Different Sizes

```python
scales = {
    'small': {'d_model': 64, 'num_layers': 2, 'lr': 1e-3},
    'medium': {'d_model': 128, 'num_layers': 3, 'lr': 3e-4},
    'large': {'d_model': 256, 'num_layers': 4, 'lr': 1e-4},
}

for name, cfg in scales.items():
    model = ANAModel(ANAConfig(**cfg))
    acc = train_and_eval(model, lr=cfg['lr'])
    print(f"{name}: {acc*100:.1f}%")
```

### 3. Track Information Extraction

```python
logits, info_log = model(x, return_info=True)

for info in info_log[:10]:  # First 10 timesteps
    print(f"Gate α: {info.get('ga_0', 0):.3f}")
    print(f"Retention: {info.get('ret_gate', 0):.3f}")
```

---

## Hyperparameter Cheatsheet

| Scale | d_model | Layers | State Dim | LR | Epochs | Expected Params |
|-------|---------|--------|-----------|-----|--------|----------------|
| Tiny | 32 | 1 | 32 | 1e-3 | 30 | ~15K |
| Small | 64 | 2 | 64 | 1e-3 | 20 | ~100K |
| Medium | 128 | 3 | 128 | 3e-4 | 30 | ~500K |
| Large | 256 | 4 | 256 | 1e-4 | 40 | ~2M |

**Training Tips**:
- Use smaller LR for larger models
- Increase epochs for higher difficulty
- Scale LR by ~2x for OneCycle schedule

---

## Common Issues

### Issue: Model not training
```python
# Solution 1: Check learning rate
print(optimizer.param_groups[0]['lr'])

# Solution 2: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

# Solution 3: Try smaller LR
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

### Issue: Out of memory
```python
# Solution 1: Reduce batch size
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Solution 2: Use gradient accumulation
accum_steps = 4
# ... accumulate gradients for 4 batches before step()

# Solution 3: Reduce d_model
config.d_model = 48  # Instead of 64
```

### Issue: Slow training
```python
# Solution 1: Use OneCycle LR (faster convergence)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr*3, epochs=20, steps_per_epoch=len(loader)
)

# Solution 2: Reduce num_kv in dataset
dataset = AssociativeRecallDataset(num_kv=6)  # Instead of 8

# Solution 3: Use fewer epochs
for epoch in range(10):  # Instead of 20
```

---

## Model Variants

### Minimal (15K params)
```python
model = ANAModel(ANAConfig(
    d_model=32, num_layers=1, state_dim=32,
    track_count=1, vocab_size=30
))
```

### Standard (100K params)
```python
model = ANAModel(ANAConfig(
    d_model=64, num_layers=2, state_dim=64,
    track_count=2, vocab_size=30
))
```

### Large (500K params)
```python
model = ANAModel(ANAConfig(
    d_model=128, num_layers=3, state_dim=128,
    track_count=2, vocab_size=30
))
```

---

## Quick Results Reference

| Task | Params | Expected Acc |
|------|--------|--------------|
| 4 KV | 100K | 99% |
| 8 KV | 100K | 99% |
| 12 KV | 100K | 96% |
| 8 KV | 15K | 62% |
| 8 KV | 25K | 68% |

---

## Saving/Loading

```python
# Save
torch.save(model.state_dict(), 'ana_model.pt')

# Load
model = ANAModel(ANAConfig(d_model=64, num_layers=2))
model.load_state_dict(torch.load('ana_model.pt'))
```

---

## Next Steps

1. **Read**: `DOCUMENTATION.md` - Complete API reference
2. **Explore**: `APPLICATIONS.md` - 10 use cases with examples
3. **Improve**: `IMPROVEMENT_GUIDE.md` - 14 strategies for better results
4. **Results**: `COMPREHENSIVE_RESULTS.md` - Full experimental validation
5. **Experiment**: Modify `experiments/` scripts for your tasks

---

## FAQ

**Q: What's the minimal parameter count?**
A: ~15K params with d_model=32, num_layers=1

**Q: How does ANA compare to Transformer?**
A: 2-3x higher accuracy at 10-30K params, similar at 100K+

**Q: When should I use ANA vs Transformer?**
A: Use ANA for associative recall, edge devices, or <100K params. Use Transformer for general sequence modeling.

**Q: Can ANA handle non-associative tasks?**
A: It can, but works best on key-value retrieval tasks.

**Q: What's the best learning rate?**
A: Scale-dependent: 1e-3 (small), 3e-4 (medium), 1e-4 (large)

**Q: How do I get 100% accuracy?**
A: Use medium config (128d, 3 layers) with lr=3e-4, 30 epochs

---

## Quick Reference Commands

```bash
# Train model
python experiments/run_all.py

# Run specific experiment
python experiments/exp_synergy_kv.py

# Check results
cat archive/experiments/synergy_by_kv.json | python -m json.tool

# Run tests
python -m pytest tests/ -v

# Count parameters
python -c "from ana.models import ANAModel; print(sum(p.numel() for p in ANAModel().parameters()))"
```

---

## Support

- **Documentation**: `DOCUMENTATION.md`
- **Applications**: `APPLICATIONS.md`
- **Improvements**: `IMPROVEMENT_GUIDE.md`
- **Results**: `COMPREHENSIVE_RESULTS.md`
- **Examples**: `experiments/` directory
