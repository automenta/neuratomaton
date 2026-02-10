# ANA Documentation

## Architecture Overview

ANA (Adaptive Neural Automaton) is a neural architecture designed for efficient associative recall through synergistic combination of:
1. **Controller** - Dynamic gating mechanism that modulates retention behavior
2. **HoloLink** - Holographic memory using outer-product binding for key-value storage

## Core Components

### Linear Recurrent Unit (LRU)
The foundational state-space model that provides efficient sequence processing.

```
h_t = α * h_{t-1} + β * u_t
```

Where:
- `h_t` is the hidden state at time t
- `α` (alpha) controls retention/forgetting
- `β` (beta) controls input integration
- `u_t` is the input projection

### Controller
Learns dynamic gating signals that modulate LRU behavior per timestep.

```
α_t = sigmoid(α_static + gate_α_t)
β_t = sigmoid(β_static + gate_β_t)
```

The gates allow the model to adaptively control:
- When to retain information (via α)
- How strongly to incorporate new input (via β)

### HoloLink
Holographic memory mechanism using outer-product binding for associative storage.

```
M_t = decay * M_{t-1} + binding_strength * (k_t ⊗ v_t)
retrieved = q_t · M_t
```

Where:
- `M_t` is the holographic memory matrix
- `k_t`, `v_t` are key-value vectors
- `q_t` is the query vector
- `⊗` denotes outer product
- `·` denotes matrix multiplication

### Multi-Track Architecture
ANA maintains multiple parallel LRU tracks with learned mixing:
- Each track can have different retention characteristics
- Learned mixing weights combine track outputs
- Enables diverse temporal processing strategies

## File Structure

```
ana/
├── models.py              # Core ANA implementation
├── config.py              # Configuration classes
├── model_space.py         # Architecture taxonomy
├── model_factory.py       # Modular model builder
└── benchmark.py           # Task definitions

experiments/
├── exp_scaling.py         # Scaling validation
├── exp_long_seq.py        # Long sequence benchmark
├── exp_language.py        # Language modeling
├── exp_extrapolation.py   # Extrapolation test
├── exp_synergy_kv.py      # Synergy across KV counts
├── exp_parameter_efficiency.py  # Param efficiency
├── exp_ultra_efficient.py       # Ultra-small scales
├── exp_noise_robustness.py      # Noise analysis
└── exp_track_ablation.py       # Track count ablation
```

## Configuration

### ANAConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 64 | Model dimension |
| `state_dim` | int | 64 | LRU state dimension |
| `num_layers` | int | 2 | Number of layers |
| `track_count` | int | 2 | Number of parallel tracks |
| `vocab_size` | int | 40 | Vocabulary size |
| `use_hololink` | bool | True | Enable HoloLink |
| `use_controller` | bool | True | Enable Controller |
| `key_dim` | int | 64 | HoloLink key dimension |
| `hololink_decay` | float | 1.0 | Memory decay rate |
| `use_parallel_scan` | bool | True | Use parallel scan for O(n log n) |

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 1e-3 | AdamW learning rate |
| `batch_size` | int | 16 | Training batch size |
| `epochs` | int | 20 | Training epochs |
| `weight_decay` | float | 0.01 | L2 regularization |

## Training Guidelines

### Scale-Specific Learning Rates

| Scale | d_model | Layers | Learning Rate | Epochs |
|-------|---------|--------|---------------|--------|
| Small | 64 | 2 | 1e-3 | 20 |
| Medium | 128 | 3 | 3e-4 | 30 |
| Large | 256 | 4 | 1e-4 | 40 |

**Important**: Use scale-appropriate learning rates for best results.

### Training Steps

```python
from ana.models import ANAModel
from ana.config import ANAConfig
import torch
from torch.utils.data import DataLoader

# 1. Configure model
config = ANAConfig(
    d_model=64,
    num_layers=2,
    state_dim=64,
    vocab_size=30,
    use_hololink=True,
    use_controller=True
)

# 2. Create model
model = ANAModel(config).to(device)

# 3. Setup optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

# 4. Training loop
for epoch in range(epochs):
    for x, y, mask in dataloader:
        optimizer.zero_grad()
        logits, info = model(x)
        loss = weighted_cross_entropy(logits, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
```

## Model Variants

### 1. Baseline SSM
Simple LRU without additional mechanisms.
- Use for comparison to understand baseline performance.

### 2. Controller-only
LRU with dynamic gating only.
- Demonstrates value of adaptive retention.

### 3. HoloLink-only
LRU with holographic memory only.
- Demonstrates value of associative storage.

### 4. Full ANA
LRU with both Controller and HoloLink.
- Shows synergistic effects when combined.

## Task Definition

### Associative Recall Task

Input sequence structure:
```
[KEY] K1 [VAL] V1 ... [KEY] K_n [VAL] V_n ... noise ... [QUERY] K_i -> [predict V_i]
```

- **KEY marker**: 1
- **VAL marker**: 2
- **QUERY marker**: 3
- **Content tokens**: 4+

### Loss Function

Masked cross-entropy where only the final position is weighted:
```python
mask = torch.ones_like(y) * 0.01
mask[-1] = 1.0  # Only care about retrieval target
loss = (CE(logits, y) * mask).sum() / mask.sum()
```

## Results Reference

### Key Findings

1. **Synergy increases with difficulty**: 0% (1 KV) → +19.5% (12 KV)
2. **Ultra-parameter efficient**: 2-3x accuracy at 10-30K params vs Transformer
3. **HoloLink dominates**: Achieves 100% at 2M params alone
4. **Noise robust**: 95-99% across all noise levels

### Result Files

- `archive/experiments/synergy_by_kv.json` - Synergy across KV counts
- `archive/experiments/ultra_efficient.json` - Ultra-small scale results
- `archive/experiments/noise_robustness.json` - Noise robustness
- `archive/PUBLICATION_FINAL.md` - Publication-ready summary

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Small scale (100K) | 89-100% (depending on task) |
| Medium scale (500K) | 100% |
| Large scale (2M) | 100% |
| Ultra-small (10K) | 81-94% |
| Inference (Python) | 3-22ms (slower than Transformer) |
| Inference (CUDA) | TBD (theoretical O(1)) |

## Applications

See `APPLICATIONS.md` for detailed application scenarios.

## Limitations

1. **Inference speed**: Python implementation slower than Transformer
2. **Task specificity**: Optimized for associative recall
3. **Training sensitivity**: Scale-specific hyperparameters needed
4. **Extrapolation**: Position encoding limits length generalization

## Future Work

1. CUDA kernel implementation for O(1) inference
2. Hybrid architectures with local attention
3. Hierarchical HoloLink for multi-scale memory
4. Pre-training on larger corpora

## Citation

```bibtex
@inproceedings{ana2026,
  title={ANA: Adaptive Neural Automaton with Synergistic Memory},
  author={[Your Name]},
  booktitle={NeurIPS},
  year={2026}
}
```

## Contact

For questions about ANA, refer to the experiments in `experiments/` directory and results in `archive/experiments/`.
