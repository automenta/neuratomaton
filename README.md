# ANA: Adaptive Neural Automaton

**Multi-track State Space Model with Holographic Binding**

---

## Architecture

```
Input ──► [Embedding] ──► Multi-Track SSM ──► HoloLink ──► Output
                               │                  │
                          Fast Track         Associative
                          Slow Track          Memory
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Multi-track SSM** | Parallel recurrence at different time-scales (fast/slow) |
| **HoloLink** | Key-value associative memory via matrix accumulation |
| **HyperController** | Dynamic gating of decay (α) and intake (β) per track |
| **Parallel Scan** | O(N) training via cumsum trick |

---

## Quick Start

```python
from ana import ANAConfig, ANAModel, BaselineSSM

# Create model
config = ANAConfig(d_model=64, vocab_size=100, state_dim=64, track_count=2)
model = ANAModel(config)

# Forward pass
import torch
input_ids = torch.randint(0, 100, (1, 32))
logits, info = model(input_ids)

# Compare with baseline
baseline = BaselineSSM(config)
logits, _ = baseline(input_ids)
```

---

## Experiments

Run the research experiment suite:

```bash
python -m ana.experiments
```

This runs:
1. **Curriculum learning** - Train on lengths 2-6, test generalization to 7-12
2. **HoloLink ablation** - With vs without associative memory
3. **Track ablation** - 1 vs 2 vs 3 tracks
4. **Baseline comparison** - ANA vs single-track SSM

---

## Research Question

**Can multi-track SSMs learn algorithms from examples?**

Current results suggest position-specific learning dominates. Curriculum learning may help.

| Task | Training | k=1.5 | k=2.0 |
|------|----------|-------|-------|
| Copy | 100% | - | - |
| Reverse | ~50% | ~30% | ~20% |

---

## Code Structure

```
ana/
├── config.py       # ANAConfig dataclass
├── models.py       # ANAModel, LinearRecurrentUnit, HoloLink, BaselineSSM
├── experiments.py  # Research experiments
├── train.py        # Simple training loop
├── eval.py         # Evaluation functions
└── tasks.py        # CopyTask, ReverseTask datasets

tests/
├── test_models.py  # Model unit tests
└── test_parallel.py # Parallel scan correctness tests
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Citation

```bibtex
@misc{ana2026,
  title={ANA: Adaptive Neural Automaton with Multi-track State Space Models},
  author={...},
  year={2026}
}
```

---

## License

MIT
