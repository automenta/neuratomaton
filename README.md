# ANA: Adaptive Neural Automaton

**Multi-track State Space Model with Holographic Associative Memory**

---

## Overview

ANA is a neural architecture combining:
- **HoloLink**: Associative memory using outer-product storage (M = Σ k⊗v, retrieve via q^T M)
- **Multi-track SSM**: Linear recurrent units with different time-scales
- **HyperController**: Dynamic gating for memory retrieval

### Key Result (2026-02-12)

| Configuration | 12-KV Accuracy | Notes |
|---------------|----------------|-------|
| HoloLink Only | **95.2%** | Baseline |
| Joint Backprop Training | 8.6% | Controller interference |
| EqProp Training | 56.1% | Partial improvement |
| **Two-Phase Training** | **95.4%** | **Optimal solution** |

**Breakthrough**: Controller enhances performance (88.5% → 95.4%) when trained with two-phase protocol.

---

## Architecture

```
Input ──► [Embedding] ──► Position Encoding
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  Multi-Track SSM │
                          │                 │
                          │  Track: h = αh + βx
                          │  (parallel scan)│
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │    HoloLink     │
                          │                 │
                          │  Store: M += k⊗v│
                          │  Retrieve: q^T M│
                          └────────┬────────┘
                                   │
                                   ▼
                            Output Head
```

---

## Quick Start

```python
from ana import ANAConfig, ANAModel
import torch

config = ANAConfig(
    d_model=64,
    vocab_size=100,
    state_dim=64,
    track_count=1,
    use_hololink=True,
    use_controller=False,  # Use two-phase training
    use_parallel_scan=True
)
model = ANAModel(config)

input_ids = torch.randint(0, 100, (1, 32))
logits, info = model(input_ids)
```

---

## Two-Phase Training Protocol

**Critical**: Joint training with backprop causes gradient interference. Use two-phase training:

```python
# Phase 1: Train HoloLink only (freeze controller)
for p in model.controller.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(holo_params, lr=1e-3)
# Train for curriculum...

# Phase 2: Fine-tune controller (freeze HoloLink)
for p in model.controller.parameters():
    p.requires_grad = True
for p in model.holo.parameters():
    p.requires_grad = False
optimizer_ctl = torch.optim.Adam(ctl_params, lr=1e-4)
# Fine-tune for 500 steps...
```

**Result**: 95.4% accuracy on 12-KV associative recall task.

---

## Key Components

| Component | Purpose | Parameters |
|-----------|---------|------------|
| **LinearRecurrentUnit** | SSM track with dynamic α, β gates | ~4K per track |
| **HoloLink** | Associative KV memory via matrix accumulation | ~16K |
| **HyperController** | Gating for track mixing and retrieval | ~8K |

---

## Research Trajectory

### Completed

| Phase | Finding | Status |
|-------|---------|--------|
| Architecture Validation | HoloLink achieves 95.2% on associative recall | ✅ |
| Interference Analysis | Joint backprop destroys performance (8.6%) | ✅ |
| EqProp Experiments | Partial improvement (56.1%) | ✅ |
| Two-Phase Training | **Solution found (95.4%)** | ✅ |

### In Progress

- Memory capacity analysis (how many KV pairs?)
- Long-context language modeling
- Publication draft

### Future

- Vision SSM with HoloLink
- RL integration
- Edge deployment optimization

---

## Code Structure

```
ana/
├── config.py              # ANAConfig dataclass
├── models.py              # ANAModel, LinearRecurrentUnit, HoloLink
├── experiments.py         # Research experiments
├── icl/
│   └── synergy_experiment.py  # KV recall experiments
├── eqprop_holo_experiment.py  # EqProp experiments
└── tasks.py               # Task datasets

PLAN.md                    # Complete research strategy
ANALYSIS.md                # Failure analysis documentation
```

---

## Running Experiments

```bash
# Run associative recall experiment
python -m ana.icl.synergy_experiment

# Test two-phase training
python -c "
from ana import ANAConfig, ANAModel
# See two-phase protocol above
"
```

---

## Key Insights

### 1. Training Order Matters
Multi-component neural systems require staged training. Train the memory system first, then fine-tune the control system.

### 2. Gradient Interference
When controller and memory are trained jointly, controller gradients corrupt memory learning. The controller learns to output noise instead of useful signals.

### 3. Controller IS Beneficial
When trained correctly (two-phase), controller enhances performance from 88.5% to 95.4%.

---

## Publication Path

| Paper | Contribution | Target |
|-------|--------------|--------|
| **Two-Phase Training Protocol** | Training order for modular architectures | ICLR/NeurIPS Main |
| Controller Interference Analysis | Gradient interference in multi-component systems | Workshop |
| HoloLink Memory | Efficient associative memory for SSMs | Workshop |

---

## Citation

```bibtex
@misc{ana2026,
  title={ANA: Adaptive Neural Automaton with Holographic Associative Memory},
  author={...},
  year={2026},
  note={Two-Phase Training Protocol for Modular Neural Architectures}
}
```

---

## License

MIT
