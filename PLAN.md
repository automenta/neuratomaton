# ANA Research Plan

## Status (February 2026)

**Architecture**: Multi-track State Space Model with Holographic Binding
**Question**: Can SSMs learn algorithms from examples?

---

## Architecture Summary

```
Input ──► [Embedding] ──► x_t
                              │
       ┌──────────────────────┴──────────────────────┐
       │              MULTI-TRACK SSM                │
       │                                             │
       │  Track A (Fast):  h = α·h + β·x    (local) │
       │  Track B (Slow):  h = α·h + β·x    (long)  │
       │                                             │
       │  α ∈ [0,1] controls memory decay           │
       │  β ∈ [0,1] controls input intake           │
       └──────────────────────┬──────────────────────┘
                              │
       ┌──────────────────────▼──────────────────────┐
       │            HOLOGRAPHIC LINK                 │
       │                                             │
       │  Key-Value associative memory:             │
       │    M = Σ k_t ⊗ v_t                         │
       │    retrieve: v ≈ q ⊗ M                     │
       │                                             │
       └──────────────────────┬──────────────────────┘
                              │
                              ▼
                         [Output]
```

---

## Key Components

| Component | Innovation | Role |
|-----------|------------|------|
| **Multi-track SSM** | Different time-scales | Fast track for local patterns, slow for long-range |
| **Dynamic Gates** | Input-dependent α, β | Adapt memory decay per token |
| **HoloLink** | Matrix accumulation | Associative recall via key-value binding |
| **Parallel Scan** | O(N) cumsum | Efficient training on GPUs |

---

## Research Questions

### Primary
**Can multi-track SSMs with holographic binding learn algorithmic patterns?**

### Secondary
1. Does curriculum learning force algorithmic generalization?
2. Is multi-track better than single-track?
3. Does HoloLink contribute to performance?
4. What is the capacity limit for algorithmic tasks?

---

## Experiments

### E1: Curriculum Learning
Train on lengths 2-6, test on 7-12. Test if multi-length training forces algorithmic learning.

```bash
python -m ana.experiments
```

### E2: HoloLink Ablation
Compare with/without associative memory.

### E3: Track Ablation
Compare 1-track vs 2-track vs 3-track.

### E4: Baseline Comparison
Compare ANA vs single-track SSM baseline.

---

## Success Criteria

| Metric | Target | Publication Path |
|--------|--------|------------------|
| k=2.0 generalization >50% | **Breakthrough** | NeurIPS/ICLR main |
| k=2.0 generalization >30% | Progress | Workshop paper |
| k=2.0 generalization <30% | Limitation | Position paper |

---

## What's Novel

| Component | Novelty | Publishable |
|-----------|---------|-------------|
| Multi-track SSM with specialized time-scales | ⭐⭐⭐ High | Yes |
| Holographic binding for associative recall | ⭐⭐ Medium | If validated |
| O(N) parallel scan implementation | ⭐ Low | Standard |

## What's NOT Novel

- Opcodes/program stacks (failed experiments)
- Thinking mode (unvalidated complexity)
- Hardcoded algorithms (cheating)

---

## Code Organization

```
ana/
├── config.py       # ANAConfig
├── models.py       # ANAModel, LRU, HoloLink, BaselineSSM
├── experiments.py  # Curriculum training, ablations
├── train.py        # Simple training loop
├── eval.py         # Evaluation
└── tasks.py        # Copy/Reverse datasets

tests/
├── test_models.py  # Unit tests (10 tests)
└── test_parallel.py # Parallel scan tests
```

---

## Next Action

Run the full experiment suite:

```bash
python -m ana.experiments
```

Expected output:
- Curriculum learning results
- HoloLink ablation comparison
- Track count ablation
- Baseline comparison

---

## Potential Breakthroughs

1. **If curriculum works**: "Multi-track SSMs learn algorithms via curriculum"
2. **If multi-track helps**: "Time-scale specialization enables algorithmic reasoning"
3. **If HoloLink helps**: "Holographic binding improves associative recall in SSMs"
4. **If all fail**: "The algorithmic generalization challenge in state space models"

All outcomes are publishable.
