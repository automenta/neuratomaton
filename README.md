# Bio-Plausible Adaptive Neural Automaton (Bio-ANA)
## *Bio-Plausibly Trained Neural Networks with Multi-Track SSM and Holographic Memory*

---

## Research Hypothesis

**Can bio-plausibly trained neural networks (using Equilibrium Propagation) achieve competitive performance on language modeling while demonstrating superior efficiency?**

This repository implements Bio-ANA - an architecture that combines:
1. **Equilibrium Propagation (EqProp)**: Bio-plausible local learning rule (no backpropagation)
2. **Multi-Track SSM**: Specialized tracks for different temporal frequencies (syntax, semantics, logic)
3. **HoloLink Memory**: Holographic associative memory with Hebbian updates
4. **Track-Specific Energy Functions**: Each track has its own energy landscape for equilibrium dynamics

---

## Why ANA Matters

### The Recall Problem in Linear Models

State Space Models (Mamba, S4, RWKV) achieve O(1) inference but suffer from the **"Achilles' Heel"**: fixed-size state compression limits recall capacity. When many key-value pairs must be remembered, performance degrades.

| Architecture | Inference | Recall Many Items | Exact Copy | Long Context |
|--------------|-----------|-------------------|------------|--------------|
| Transformer | O(N) | ✓ | ✓ | Limited (window) |
| Mamba/S4 | O(1) | Degrades | Struggles | Degrades beyond training |
| **ANA** | O(1) | **HoloLink** | **Multi-Track** | **Infinite (decay)** |

### ANA's Approach

1. **Multi-Track Decomposition**: Different information types (local syntax vs. global semantics) have different optimal decay rates. A single SSM can't optimize both.

2. **HoloLink as External Memory**: Instead of cramming everything into hidden state, use holographic binding (superposition of key-value pairs) as a separate read/write memory.

3. **Dynamic Modulation**: The HyperController learns WHEN to store vs. retrieve, adapting behavior to input complexity.

---

## Benchmarks & Success Criteria

### Tier 1: Synthetic Tasks (Proof of Concept)

| Task | What It Tests | Mamba | Target | ANA Advantage |
|------|---------------|-------|--------|---------------|
| **Associative Recall (AR)** | Single KV retrieval | 95%+ | 98%+ | HoloLink recall |
| **Multi-Query AR (MQAR)** | Many KV pairs (16-256) | Degrades | Stable | HoloLink capacity |
| **Copy Task** | Exact sequence reproduction | Struggles | 99%+ | Logic track |
| **Reverse Task** | Sequence manipulation | 60% | 85%+ | Working memory |
| **Induction Heads** | In-context pattern | 90%+ | 95%+ | Controller gating |

### Tier 2: Language Modeling (Validation)

| Metric | Model Size | Mamba | Transformer | ANA Target |
|--------|------------|-------|-------------|------------|
| WikiText-103 PPL | 125M | 33.1 | 33.0 | <32.0 |
| Pile PPL (slice) | 125M | ~11.0 | 9.4 | <10.5 |
| Throughput (tok/s) | 125M | 50K | 15K | >40K |

### Tier 3: Downstream (Production)

| Benchmark | 1.4B Model | Target |
|-----------|------------|--------|
| MMLU | 35-40% | >38% |
| HellaSwag | 50-55% | >52% |
| PIQA | 75-78% | >76% |

---

## Project Status (Updated 2026-02-10)

| Phase | Status | Result |
|-------|--------|--------|
| **Phase 1: EqProp** | ✅ Complete | 99% XOR accuracy, energy convergence validated |
| **Phase 2: Bio-ANA** | ✅ Complete | 17/17 tests pass, architecture integrated |
| **Phase 3: Training** | ✅ Complete | 5.31x speedup, 100% AR accuracy in 25 steps |
| **Phase 3.5: WikiText** | 🚀 Ready | Infrastructure complete, awaiting execution |
| **Phase 4: Full Eval** | 📅 Planned | 41 GPU hours |
| **Phase 5: Publication** | 📅 Planned | Week 4 |

**Performance Improvements**:
- Forward pass: 608ms → 115ms (**5.31x faster**)
- Memory: 74MB → 68MB (-11%)
- Tokens/sec: 139 → 616 (4.43x improvement)

---

## Quick Start

**To begin Phase 3.5 WikiText validation:**
```bash
python run_wikitext_validation.py --variant small --vocab-size 10000 --epochs 5
```

**Expected**: 6 hours → PPL 28-35 → Decision to proceed to full evaluation

**For detailed information, see:**
- [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) - Complete experimental plan
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick start guide
- [PLAN.md](PLAN.md) - Project status and milestones

---

## Architecture

```
Input Token
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    ANA Layer                            │
│  ┌─────────────┐                                       │
│  │ Controller  │──► alpha_A, beta_A, alpha_B, beta_B   │
│  │ (tiny MLP)  │──► mix_A, mix_B, ret_gate, halt       │
│  └─────────────┘                                       │
│         │                                              │
│    ┌────┴────┐                                         │
│    ▼         ▼                                         │
│ ┌──────┐  ┌──────┐    Parallel LRU Tracks              │
│ │Track │  │Track │    h_t = α·h_{t-1} + β·x_t          │
│ │  A   │  │  B   │                                     │
│ │(Fast)│  │(Slow)│                                     │
│ └──┬───┘  └───┬──┘                                     │
│    │          │                                        │
│    └────┬─────┘                                        │
│         ▼                                              │
│  ┌─────────────┐                                       │
│  │  HoloLink   │    M_t = M_{t-1} + K(h)·V(h)^T       │
│  │ (Linear Attn)│   retrieve = M_t · Q(x)             │
│  └──────┬──────┘                                       │
│         │                                              │
│    Mix + Residual                                      │
└─────────┼───────────────────────────────────────────────┘
          ▼
      Next Layer
```

---

## Research Roadmap

### Phase 1: Baseline Validation (Current)
- [x] Core architecture implementation
- [x] Training pipeline with curriculum
- [x] Synthetic task evaluation (AR, Copy, Reverse)
- [ ] **Next**: Scale to 125M params, compare with BaselineSSM

### Phase 2: HoloLink Optimization
- [ ] Ablation: HoloLink vs. no HoloLink on MQAR
- [ ] Key dimension scaling study (32 → 256)
- [ ] Orthogonal vs. learned projections
- [ ] Decay scheduling experiments

### Phase 3: Controller Refinement
- [ ] Controller depth study (1-4 layers)
- [ ] ACT (Adaptive Computation Time) integration
- [ ] Curriculum strategies for controller
- [ ] Regularization (KL divergence to prior)

### Phase 4: Scaling & Efficiency
- [ ] 70M → 360M → 1.4B parameter sweep
- [ ] Custom CUDA kernels for parallel scan
- [ ] Memory profiling and optimization
- [ ] Mixed precision training (FP16/BF16)

### Phase 5: Real-World Validation
- [ ] Pretrain on Pile (10B tokens minimum)
- [ ] WikiText-103 benchmark
- [ ] Long-context evaluation (4K→32K tokens)
- [ ] Zero-shot downstream (MMLU, HellaSwag)

---

## Key Research Questions

1. **HoloLink Capacity**: What is the effective memory capacity of holographic binding? How many KV pairs can be stored before interference degrades recall?

2. **Track Specialization**: Do tracks naturally specialize (syntax vs. semantics) or must they be forced via auxiliary losses?

3. **Controller Necessity**: Is dynamic gating essential, or can static per-track parameters achieve similar performance?

4. **Extrapolation**: How does ANA perform on sequences 2×, 4×, 8× longer than training?

5. **Efficiency Trade-offs**: Where is the sweet spot between state dimension, key dimension, and compute?

---

## Ablation Studies

Run systematic ablations:

```bash
# No HoloLink
python run_experiment.py train 3a --no-hololink

# No Controller (static tracks only)
python run_experiment.py train 3a --no-controller

# Single track
python run_experiment.py train 3a --tracks 1

# Four tracks
python run_experiment.py train 3a --tracks 4

# Thinking steps
python run_experiment.py train 3a --thinking-steps 4
```

---

## Comparison with State of the Art

| Architecture | Params | WikiText PPL | AR Accuracy | MQAR (64 pairs) | Inference |
|--------------|--------|--------------|-------------|-----------------|-----------|
| Transformer | 125M | 33.0 | 99% | 99% | O(N) KV cache |
| Mamba | 130M | 33.1 | 95% | 72% | O(1) state |
| RWKV-4 | 125M | 34.0 | 93% | 68% | O(1) state |
| **ANA (target)** | 125M | <32.0 | 98% | 90% | O(1) state |

*Note: ANA results pending proper scale-up experiments*

---

## Contributing

Key areas for contribution:
1. **CUDA kernels**: Efficient parallel scan implementation
2. **Evaluation**: Additional downstream benchmarks
3. **Architecture**: Novel track types, controller designs
4. **Theory**: Capacity analysis of HoloLink memory

---

## Citation

```bibtex
@software{ana2024,
  title = {Adaptive Neural Automaton: Multi-Track SSM with Holographic Memory},
  author = {ANA Research Team},
  year = {2024},
  url = {https://github.com/automenta/neuratomaton}
}
```

---

## License

MIT License - See LICENSE file for details.
