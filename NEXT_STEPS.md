# Strategic Research Plan
## Beyond ANA: General-Purpose Innovation

---

## Mission Statement

**Goal:** Produce tangible, beneficial general-purpose innovation in neural network training methodology.

**Not Goal:** Optimize ANA/HoloLink specifically, chase incremental improvements on associative recall.

**Success Metric:** Discoveries that apply to MANY architectures, not just one specific model.

---

## Current State Assessment

### What We Have
- **Verified Finding:** Training order matters for modular architectures (7% → 99%)
- **Working Solution:** Two-phase training protocol
- **Partial Alternative:** EqProp (56% vs 8% joint backprop)
- **Architecture:** ANA with HoloLink + Controller

### What We Need
- **Generalization proof:** Does this apply beyond ANA?
- **Better solution:** EqProp might eliminate need for staging
- **Theoretical grounding:** Why does this happen?
- **Practical impact:** Real-world models, not just synthetic tasks

---

## Strategic Paths Forward

### Path A: EqProp Deep Dive
**Hypothesis:** EqProp's local learning eliminates gradient interference, enabling single-phase modular training.

**Why Important:** If true, this is a general solution applicable to ANY modular architecture.

**Experiments:**
1. Test EqProp on ANA with proper implementation
2. Compare: EqProp vs Two-Phase vs Combined
3. Analyze: Does EqProp provide isolation during training?

**Timeline:** 4-6 hours
**Risk:** EqProp may not fully solve the problem (previous: 56%)
**Reward:** General-purpose training method

### Path B: Cross-Architecture Validation
**Hypothesis:** Training order matters for ALL modular architectures, not just ANA.

**Why Important:** Proves this is a general principle, not an ANA-specific artifact.

**Experiments:**
1. Test on Transformer with Adapter layers
2. Test on simple MoE (2 experts + router)
3. Test on a simple RAG setup

**Timeline:** 8-12 hours
**Risk:** May not apply to other architectures
**Reward:** Universal principle discovered

### Path C: Theoretical Analysis
**Hypothesis:** Gradient interference can be predicted from architecture and loss landscape.

**Why Important:** Enables automatic detection and mitigation without trial-and-error.

**Experiments:**
1. Visualize gradient interactions between components
2. Measure gradient cosine similarity during training
3. Analyze loss landscape curvature by component

**Timeline:** 6-8 hours
**Risk:** May be too theoretical, hard to extract practical insights
**Reward:** Deep understanding, predictive tools

### Path D: Practical Application
**Hypothesis:** Two-phase training improves real-world models.

**Why Important:** Demonstrates immediate practical value.

**Experiments:**
1. Fine-tuning LoRA adapters with two-phase
2. Multi-head attention with staged training
3. Classification head training on frozen backbone

**Timeline:** 4-6 hours
**Risk:** May not show improvement on already well-tuned systems
**Reward:** Immediate practical impact

---

## Recommended Priority Order

```
┌─────────────────────────────────────────────────────────────────────┐
│  PRIORITY 1: EqProp Deep Dive                                        │
│  Rationale: Could provide general solution, highest impact potential │
│  Time: 4-6 hours                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  PRIORITY 2: Cross-Architecture Validation                           │
│  Rationale: Proves generalization, validates broader claim           │
│  Time: 8-12 hours (can run experiments in parallel)                  │
├─────────────────────────────────────────────────────────────────────┤
│  PRIORITY 3: Practical Application                                   │
│  Rationale: Shows immediate value, good for paper/talks              │
│  Time: 4-6 hours                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  PRIORITY 4: Theoretical Analysis                                    │
│  Rationale: Important but can be done in parallel or after           │
│  Time: 6-8 hours                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Plan: EqProp Deep Dive

### Why EqProp?

Equilibrium Propagation offers a fundamentally different learning paradigm:

| Aspect | Backprop | EqProp |
|--------|----------|--------|
| Credit Assignment | Global (through entire graph) | Local (energy differences) |
| Gradient Flow | Through all components | Within each component |
| Interference Risk | High (gradients couple) | Low (local updates) |

### Key Questions

1. **Can EqProp achieve two-phase performance in single phase?**
   - If yes: Major breakthrough, general training method
   - If no: Why not? What's missing?

2. **What's the right EqProp formulation for modular networks?**
   - Separate energy per module?
   - Shared energy with local terms?
   - Hybrid approach?

3. **Does EqProp + Two-Phase combine benefits?**
   - EqProp during each phase?
   - Better than either alone?

### Experimental Protocol

```python
# Experiment 1: EqProp on Full ANA (single phase)
# Hypothesis: Local learning eliminates interference

# Experiment 2: EqProp + Two-Phase
# Hypothesis: Combination achieves best results

# Experiment 3: EqProp on Transformer + Adapter
# Hypothesis: Generalizes beyond ANA
```

### Success Criteria

| Result | Interpretation | Next Step |
|--------|----------------|-----------|
| EqProp > 90% | **Breakthrough** | Generalize to other architectures |
| EqProp 60-90% | Partial success | Optimize, combine with two-phase |
| EqProp < 60% | EqProp insufficient | Focus on two-phase generalization |

---

## Detailed Plan: Cross-Architecture Validation

### Test Architectures

**1. Transformer + Adapter**
```python
# Standard: Train adapter and base together
# Two-Phase: Freeze base, train adapter; then fine-tune together
```

**2. Mixture of Experts (simplified)**
```python
# Standard: Train experts and router together
# Two-Phase: Train experts; then train router
```

**3. Multi-Head Attention**
```python
# Standard: Train all heads together
# Two-Phase: Train heads; then train head-combination layer
```

### Success Criteria

If 2/3 architectures show improvement with two-phase training, we have evidence of a general principle.

---

## Resource Allocation

| Phase | Activity | Time | GPU Hours |
|-------|----------|------|-----------|
| 1 | EqProp implementation & testing | 4-6h | 8-12 |
| 2 | Cross-architecture experiments | 8-12h | 16-24 |
| 3 | Analysis & writeup | 4-6h | 0 |
| **Total** | | **16-24h** | **24-36h** |

---

## Decision Points

### After EqProp Experiments

| Outcome | Decision |
|---------|----------|
| EqProp > 90% | Focus on EqProp generalization, paper on "Local Learning for Modular Networks" |
| EqProp 60-90% | Combine EqProp + Two-Phase, hybrid approach paper |
| EqProp < 60% | Proceed to cross-architecture validation of two-phase alone |

### After Cross-Architecture Validation

| Outcome | Decision |
|---------|----------|
| Works on 2+ architectures | Paper on "Training Order in Modular Networks" |
| Works on 1 architecture | Architecture-specific paper |
| Works on none | Investigate why ANA is special case |

---

## Expected Deliverables

### Minimum (End of Session)
- [ ] EqProp experiment results on ANA
- [ ] Clear go/no-go on EqProp as general solution
- [ ] Updated paper draft with findings

### Target (End of Week)
- [ ] EqProp optimized or ruled out
- [ ] Cross-architecture validation (at least 1 other architecture)
- [ ] Paper ready for submission

### Stretch (Full Project)
- [ ] EqProp generalization to multiple architectures
- [ ] Theoretical analysis of interference
- [ ] Practical guidelines for training modular networks

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| EqProp doesn't work | Have two-phase as fallback, focus on generalization |
| Cross-architecture fails | Investigate what makes ANA special, paper on findings |
| Time overruns | Prioritize EqProp, skip theoretical analysis if needed |

---

## Immediate Next Actions

1. **Implement proper EqProp for ANA** (not the partial version from before)
2. **Run EqProp experiment** with proper hyperparameters
3. **Analyze results** and decide on next direction
4. **Document findings** as we go

---

## Key Insight to Remember

> The goal is not to make ANA better. The goal is to discover principles that make ALL modular architectures better.

If EqProp works, we have a general training method.
If two-phase works across architectures, we have a general principle.
Either way, we win - but only if we generalize beyond ANA.
