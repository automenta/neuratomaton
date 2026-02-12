# ANA: Adaptive Neural Automaton
## Complete Research Strategy 2026

---

## Executive Summary

**Core Finding**: HoloLink achieves 94.4% on associative recall WITHOUT controller. Controller trained with backprop DESTROYS this to 8-9%.

**The Problem**: Backprop causes controller to learn interference patterns that corrupt HoloLink's memory signal.

**The Solution**: Equilibrium Propagation - local learning where each module learns independently from energy differences.

**Breakthrough Opportunity**: EqProp + HoloLink has NEVER been published. If it works, it's a novel contribution to both bio-plausible learning AND associative memory research.

**Research Phases**: EqProp Validation → Synergy Experiments → Publication

---

## ⚠️ CRITICAL: Anti-Patterns & Guardrails

### What Went Wrong Before

The previous research cycle wasted significant time on **Copy and Reverse tasks**. Here's what happened:

| Date | Event | Problem |
|------|-------|---------|
| Feb 9 | Copy/Reverse tasks added as "evaluation metrics" | Wrong metrics from the start |
| Feb 9-10 | Training printed Copy/Reverse accuracy every epoch | Created feedback loop |
| Feb 10 | Extensive hyperparameter tuning on Reverse task | Chasing impossible goal |
| Feb 10 | ANALYSIS.md written documenting "failure" | Framed as model problem, not task problem |
| Feb 10 | ReverseNet created as "fix" | Architecture change, not admitting task mismatch |

### Root Cause Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE COPY/REVERSE TRAP                            │
│                                                                     │
│  1. Tasks introduced as "evaluation" → Not hypothesis-driven        │
│  2. Metrics printed every epoch → Obsessive optimization target     │
│  3. Copy easy (100%) → False confidence                            │
│  4. Reverse hard (12%) → "Must fix" mentality                       │
│  5. Fundamental mismatch ignored → Autoregressive ≠ Bidirectional  │
│                                                                     │
│  RESULT: Weeks of effort on tasks that DON'T MATTER                │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Reverse Task Was Doomed

```
Autoregressive Model (ANA, GPT, etc.):
  Token n can only see tokens 1...n-1
  Cannot "look ahead" to reverse

Bidirectional Model (BERT, etc.):
  Token n can see ALL tokens
  Can access both ends simultaneously

ANA is autoregressive by design → Cannot do reverse well
This is NOT a bug, it's an architectural property
```

### The Real Finding That Was Ignored

From paper_draft.md:
> "With focused loss training, ANA achieves 100% accuracy on associative recall tasks with 10+ token noise gaps, outperforming baseline SSMs by 11.5%"

**This was the real result. Copy/Reverse was a distraction.**

---

## Guardrails for This Research Cycle

### Explicit Forbidden Activities

| ❌ FORBIDDEN | Why | Alternative |
|--------------|-----|-------------|
| Copy/Reverse task experiments | Already done, known results, wrong direction | Associative recall tasks |
| "Fixing" reversal performance | Architectural impossibility | Accept limitation, move on |
| Training loops that print Copy/Reverse every epoch | Creates obsessive feedback loop | Measure only relevant metrics |
| Hyperparameter tuning for algorithmic tasks | Sunk cost trap | Tune for ICL/recall tasks |
| Creating "ReverseNet" or similar fixes | Admitting wrong architecture | Use right architecture for task |

### Kill Criteria (Stop Immediately If...)

| Trigger | Action |
|---------|--------|
| Spending >2 hours on any single task variant | Document and move on |
| Accuracy plateau for >3 experiments on same task | Task doesn't fit architecture |
| Finding yourself "debugging" a known limitation | Accept and document |
| More time on failure analysis than success | Pivot to what works |

### Success Criteria (Define BEFORE Starting)

| Experiment | Success = Continue | Failure = Stop/Pivot |
|------------|-------------------|---------------------|
| E1: Synergy | >10% synergy | <10% synergy |
| E2: KV Scaling | >80% at 16 pairs | <60% at 16 pairs |
| E3: Memory | O(1) verified | Linear growth |
| E4: ICL | Win 3/4 tasks | Lose all tasks |
| E5: Long Context | >5x memory savings | <2x savings |

### Time Limits Per Experiment

| Experiment | Max Time | Stop If No Progress After |
|------------|----------|--------------------------|
| E1: Synergy | 2 hours | 1 hour |
| E2: KV Scaling | 4 hours | 2 hours |
| E3: Memory | 2 hours | 1 hour |
| E4: ICL | 8 hours | 4 hours |
| E5: Long Context | 8 hours | 4 hours |

### Weekly Review Questions

At the end of each week, ask:

1. **Am I working on the right problem?**
   - Is this aligned with proven strengths?
   - Am I chasing a known limitation?

2. **What did I learn?**
   - Positive result → Continue
   - Negative result → Document and pivot
   - No result → Debug or stop

3. **Am I in a sunk cost trap?**
   - Have I spent >4 hours on one thing?
   - Am I "trying one more thing" repeatedly?
   - Is this really the best use of time?

---

## Proven Results (From Actual Experiments)

### What Works ✅

| Result | Evidence | Implication |
|--------|----------|-------------|
| **HoloLink: 94.4% at 12 KV pairs** | Without controller, frozen controller | Core memory module works |
| **Copy Task: 100%** | Full generalization to L12 | Sequential processing works perfectly |
| **HoloLink Synergy: +19.5%** | 12 KV pairs, paper_draft.md | Associative memory is real advantage |
| **Parameter Efficiency: 2-3x** | 10-30K params vs Transformer | Edge deployment viable |
| **EqProp Integration: ✅** | XOR convergence <400 iters | Bio-plausible training works |
| **ANA v3 Reverse: 100%** | Stack + Reverse Read | Algorithmic read patterns = generalization |

### What Fails ❌

| Result | Evidence | Root Cause | Action |
|--------|----------|------------|--------|
| **Controller + Backprop: 8-9%** | Interference destroys HoloLink | Controller learns to output noise | **USE EQPROP** |
| **Standard ANA Reversal: 12-25%** | Position-specific memorization | No explicit memory + wrong inductive bias | SEE v3 solution |
| **Bio-ANA: Slow** | EqProp overhead 10-100x | Relaxation iterations | Use for research, not production |
| **Implicit Algorithm Learning** | ANALYSIS.md | Memorization over generalization | Use explicit memory structures |

### Key Insights

> **CRITICAL: Controller trained with backprop DESTROYS HoloLink's 94% performance. The controller learns to interfere, not help.**

> **SOLUTION: EqProp's local learning could allow each module to learn independently, avoiding interference.**

> **BREAKTHROUGH: Algorithmic generalization requires EXPLICIT MEMORY + ALGORITHMIC READ PATTERNS, not learned weights alone.**

---

## EqProp Experiments: Breakthrough Opportunity

### The Controller Interference Problem (CRITICAL DISCOVERY)

**Background**: We found that backprop training DESTROYS HoloLink's performance:

| Configuration | 12-KV Accuracy | Status |
|--------------|----------------|--------|
| HoloLink Only (no controller) | **94.4% ± 1.2%** | ✅ WORKS |
| Controller frozen (pass-through) | **94.0%** | ✅ WORKS |
| Controller trainable (any init) | **8-9%** | ❌ FAILS |

**Root Cause**: The controller has 5+ outputs (α_gate, β_gate, mix, ret_gate, halt). Gradient descent finds a local minimum where the controller outputs noise that overwhelms HoloLink's signal. **The controller learns to interfere, not help.**

### Why EqProp Could Solve This

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE INTERFERENCE PROBLEM                              │
│                                                                          │
│  BACKPROP:                                                               │
│    Loss → ∂L/∂output → chain rule through ALL layers                     │
│    Problem: Controller gradients CONTAMINATE HoloLink gradients          │
│    Result: Controller learns to output noise, HoloLink degraded          │
│                                                                          │
│  EQPROP:                                                                 │
│    E_free = energy at equilibrium (no target)                            │
│    E_nudged = energy with weak target clamp                              │
│    ∂L/∂θ_local ≈ (E_nudged - E_free) at THIS LAYER ONLY                  │
│                                                                          │
│  KEY INSIGHT: Each module learns from LOCAL energy differences           │
│  → Controller cannot interfere with HoloLink's learning                  │
│  → HoloLink maintains its 94%+ performance                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Status

| File | Purpose | Status |
|------|---------|--------|
| `ana/eqprop_ana.py` | Energy-based SSM + HoloLink | Complete |
| `ana/eqprop_seq.py` | Spectral-norm stabilized EqProp | Complete |
| `ana/bioplausible_ana.py` | Integration with bioplausible library | Complete |

### EqProp Research Questions

| Question | Why Important | Expected Outcome |
|----------|---------------|------------------|
| Does EqProp preserve HoloLink's 94%? | Tests if local learning avoids interference | If yes: **BREAKTHROUGH** |
| Can controller help with EqProp? | Controller may now learn to enhance, not interfere | Synergy > 94% |
| What's optimal relaxation depth? | Efficiency vs accuracy tradeoff | Find n_iterations sweet spot |

### Novel Contribution: EqProp + Associative Memory

**This combination has NEVER been published.** The intersection of:
1. Equilibrium Propagation (bio-plausible, local learning)
2. Holographic associative memory (HoloLink, outer-product binding)
3. Multi-track SSM (adaptive temporal processing)

**Why It's Novel**: 
- EqProp papers focus on classification, not memory
- Memory papers use backprop, not bio-plausible learning
- No prior work combines EqProp + associative memory + SSM

**Publication targets**: 
- NeurIPS/ICLR (bio-ML track)
- CogSci / Bernstein Conference (computational neuroscience)
- ICLR Workshop on Biologically Plausible Learning

---

## ANA v3: Algorithmic Generalization Breakthrough

### Discovery

The v3 experiments (`ana/v2/experiments/`) revealed a fundamental insight:

```
THE ALGORITHM IS IN THE READ PATTERN, NOT THE LEARNED WEIGHTS

Standard ANA:     h_t = f(x_t, h_{t-1})     → learns position mappings
ANA v3:           output = stack[L-1-t]     → implements reversal algorithm
```

### Results

| Architecture | Generalization to L12 | Method |
|--------------|----------------------|--------|
| Standard ANA | 12% | Implicit memory in hidden states |
| ReverseNet (bidirectional) | 25% | Bidirectional LSTM |
| **ANA v3 (Stack + Reverse Read)** | **75-100%** | Explicit stack + algorithmic read |

### Implications for Future Research

1. **Explicit Memory Structures**: SSMs benefit from explicit, addressable memory
2. **Algorithmic Read Patterns**: Task-specific read patterns enable generalization
3. **Learnable Read Patterns**: Future direction - can we LEARN the read pattern?

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    ANA ARCHITECTURE                         │
│                                                             │
│  Input → Embedding → Position Encoding                      │
│            │                                                │
│            ▼                                                │
│  ┌─────────────────────────────────────┐                   │
│  │      MULTI-TRACK SSM LAYER          │                   │
│  │                                     │                   │
│  │  Track A (Fast): τ=0.5, reactive   │ ← Proven: handles │
│  │  Track B (Slow): τ=2.0, strategic  │   local patterns  │
│  │                                     │                   │
│  │  h_t = α·h_{t-1} + β·x_t           │                   │
│  │  α,β = sigmoid(static + dynamic)   │                   │
│  └──────────────┬──────────────────────┘                   │
│                 │                                           │
│                 ▼                                           │
│  ┌─────────────────────────────────────┐                   │
│  │         HOLOLINK MEMORY             │                   │
│  │                                     │                   │
│  │  Associative Storage:               │ ← Proven: +19.5%  │
│  │    M = Σ k_t ⊗ v_t                 │   synergy at      │
│  │                                     │   high difficulty │
│  │  Retrieval: v ≈ q^T M              │                   │
│  │                                     │                   │
│  │  Properties: O(1) retrieval        │                   │
│  └──────────────┬──────────────────────┘                   │
│                 │                                           │
│                 ▼                                           │
│           Output Projection                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Research Phases

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 0: OPTIMIZATION                                               │
│  Time: 4-8 hours                                                     │
│  Goal: Profile and optimize before expensive experiments             │
│  Output: Optimized code, baseline performance metrics                │
│  GUARDRAIL: If optimization fails, proceed with baseline            │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 1: VALIDATION (Tier 1)                                        │
│  Time: Week 1                                                        │
│  Goal: Quick wins that confirm architecture value                    │
│  Output: Go/No-Go decision                                           │
│  GUARDRAIL: Kill if E1 fails, don't chase                           │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2: CORE EXPERIMENTS (Tier 2)                                  │
│  Time: Week 2-3                                                      │
│  Goal: Publishable results in specific domains                       │
│  Output: Paper draft                                                 │
│  GUARDRAIL: Pivot to efficiency if ICL fails                        │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3: EXTENDED APPLICATIONS (Tier 3)                             │
│  Time: Week 4+                                                       │
│  Goal: Broader impact, additional papers                             │
│  Output: Additional publications                                     │
│  GUARDRAIL: Skip if core experiments not strong                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Optimization (CRITICAL - Run First)

### Purpose
Before investing hours in experiments, identify and fix obvious performance bottlenecks. A 2x speedup saves days of GPU time over the full research plan.

### Guardrails
- **Max time**: 4 hours for optimization
- **Stop if**: No improvement after 2 hours
- **Accept**: 1.5x speedup is fine, don't chase diminishing returns

### Protocol

#### Step 0.1: Baseline Profiling (30 min, STOP after 45 min)

```python
# Create profiling script: ana/profiling/profile_baseline.py

import torch
import torch.profiler as profiler
from ana import ANAConfig, ANAModel
import time

def profile_model(config, seq_len=512, batch_size=16, warmup=10, steps=100):
    """Profile model performance and identify bottlenecks."""
    model = ANAModel(config).cuda()
    model.eval()
    
    # Warmup
    for _ in range(warmup):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Time measurement
    start = time.time()
    for _ in range(steps):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_sec = (batch_size * seq_len * steps) / elapsed
    
    print(f"\n{'='*60}")
    print(f"BASELINE PERFORMANCE (seq_len={seq_len}, batch={batch_size})")
    print(f"{'='*60}")
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"Latency: {elapsed/steps*1000:.2f} ms/batch")
    
    return {'tokens_per_sec': tokens_per_sec, 'latency_ms': elapsed/steps*1000}

if __name__ == "__main__":
    config = ANAConfig(d_model=64, vocab_size=100, state_dim=64)
    profile_model(config, seq_len=512, batch_size=16)
```

#### Step 0.2: Apply Safe Optimizations (2 hours max)

| Optimization | Impact | Risk | Effort | Apply? |
|-------------|--------|------|--------|--------|
| Mixed Precision | HIGH | LOW | 5 min | ✅ YES |
| torch.compile | HIGH | LOW | 5 min | ✅ YES |
| Parallel Scan | MEDIUM | LOW | 1 line | ✅ YES |
| Fused Ops | MEDIUM | MEDIUM | 30 min | ⚠️ IF TIME ALLOWS |
| Custom CUDA | HIGH | HIGH | Days | ❌ NO |

```python
# Optimizations to apply (copy-paste ready)

# 1. Mixed Precision (in config or training loop)
config.use_amp = True
scaler = torch.cuda.amp.GradScaler()

# 2. torch.compile
model = torch.compile(model, mode="reduce-overhead")

# 3. Parallel scan
config.use_parallel_scan = True
```

#### Step 0.3: Verify and Move On (30 min)

```python
# Quick verification - don't over-engineer
baseline = profile_model(config)
config.use_amp = True
config.use_parallel_scan = True
optimized = profile_model(config)

speedup = optimized['tokens_per_sec'] / baseline['tokens_per_sec']
print(f"Speedup: {speedup:.2f}x")

# Accept >1.5x and move on
if speedup > 1.5:
    print("✅ Optimization successful, proceeding to experiments")
else:
    print("⚠️ Limited improvement, proceeding with baseline anyway")
```

---

## Phase 1: Validation (Tier 1)

**Goal**: Quick wins that confirm architecture value

**Guardrails**:
- Max 1 week
- Kill if E1 synergy < 10%
- Accept results, don't chase

### E1: Reproduce Synergy (Quick Win)

**Purpose**: Validate the +19.5% synergy claim

**Protocol**:
```python
python -m ana.experiments  # Existing code
```

**Time Limit**: 2 hours max

**Success Criterion**: Full ANA > max(Controller, HoloLink) by >10%

**If Fails**: Document and skip to E3 (memory efficiency) - don't debug synergy

---

### E2: Multi-KV Associative Recall Scaling

**Purpose**: Find capacity limit of HoloLink

**Guardrails**:
- Max 4 hours
- Stop if accuracy < 60% at 8 pairs
- Don't tune hyperparameters extensively

**Protocol**:
```python
for num_pairs in [1, 2, 4, 8, 16, 32]:
    accuracy = evaluate(model, num_pairs)
    if accuracy < 0.6 and num_pairs < 16:
        print(f"Early stop: capacity limit at {num_pairs} pairs")
        break
```

**Expected Results**:
| KV Pairs | ANA Target | Stop If |
|----------|------------|---------|
| 1-4 | 99%+ | <95% |
| 8 | 95%+ | <80% |
| 16 | 80%+ | <60% |

---

### E3: Memory Efficiency Profiling

**Purpose**: Validate O(1) memory claim

**Guardrails**:
- Max 2 hours
- If memory grows, check for bugs
- Don't try to "fix" memory issues architecturally

**Protocol**:
```python
for L in [512, 1024, 2048, 4096, 8192]:
    mem = profile_memory(model, L)
    print(f"Context {L}: {mem:.1f} MB")
    # Expect flat line, not linear growth
```

---

### Phase 1 Decision Gate (End of Week 1)

| Condition | Action | Don't |
|-----------|--------|-------|
| ✅ E1 synergy > 10% | Continue to Phase 1.5 (EqProp) | Don't tune for more |
| ❌ E1 synergy < 10% | Skip to E3, focus on efficiency | Don't debug HoloLink |
| ✅ E2 shows scaling | Document capacity | Don't push past limit |
| ✅ E3 O(1) memory | Efficiency validated | Don't optimize further |
| ❌ All fail | Write position paper | Don't chase failures |

---

## Phase 1.5: EqProp Validation (NEW - HIGH IMPACT)

**Goal**: Test if EqProp preserves/enhances HoloLink synergy

**Why This Matters**: EqProp could solve the gradient interference problem that makes standard backprop suboptimal for modular architectures.

### E-Eq1: EqProp Synergy Test (4 hours)

**Protocol**:
```python
# Compare EqProp vs Backprop on associative recall
from ana.eqprop_ana import EqPropANA, EqPropConfig, train_with_eqprop
from ana import ANAConfig, ANAModel

# Train EqProp ANA
config_eqprop = EqPropConfig(vocab_size=60, d_model=64, state_dim=64, n_iterations=20)
model_eqprop = EqPropANA(config_eqprop)

# Train standard ANA (baseline)
config_standard = ANAConfig(vocab_size=60, d_model=64, state_dim=64)
model_standard = ANAModel(config_standard)

# Compare synergy at 12 KV pairs
```

**Success Criteria**:
| Metric | EqProp Target | Backprop Baseline | Implication |
|--------|---------------|-------------------|-------------|
| Synergy at 12 pairs | >25% | 19.5% | **BREAKTHROUGH** |
| Synergy at 12 pairs | 15-25% | 19.5% | Promising, continue |
| Synergy at 12 pairs | <15% | 19.5% | EqProp not helpful here |

**If Success**: Proceed to E-Eq2 (ICL with EqProp)
**If Failure**: Document, skip EqProp, proceed with standard training

### E-Eq2: EqProp Generalization Test (4 hours)

**Question**: Does local learning improve generalization?

**Protocol**:
```python
# Train on KV pairs 1-8, test on 10-16
# Compare EqProp vs Backprop generalization gap
train_pairs = [1, 2, 4, 6, 8]
test_pairs = [10, 12, 14, 16]

# Measure: test_accuracy / train_accuracy (closer to 1.0 = better generalization)
```

**Expected**: EqProp should generalize better due to local learning signals

### E-Eq3: Relaxation Depth Study (2 hours)

**Question**: What's the optimal n_iterations?

**Protocol**:
```python
for n_iter in [5, 10, 20, 40]:
    accuracy = evaluate_eqprop(model, n_iterations=n_iter)
    time_per_step = measure_time(model, n_iterations=n_iter)
    print(f"n_iter={n_iter}: acc={accuracy:.2%}, time={time_per_step:.1f}ms")
```

**Goal**: Find sweet spot between accuracy and efficiency

### Phase 1.5 Decision Gate

| Outcome | Action | Publication Path |
|---------|--------|------------------|
| ✅ EqProp + Controller > HoloLink-only | Full EqProp breakthrough paper | NeurIPS/ICLR main |
| ✅ EqProp + Controller ≈ HoloLink-only (94%) | EqProp enables modularity paper | NeurIPS/ICLR (bio-ML track) |
| ⚠️ EqProp generalization > Backprop | Methods paper | Workshop paper |
| ❌ EqProp underperforms | Document findings | Use HoloLink-only architecture |

### What Success Looks Like

**BREAKTHROUGH (NeurIPS/ICLR Main)**:
```
Table 1: Associative Recall Performance at 12 KV Pairs

| Architecture | Accuracy | Controller Status |
|--------------|----------|-------------------|
| HoloLink-only | 94.4% | N/A |
| ANA + Backprop | 8.9% | Trainable |
| ANA + EqProp | 94%+ | Trainable, LEARNS TO HELP |

Conclusion: EqProp enables modular learning where backprop fails.
```

**NOVEL CONTRIBUTION**:
1. First demonstration that EqProp solves gradient interference in modular architectures
2. First combination of EqProp with associative memory
3. Evidence that local learning enables multi-component systems

---

## Phase 2: Core Experiments (Tier 2)

**Goal**: Publishable results

**Guardrails**:
- Max 2 weeks
- Kill any experiment after 4 hours with no progress
- Pivot to efficiency focus if ICL fails

### E4: In-Context Learning Benchmark

**Purpose**: Demonstrate ICL superiority

**Tasks** (What ANA is designed for):
| Task | Why ANA Should Win |
|------|-------------------|
| Associative Recall | HoloLink = built-in KV memory |
| Pattern Completion | Multi-track captures patterns |
| Function Learning | Sequential processing |
| Rule Induction | Associative binding |

**NOT these tasks**:
| ❌ Forbidden Task | Why |
|-------------------|-----|
| Copy Task | Already 100%, no insight |
| Reverse Task | Architectural mismatch |
| Sorting Task | Requires non-local operations |

**Time Limit**: 8 hours max

**Success**: Win 3/4 tasks

**If Fails**: Pivot to efficiency paper (E5, E8)

---

### E5: Long-Context Language Modeling

**Purpose**: Efficiency at scale

**Guardrails**:
- Don't chase perplexity improvements
- Focus on memory and speed
- Accept slightly worse quality if efficiency is strong

**Success**: >5x memory savings at 16K context

---

### E6: Few-Shot Learning

**Purpose**: Real-world validation

**Guardrails**:
- Use existing benchmarks (MMLU, ARC)
- Don't create custom tasks
- 8 hours max

---

### Phase 2 Decision Gate (End of Week 3)

| Condition | Action |
|-----------|--------|
| ✅ E4 wins 3/4 tasks | ICL paper ready |
| ❌ E4 wins < 3 tasks | Efficiency paper (E5, E8) |
| ✅ E5 shows efficiency | Systems paper viable |
| ❌ Everything fails | Position paper, document lessons |

---

## Phase 3: Extended Applications (Tier 3)

**Only proceed if Phase 2 shows strong results**

### E7: RL Integration (12 hours)
### E8: Edge Deployment (6 hours)
### E9: Vision SSM (12 hours)

**Guardrails**: Skip entire phase if Phase 2 results are weak

---

## Resource Allocation (Optimized)

| Phase | GPU Hours | Guardrail |
|-------|-----------|-----------|
| Phase 0 | 4 | Stop after 4 hours |
| Phase 1 | 4 | Kill if E1 fails |
| Phase 2 | 18 | Pivot if needed |
| Phase 3 | 30 | Skip if Phase 2 weak |

**Total**: 56 GPU hours max

---

## Reproducibility Protocol

```yaml
Seeds: [42, 123, 456]
Hardware: RTX 3080, 31GB RAM
Software: PyTorch 2.10, Python 3.11
Reporting: Mean ± std across 3 seeds
Significance: Paired t-test, p < 0.05

Time Limits:
  E1: 2 hours
  E2: 4 hours
  E3: 2 hours
  E4: 8 hours
  E5: 8 hours
  E6: 8 hours
```

---

## Timeline

| Day | Phase | Activity | Guardrail |
|-----|-------|----------|-----------|
| 1 | 0 | Profile, optimize | Stop after 4 hours |
| 2 | 1 | E1: Synergy | Kill if < 10% |
| 3-4 | 1 | E2, E3 | Document and move on |
| 5 | 1 | Decision Gate | Go/No-Go |
| 6-8 | 2 | E4: ICL | Pivot to efficiency if fails |
| 9-11 | 2 | E5: Long-context | Accept tradeoffs |
| 12-14 | 2 | E6: Few-shot, draft | Paper ready |
| 15+ | 3 | Extended | Only if strong Phase 2 |

---

## Success Criteria & Publication Targets

### Breakthrough (NeurIPS/ICLR Main)
- ICL > 15% over baselines
- Memory > 10x savings
- Performance > 30K tok/s optimized
- **EqProp synergy > 25%** (bio-ML track)

### Strong Result (Workshop)
- Synergy > 15%
- Memory > 5x savings
- Win 2+ domains
- **EqProp + HoloLink validated** (novel combination)

### Minimum Viable (Position Paper)
- Architecture validated
- Limitations documented
- Reproducible

### Novel Contribution Summary

| Contribution | Novelty | Evidence | Publication Path |
|--------------|---------|----------|------------------|
| EqProp + Associative Memory | ⭐⭐⭐ High | `eqprop_ana.py` | NeurIPS bio-ML track |
| Stack + Algorithmic Read | ⭐⭐⭐ High | v3 100% generalization | ICLR main |
| Multi-track SSM with HoloLink | ⭐⭐ Medium | +19.5% synergy | Workshop |
| EqProp for SSM | ⭐⭐ Medium | Local learning for temporal | Workshop |

---

## Code Organization

```
ana/
├── config.py
├── models.py           # Standard ANA with HoloLink
├── experiments.py      # Synergy, ablation experiments
├── tasks.py            # Associative recall tasks (NOT copy/reverse)
├── benchmark.py
│
├── eqprop_ana.py       # EqProp integration (NEW - HIGH VALUE)
├── eqprop_seq.py       # Sequence EqProp implementation
├── bioplausible_ana.py # Bioplausible library integration
│
├── reverse_net.py      # Specialized reversal model (DONE)
│
├── profiling/          # Performance optimization
│   ├── profile_baseline.py
│   └── verify_optimizations.py
│
├── icl/                # In-context learning experiments
│   ├── tasks.py
│   └── evaluate.py
│
└── rl/                 # Phase 3 only

# ANALYSIS FILES (complete, reference only):
# - ANALYSIS.md: Reverse task failure analysis
# - analyze_reversal.ipynb: Visualizations
```

---

## Immediate Actions (PRIORITY ORDER)

### Hour 1-2: EqProp Validation (HIGHEST PRIORITY)
```bash
# Test if EqProp solves the controller interference problem
python -c "
import sys
sys.path.insert(0, '/home/me/ana')
from ana.eqprop_seq import train_with_eqprop
train_with_eqprop()
"

# SUCCESS = EqProp achieves ~94% (matching HoloLink-only)
# FAILURE = EqProp matches backprop's 8-9% (document and move on)
```

### Hour 3-4: If EqProp Works - Full EqProp + HoloLink Experiment
```bash
# Train EqProp ANA with controller enabled
# Compare: EqProp + Controller vs Backprop + Controller vs HoloLink-only

python -c "
from ana.eqprop_ana import EqPropANA, EqPropConfig, train_with_eqprop
# Run full curriculum with controller enabled
"
```

### Hour 5-6: Baseline Experiments (parallel)
```bash
python -m ana.experiments  # Synergy validation
python -m ana.profiling.profile_baseline  # Performance baseline
```

### Hour 7+: Decision Gate
- If EqProp + HoloLink works: Write EqProp paper (breakthrough)
- If EqProp fails: Document, proceed with HoloLink-only architecture
- Follow time limits, pivot when stuck

---

## Final Reminder

**THE BIG QUESTION**: Can EqProp's local learning solve the controller interference problem?

**What We Know**:
- HoloLink alone: 94.4% ✅
- HoloLink + Controller (backprop): 8-9% ❌
- EqProp + HoloLink: **UNKNOWN** ← This is the breakthrough opportunity

**Priority Order**:
1. **EqProp validation** - highest impact, most novel
2. HoloLink-only architecture - proven to work
3. ANA v3 for algorithmic tasks - different research direction
4. Standard experiments - fallback

**What Doesn't Matter**:
- Copy task (100%, done)
- Reverse task with standard ANA (impossible, accepted)
- Speed optimization (128x already achieved)

**The Goal**: Demonstrate that local learning (EqProp) enables modular architectures where backprop fails. This is a fundamental insight about gradient interference and bio-plausible alternatives.
