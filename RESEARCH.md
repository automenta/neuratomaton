# ANA Research Agenda
## Adaptive Neural Automaton: Multi-Track SSM with Holographic Memory

---

## Executive Summary

**Mission**: Achieve O(1) inference with Transformer-level recall capability through multi-track state space models with holographic associative memory.

**Current State** (as of Feb 2026):
- 100% accuracy on single-KV associative recall (noise 2-10 tokens)
- +11-21% improvement over BaselineSSM on short-medium sequences
- Key breakthrough: Focused loss (query position only) vs diluted loss
- ~100K parameters tested

**Critical Gaps**:
1. Multi-KV capacity unvalidated (core HoloLink claim)
2. Copy/Reverse tasks at 0% (architectural limitation)
3. No extrapolation data beyond training lengths
4. Not scaled beyond toy models

---

## Research Questions

### Primary Questions (Must Answer)

| # | Question | Why Critical | Current Status |
|---|----------|--------------|----------------|
| Q1 | How many KV pairs can HoloLink store before interference degrades recall? | Validates core architectural innovation | Unknown |
| Q2 | Does ANA generalize to 2x/4x/8x training sequence length? | Tests "infinite context" claim | Unknown |
| Q3 | Why does Copy task fail at 0%? | Reveals architectural blind spots | Uninvestigated |
| Q4 | What is the compute-optimal scaling configuration? | Required for practical deployment | Unknown |

### Secondary Questions (Should Answer)

| # | Question | Why Important |
|---|----------|---------------|
| Q5 | Do tracks naturally specialize (syntax vs semantics)? | Validates multi-track hypothesis |
| Q6 | Is dynamic gating essential or can static parameters suffice? | Ablation shows -11.6% without controller |
| Q7 | What key_dim is needed for N KV pairs? | Theoretical capacity bound |
| Q8 | Does orthogonal initialization reduce interference? | Memory efficiency |

---

## Phase 1: Capacity & Interference Characterization

### 1.1 Multi-KV Capacity Curve

**Objective**: Quantify HoloLink memory capacity and locate the "interference cliff"

**Hypothesis**: Holographic binding (outer product) allows more KV pairs than fixed-state SSM before degradation

**Experiment Design**:
```
Independent Variables:
  - num_kv_pairs: [1, 2, 4, 8, 16, 32, 64]
  - noise_between_pairs: [3, 5, 10, 20]
  - key_dim: [32, 64, 128, 256]

Dependent Variables:
  - Final retrieval accuracy (per KV position)
  - Interference matrix (recall of KV_i when querying KV_j)
  - Training convergence speed

Models to Compare:
  - ANA (full)
  - ANA (no HoloLink)
  - BaselineSSM
  - Transformer (reference upper bound)
```

**Success Criteria**:
- ANA maintains >90% accuracy at 2x the KV pairs of BaselineSSM
- Clear interference cliff identified (accuracy vs num_kv_pairs)

**Implementation**:
```bash
# Create experiment script
python -c "
from ana.models import ANAModel, BaselineSSM
from ana.config import ANAConfig
from experiment_v2 import MultiKVDataset, train_model, evaluate_model
import json

results = {}
for num_kv in [1, 2, 4, 8, 16, 32]:
    for key_dim in [32, 64, 128]:
        config = ANAConfig(d_model=64, state_dim=64, key_dim=key_dim)
        # ... run training and evaluation
        results[f'kv{num_kv}_key{key_dim}'] = accuracy
        
with open('archive/capacity_study.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

**Deliverables**:
- Capacity curve plot: Accuracy vs Num KV Pairs
- Interference heatmap for each key_dim
- Comparison table: ANA vs Baseline capacity limits

**Estimated Time**: 2 days

---

### 1.2 Key Dimension Scaling Study

**Objective**: Derive theoretical relationship between key_dim and memory capacity

**Background**: In holographic memory, key_dim controls the dimensionality of the binding space. Higher dimensions should reduce interference.

**Experiment**:
```python
# Test matrix
key_dims = [16, 32, 64, 128, 256]
num_kv_pairs = [4, 8, 16, 32]

# For each combination, measure:
# - Retrieval accuracy
# - Key collision rate (similarity between learned key vectors)
# - Memory footprint
```

**Expected Finding**: capacity ≈ key_dim / log(vocab_size)

**Deliverables**:
- Empirical scaling law for key_dim vs capacity
- Theoretical bound analysis
- Recommendations for key_dim at different scales

**Estimated Time**: 1 day

---

### 1.3 Orthogonal vs Learned Projections

**Objective**: Determine if random orthogonal initialization provides interference reduction

**Hypothesis**: Orthogonal keys maximize angular separation, reducing crosstalk

**Experiment**:
```bash
python run_experiment.py train 3a --orthogonal-init --epochs 20
python run_experiment.py train 3a --no-orthogonal-init --epochs 20
# Compare on Multi-KV task
```

**Deliverables**:
- Accuracy comparison: orthogonal vs learned
- Key vector similarity analysis
- Recommendation for default setting

**Estimated Time**: 0.5 days

---

## Phase 2: Failure Mode Analysis

### 2.1 Copy Task Investigation

**Objective**: Understand and fix 0% copy accuracy

**Current State**: Copy accuracy remains at 0% throughout training

**Hypotheses to Test**:

| Hypothesis | Test | Expected Outcome if True |
|------------|------|--------------------------|
| H1: Position encoding insufficient | Remove pos encoding, add learned position | Accuracy improves |
| H2: Track decay too aggressive | Set α=0.999 for one track | Accuracy improves |
| H3: Loss doesn't penalize copy | Add copy auxiliary loss | Accuracy improves |
| H4: State dimension too small | Increase state_dim to 256 | Accuracy improves |
| H5: Copy requires attention | Compare with transformer | Transformer succeeds, ANA fails |

**Experiment Design**:
```python
# Create CopyTask dataset
class CopyDataset(Dataset):
    """
    Sequence: [COPY] [A] [B] [C] ... [END] [A] [B] [C] ...
    Task: Reproduce input verbatim after END marker
    """
    def __init__(self, size=1000, seq_len=10, vocab_size=30):
        self.TOK_COPY = 1
        self.TOK_END = 2
        self.content = list(range(3, vocab_size))
        # ...

# Test configurations
configs = [
    {'state_dim': 64, 'track_count': 2},
    {'state_dim': 256, 'track_count': 2},
    {'state_dim': 64, 'track_count': 2, 'verbatim_track': True},
    {'state_dim': 64, 'track_count': 2, 'copy_aux_loss': True},
]
```

**Deliverables**:
- Root cause analysis
- Fix implementation (if possible)
- Architectural recommendations

**Decision Point**: If copy fundamentally requires attention, document as architectural limitation

**Estimated Time**: 2 days

---

### 2.2 Reverse Task Investigation

**Objective**: Understand reversal capability and limitations

**Background**: Reversal requires either bidirectional processing or explicit stack-like storage

**Experiment**:
```python
class ReverseDataset(Dataset):
    """
    Sequence: [REVERSE] [A] [B] [C] [END] -> [C] [B] [A]
    """
    pass

# Test configurations:
# 1. Standard ANA (likely fails)
# 2. ANA with backward track (bidirectional)
# 3. ANA with explicit stack mechanism
# 4. Transformer baseline
```

**Potential Solutions**:
1. Bidirectional encoding during training, causal during inference
2. Dedicated "stack track" with push/pop gating
3. Accept as architectural limitation (autoregressive models struggle)

**Estimated Time**: 1.5 days

---

## Phase 3: Extrapolation & Generalization

### 3.1 Sequence Length Extrapolation

**Objective**: Test if ANA generalizes beyond training sequence lengths

**Critical Question**: Can a model trained on noise 5-15 work on noise 60-240?

**Experiment Design**:
```
Training: noise ∈ [5, 15]
Testing:  noise ∈ [15, 30, 60, 120, 240, 480]

Measure:
  - Accuracy degradation curve
  - Ret gate values at different lengths
  - Track alpha values (do they adapt?)
```

**Expected Outcomes**:
- SSM-based models should extrapolate better than attention (no position limit)
- Performance may degrade if decay rates are too aggressive for long sequences

**Deliverables**:
- Extrapolation curve: Accuracy vs Test Length
- Comparison: ANA vs Transformer on extrapolation
- Analysis of learned decay rates

**Estimated Time**: 1 day

---

### 3.2 Distribution Shift Robustness

**Objective**: Test generalization to different noise patterns

**Experiment**:
```
Training: Random uniform noise
Testing:
  - Repeated patterns: [A, A, A, B, B, B, ...]
  - Structured noise: [1, 2, 3, 1, 2, 3, ...]
  - Rare tokens: High-frequency vocab items during train, rare during test
  - Adversarial noise: Tokens that look like key/value markers
```

**Estimated Time**: 1 day

---

## Phase 4: Scaling Laws

### 4.1 Compute-Optimal Scaling Configuration

**Objective**: Determine optimal architecture configuration for different parameter budgets

**Parameter Dimensions**:
- d_model: [64, 128, 256, 512, 768]
- state_dim: [64, 128, 256, 512, 768]
- num_layers: [2, 4, 6, 8, 12]
- track_count: [1, 2, 4, 8]
- key_dim: [32, 64, 128, 256]

**Experiment**:
```bash
# Systematic sweep (use existing scaling configs)
python ana/experiments.py --study scaling --scale tiny
python ana/experiments.py --study scaling --scale small
python ana/experiments.py --study scaling --scale medium
python ana/experiments.py --study scaling --scale large
python ana/experiments.py --study scaling --scale xlarge
```

**Analysis**:
- Loss vs parameters curve
- Inference speed vs parameters
- Memory usage vs parameters
- Optimal config for each budget

**Estimated Time**: 3-4 days (GPU intensive)

---

### 4.2 Track Count Scaling

**Objective**: Understand if more tracks help at scale

**Hypothesis**: More tracks may help with specialization but add compute overhead

**Experiment**:
```
For each param_budget in [100K, 500K, 2M, 10M]:
  For track_count in [1, 2, 4, 8]:
    Train on Multi-KV task
    Measure: Accuracy, training time, inference time
```

**Estimated Time**: 2 days

---

## Phase 5: Architectural Enhancements

### 5.1 Hierarchical HoloLink

**Objective**: Multi-scale memory for different temporal horizons

**Design**:
```
Layer 0: Fast HoloLink (decay=0.9) - recent context
Layer 1: Medium HoloLink (decay=0.99) - medium range
Layer 2: Slow HoloLink (decay=0.999) - long range
```

**Implementation**:
```python
class HierarchicalHoloLink(nn.Module):
    def __init__(self, config, decay_schedule=[0.9, 0.99, 0.999]):
        self.levels = nn.ModuleList([
            HoloLink(config, decay=d) for d in decay_schedule
        ])
    
    def forward(self, x, h):
        outputs = [level(x, h) for level in self.levels]
        return torch.stack(outputs).mean(dim=0)
```

**Estimated Time**: 2 days

---

### 5.2 Content-Addressable Retrieval Enhancement

**Objective**: Learn better key hashing to reduce interference

**Design**: Add MLP before key projection to create more discriminative keys

```python
class EnhancedHoloLink(HoloLink):
    def __init__(self, config):
        super().__init__(config)
        self.key_hash = nn.Sequential(
            nn.Linear(config.state_dim, config.key_dim * 2),
            nn.ReLU(),
            nn.Linear(config.key_dim * 2, config.key_dim)
        )
    
    def forward(self, x, h, M):
        k = F.normalize(self.key_hash(h), dim=-1)
        # ... rest of HoloLink
```

**Estimated Time**: 1 day

---

### 5.3 Adaptive Computation Time (ACT)

**Objective**: Allow model to "think" longer on hard queries

**Current Implementation**: `max_thinking_steps` parameter exists but not evaluated

**Experiment**:
```bash
python run_experiment.py train 3a --thinking-steps 4
# Compare accuracy and compute vs thinking-steps=0
```

**Metrics**:
- Accuracy improvement vs thinking steps
- Compute overhead
- Which samples trigger more thinking?

**Estimated Time**: 1 day

---

## Phase 6: Language Modeling Validation

### 6.1 WikiText-2/103 Benchmark

**Objective**: Validate on standard language modeling benchmarks

**Target**: Perplexity competitive with Mamba at same parameters

**Experiment**:
```bash
# Prepare data
python scripts/prepare_wikitext.py --dataset wikitext-2

# Train ANA
python run_experiment.py train lm --dataset wikitext-2 --epochs 10

# Evaluate
python run_experiment.py eval --checkpoint archive/model_wikitext.pt
```

**Success Criteria**:
- PPL < Mamba at 125M params (Mamba: ~33 on WikiText-103)
- Inference throughput >40K tok/s

**Estimated Time**: 3-5 days

---

### 6.2 Needle-in-Haystack (NIAH) Benchmark

**Objective**: Long-context retrieval validation

**Protocol**:
```
Context lengths: [1K, 2K, 4K, 8K, 16K, 32K]
Needle positions: [0%, 25%, 50%, 75%, 100%] of context
Measure: Retrieval accuracy at each position/length combination
```

**Success Criteria**:
- >90% accuracy at 4K context
- >80% accuracy at 16K context
- Graceful degradation (not cliff)

**Estimated Time**: 2-3 days

---

## Experiment Priority Matrix

| Priority | Experiment | Novelty | Impact | Effort | Dependencies |
|----------|------------|---------|--------|--------|--------------|
| P0 | Multi-KV Capacity (1.1) | High | High | Low | None |
| P0 | Copy Task Fix (2.1) | Medium | High | Medium | None |
| P1 | Extrapolation Study (3.1) | High | High | Low | None |
| P1 | Key Dim Scaling (1.2) | High | Medium | Low | 1.1 |
| P2 | Scaling Laws (4.1) | Medium | High | High | 1.1 |
| P2 | Hierarchical HoloLink (5.1) | Very High | High | Medium | 1.1, 1.2 |
| P3 | WikiText Benchmark (6.1) | Medium | High | High | 4.1 |
| P3 | NIAH Benchmark (6.2) | Medium | High | Medium | 3.1 |

---

## Timeline

### Week 1-2: Capacity & Failure Modes
- Days 1-2: Multi-KV Capacity Study (1.1)
- Days 3-4: Key Dimension Scaling (1.2)
- Day 5: Orthogonal Projections (1.3)
- Days 6-8: Copy Task Investigation (2.1)
- Days 9-10: Reverse Task Investigation (2.2)

### Week 3: Extrapolation & Scaling
- Day 11: Sequence Length Extrapolation (3.1)
- Day 12: Distribution Shift (3.2)
- Days 13-17: Scaling Laws Study (4.1, 4.2)

### Week 4-5: Architectural Enhancements
- Days 18-20: Hierarchical HoloLink (5.1)
- Day 21: Enhanced Retrieval (5.2)
- Day 22: ACT Evaluation (5.3)

### Week 6-8: Language Modeling
- Days 23-28: WikiText Benchmark (6.1)
- Days 29-33: NIAH Benchmark (6.2)

---

## Success Metrics

### Tier 1: Proof of Concept (Current)
- [x] Single-KV recall >95%
- [ ] Multi-KV (8 pairs) recall >80%
- [ ] Copy task >50%

### Tier 2: Research Contribution
- [ ] Extrapolation 4x training length >80%
- [ ] Clear capacity scaling law derived
- [ ] WikiText-2 PPL < 30 (at 125M params)

### Tier 3: Production Ready
- [ ] WikiText-103 PPL < 33
- [ ] NIAH 16K context >80%
- [ ] Inference >40K tok/s
- [ ] MMLU >38% (1.4B model)

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HoloLink capacity lower than expected | Medium | High | Fall back to hybrid attention-SSM |
| Copy task fundamentally impossible | Medium | Medium | Document as limitation, focus on recall |
| Scaling doesn't maintain advantage | Medium | High | Investigate layer normalization, initialization |
| Training instability at scale | High | Medium | Implement gradient checkpointing, mixed precision |

---

## Resources Required

### Compute
- GPU: 1x A100 or equivalent for scaling studies
- Storage: 50GB for datasets and checkpoints
- Time: ~200 GPU-hours for full agenda

### Code Dependencies
- PyTorch 2.0+
- transformers (for comparison)
- datasets (for WikiText)
- wandb or tensorboard (for logging)

---

## Reproducibility Checklist

- [ ] All experiments logged with random seeds
- [ ] Configurations saved to JSON
- [ ] Model checkpoints saved
- [ ] Training curves plotted
- [ ] Statistical significance computed (3+ runs per experiment)

---

## Publication Targets

### Primary
- ICLR/NeurIPS (architecture + scaling laws)

### Secondary
- ICML (theoretical capacity analysis)
- arXiv (rapid dissemination of findings)

---

## References

1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
2. S4: Efficiently Modeling Long Sequences with Structured State Spaces
3. RWKV: Reinventing RNNs for the Transformer Era
4. Plate, 1995: Holographic Reduced Representations
5. Associative Recall in the Wild (ICLR 2024)

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-09 | Initial research agenda | ANA Team |

---

## Quick Reference: Running Experiments

```bash
# Multi-KV Capacity
python experiment_v2.py

# Ablation Studies
python ana/experiments.py --study ablation --ablation full
python ana/experiments.py --study ablation --ablation no_hololink
python ana/experiments.py --study ablation --ablation no_controller
python ana/experiments.py --study ablation --ablation static_only

# Scaling Study
python ana/experiments.py --study scaling --scale small
python ana/experiments.py --study comparison --scales small medium large

# Full Study
python ana/experiments.py --study full --output archive/full_study

# Training with specific config
python run_experiment.py train 3a --epochs 30 --thinking-steps 2
python run_experiment.py train 3a --no-hololink
python run_experiment.py train 3a --no-controller

# Evaluation
python run_experiment.py eval --checkpoint archive/model.pt
python run_experiment.py benchmark --compare
python run_experiment.py analyze --checkpoint archive/model.pt
```
