# Bio-Plausible Adaptive Neural Automaton (Bio-ANA)

**Integration of neuratomaton (ANA) + bioplausible (EqProp)**

---

## Executive Summary

Develop bio-plausibly trained neural networks by combining:
- **ANA Architecture**: Multi-track SSM with HyperController gating + HoloLink holographic memory (O(1) inference)
- **EqProp Learning**: Equilibrium Propagation for local, energy-based training (O(1) memory, no backprop)

**Goal**: Deployable models on commodity hardware (≤10GB VRAM) that outperform backprop-trained equivalents in efficiency while matching/exceeding accuracy.

---

## Research Questions

| # | Question | Success Criterion | Test Method |
|---|----------|-------------------|-------------|
| 1 | Can EqProp train SSM dynamics stably? | >95% AR accuracy, monotonic energy decrease | Energy monitoring, synthetic tasks |
| 2 | Does HoloLink memory integrate with energy-based learning? | >90% MQAR (64 pairs), capacity scaling log-linear | Capacity sweep experiments |
| 3 | What is the efficiency/accuracy tradeoff vs backprop? | <10% accuracy gap, >5x memory savings | Side-by-side training, resource profiling |
| 4 | Does bio-plausibility improve noise tolerance/continual learning? | >5% improvement on noisy data, <10% catastrophic forgetting | Noise injection, continual learning benchmarks |
| 5 | Can models deploy on edge hardware? | Functional on INT8/FP16, <2GB RAM, <100ms latency | Edge device testing |

---

## Architecture Specification

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Bio-ANA Layer                            │
│                                                                 │
│  ┌──────────────┐                                              │
│  │HyperController│──► α_syntax, α_semantic, α_logic            │
│  │   (MLP)      │──► β_syntax, β_semantic, β_logic             │
│  └──────────────┘──► memory_gate, recall_gate                  │
│         │                                                       │
│    ┌────┴────────────────┐                                     │
│    ▼                     ▼                                     │
│ ┌────────┐          ┌────────┐    Multi-Track LRU              │
│ │ Syntax │          │Semantic│    h_t = α·h_{t-1} + β·x_t      │
│ │ (fast) │          │ (slow) │    [EqProp treats states as     │
│ └───┬────┘          └───┬────┘     equilibrium points]         │
│     │                   │                                       │
│     └─────────┬─────────┘                                       │
│               ▼                                                 │
│        ┌─────────────┐                                         │
│        │  HoloLink   │    M_t = M_{t-1} + K(h)·V(h)^T         │
│        │(Linear Attn)│    retrieve = M_t · Q(x)               │
│        │ [Hebbian]   │    ΔM = η(q·k^T - λ·M) [Oja's rule]     │
│        └──────┬──────┘                                         │
│               │                                                 │
│          Mix + Residual                                         │
└───────────────┼─────────────────────────────────────────────────┘
                ▼
```

### EqProp Integration

| Phase | Operation | Mathematical Form | Purpose |
|-------|-----------|-------------------|---------|
| **Free Phase** | Relax states to equilibrium | h* = argmin_h E(h, x) via dh/dt = -∂E/∂h | System settles to energy minimum |
| **Nudged Phase** | Perturb toward target | h̃ = h* + ε·∇_y L(y, target) | Compute local gradient signal |
| **Update** | Local weight change | ΔW = (h̃ - h*)·x^T | Hebbian-like local update |

**Energy Function for SSM Track**:
```
E_i(h_i, x, h_{neighbors}) = h_i²/(2τ) - h_i·f_i(x, h_{neighbors})
f_i = tanh(W_in·x + W_rec·h_{neighbors} + b)
```

**Key Innovation**: Treat SSM hidden states (h_syntax, h_semantic, h_logic) as EqProp equilibrium variables with track-specific energy functions.

---

## Implementation Roadmap

### Phase 1: EqProp Core Implementation

**Objective**: Implement Equilibrium Propagation primitives

**Tasks**:
- [ ] Implement energy function base class
  ```python
  class EnergyFunction:
      def energy(self, h, x): pass
      def gradient(self, h, x): pass
  ```
- [ ] Implement free phase relaxation with adaptive iterations
  - Fixed iterations: 20 (configurable)
  - Early stopping: ||dh/dt|| < 1e-6
  - Momentum-based acceleration
- [ ] Implement nudged phase with configurable ε (0.05-0.5)
- [ ] Implement local weight update with spectral normalization
  ```
  ΔW = (h_nudged - h_free) · x^T
  W ← W + η·ΔW
  W ← W / max(||W||_2, σ_max)  # Spectral norm
  ```
- [ ] Add Dale's law constraint (sign-preserving via separate W+, W-)
  ```
  ΔW+ = max(0, ΔW); ΔW- = max(0, -ΔW)
  W+ ← (1-η_wd)·W+ + η·ΔW+
  W- ← (1-η_wd)·W- + η·ΔW-
  ```
- [ ] Unit tests:
  - XOR convergence (<1000 iterations)
  - Energy monotonicity verification
  - Gradient accuracy (finite diff check)
- [ ] Visualization tools: energy landscape, convergence curves

**Deliverables**:
- `ana/eqprop/energy.py` - Energy function base
- `ana/eqprop/relaxation.py` - Free/nudged phase
- `ana/eqprop/update.py` - Local update rules
- `ana/eqprop/constraints.py` - Dale's law, spectral norm
- `tests/test_eqprop.py` - Unit tests
- `tools/visualize_energy.py` - Visualization

**Success Criteria**:
| Test | Target | Status |
|------|--------|--------|
| XOR convergence | <1000 iterations | ✓ Required |
| Energy decrease | Monotonic >95% steps | ✓ Required |
| Gradient accuracy | Error <1e-6 | ✓ Required |

---

### Phase 2: Bio-ANA Architecture

**Objective**: Integrate EqProp with ANA multi-track SSM

**Tasks**:
- [ ] Define SSM state variables as EqProp equilibrium variables
  ```python
  class SSMEnergyFunction(EnergyFunction):
      def __init__(self, track_type, dim):
          self.track_type = track_type  # 'syntax', 'semantic', 'logic'
          self.dim = dim
          self.tau = {'syntax': 0.5, 'semantic': 2.0, 'logic': 1.0}[track_type]
  ```
- [ ] Implement track-specific energy functions:
  - **Syntax track**: Fast decay (τ=0.5), sparse activation
  - **Semantic track**: Slow decay (τ=2.0), dense representations
  - **Logic track**: Binary-like dynamics (tanh³), for reasoning
- [ ] Implement HoloLink with Hebbian update (Oja's rule variant)
  ```python
  class HoloLinkMemory:
      def update(self, query, key, value):
          delta = eta * (query @ key.T - lambda_decay * self.memory)
          self.memory += delta
  ```
- [ ] Integrate HyperController as modulation of equilibrium dynamics
  ```python
  alpha_mod = self.controller(x)  # [0, 1]
  effective_tau = tau * (1 + alpha_mod)
  ```
- [ ] Implement top-k sparsity constraint (10% by default)
- [ ] Validate forward pass compatibility with existing `model_v3.py`

**Deliverables**:
- `ana/bio_ana.py` - Bio-ANA model
- `ana/bio_ana/tracks.py` - Track-specific energy functions
- `ana/bio_ana/hololink.py` - HoloLink with Hebbian updates
- `ana/bio_config.py` - Extended configuration
- Integration tests with existing benchmarks

**Success Criteria**:
| Test | Target | Status |
|------|--------|--------|
| Forward pass | Valid outputs, shape match | ✓ Required |
| Free phase convergence | <50 iterations | ✓ Required |
| Memory overhead | <2× standard ANA | ✓ Required |
| Gradient check | Error <1e-4 | ✓ Required |

---

### Phase 3: Training Pipeline

**Objective**: Bio-plausible training curriculum

**Tasks**:
- [ ] Implement 3-stage curriculum:
  - **Stage 0**: Simple AR (5-15 noise tokens), single KV pair
  - **Stage 1**: Complex AR + Stack (15-30 noise), multi-pair
  - **Stage 2**: MQAR + Text (30-50 noise), full context
- [ ] Implement adaptive relaxation scheduler:
  - Start: 50 iterations
  - End: 10 iterations
  - Decay: linear or cosine
- [ ] Implement curriculum advancement:
  - Stage 0 → 1: AR accuracy >98% for 3 consecutive epochs
  - Stage 1 → 2: MQAR (16 pairs) >90% for 3 epochs
- [ ] Add gradient consistency checks:
  - Finite difference vs EqProp: <5% variance
  - Weekly drift monitoring
- [ ] Implement mixed precision (FP16/BF16) with loss scaling
- [ ] Implement learning rate scheduling:
  - Warmup: 10% of total steps
  - Decay: cosine with minimum 1e-5

**Deliverables**:
- `ana/bio_training.py` - Training pipeline
- `ana/bio_training/curriculum.py` - Stage management
- `ana/bio_training/scheduler.py` - Relaxation/LR scheduling
- `run_bio_experiment.py` - CLI interface (extends `run_experiment.py`)
- Training logs: TensorBoard, JSON checkpoints

**Success Criteria**:
| Stage | Metric | Target | Status |
|-------|--------|--------|--------|
| 0 | AR accuracy | >98% | ✓ Required |
| 1 | AR + stack | >90% | ✓ Required |
| 2 | MQAR (32 pairs) | >85% | ✓ Required |

---

### Phase 4: Optimization & Efficiency

**Objective**: Maximize training/inference efficiency

**Tasks**:
- [ ] Implement lazy/early stopping for relaxation:
  ```python
  def relax(h, x, max_iter=20, tol=1e-6):
      for i in range(max_iter):
          dh = compute_gradient(h, x)
          h -= lr * dh
          if torch.norm(dh) < tol:
              break
  ```
- [ ] Implement event-driven updates:
  - Skip relaxation for regions with <1% change
  - Track activation statistics
- [ ] Implement INT8 quantization:
  - Static: weights post-training
  - Dynamic: activations during inference
- [ ] Implement ternary quantization:
  ```python
  def ternarize(W, threshold=0.7):
      scale = torch.mean(torch.abs(W))
      W_ternary = torch.zeros_like(W)
      W_ternary[W > threshold * scale] = 1
      W_ternary[W < -threshold * scale] = -1
      return W_ternary * scale
  ```
- [ ] CUDA kernels for parallel scan (optional, leverage PyTorch native first)
- [ ] Memory profiling with PyTorch Profiler
- [ ] Benchmark suite:
  - Inference: tokens/sec @ 512, 2048, 8192 seq lengths
  - Training: memory usage @ batch sizes 8, 16, 32
  - Power: W/GPU via nvidia-smi

**Deliverables**:
- `ana/eqprop_cuda.py` - CUDA kernels (optional)
- `ana/quantization.py` - Quantization utilities
- `ana/profiling.py` - Profiling tools
- `results/efficiency/` - Benchmark results

**Success Criteria**:
| Metric | Target | Status |
|--------|--------|--------|
| Inference speed | >40K tokens/sec (125M) | ✓ Required |
| Training memory | <4GB (125M) | ✓ Required |
| INT8 accuracy loss | <2% | ✓ Required |
| Power efficiency | <180W @ 40K tok/s | ✓ Required |

---

### Phase 5: Comprehensive Evaluation

**Objective**: Rigorous comparison with baselines

#### Benchmark Suite

| Category | Benchmark | Metric | Target | Baseline (Backprop) |
|----------|-----------|--------|--------|---------------------|
| **Synthetic** | AR (single KV) | Accuracy | >98% | 99% |
| | AR (50 noise) | Accuracy | >95% | 97% |
| | MQAR (16 pairs) | Accuracy | >95% | 90% (Mamba) |
| | MQAR (64 pairs) | Accuracy | >90% | 72% (Mamba) |
| | Copy task | Accuracy | >99% | 95% |
| | Reverse task | Accuracy | >85% | 60% |
| | Induction Heads | Accuracy | >95% | 90% |
| **Language** | WikiText-103 PPL | Perplexity | <32 | 33 (Mamba) |
| | Pile PPL | Perplexity | <10.5 | 11.0 (Mamba) |
| | Char-level LM | PPL | <1.5 | 1.6 |
| **Downstream** | MMLU (1.4B) | Accuracy | >38% | 35-40% |
| | HellaSwag | Accuracy | >52% | 50-55% |
| | PIQA | Accuracy | >76% | 75-78% |
| **Efficiency** | Memory (8K ctx) | VRAM | <1GB | >12GB (Transformer) |
| | Memory (64K ctx) | VRAM | <1.5GB | >100GB (Transformer) |
| | Inference speed | Tokens/sec | >40K | 15K (Transformer) |
| | Training speed | Samples/sec | 0.8× backprop | 1.0× (backprop) |
| **Bio-Fidelity** | Noise tolerance (5% Gaussian) | Accuracy delta | >+5% | baseline |
| | Noise tolerance (10% analog) | Accuracy delta | >+3% | baseline |
| | Continual learning | Forgetting | <10% | 10-20% |
| | Spike-timing correlation | Correlation | >0.7 | N/A |

#### Experimental Protocol

**Reproducibility Protocol**:
- Seeds: [42, 123, 456] for all experiments
- Checkpoint management:
  - Save every 1000 steps (latest)
  - Save best validation model
  - Save epoch boundaries
  - Use `torch.save` with version control
- Environment tracking:
  - Record PyTorch version, CUDA version
  - Log GPU specs (nvidia-smi)
  - Capture random states
- Data integrity:
  - Hash all datasets on load
  - Validate splits (train/val/test)

**Model Variants to Train**:

| Config | Params | d_model | Tracks | Stack | Hardware | Seeds |
|--------|--------|---------|--------|-------|----------|-------|
| nano | 10M | 128 | 2 | 3 | RTX 3080 | 3 |
| small | 125M | 512 | 3 | 5 | RTX 3080 | 3 |
| base | 360M | 768 | 3 | 5 | A100 | 3 |
| large | 1.4B | 1024 | 4 | 8 | A100×2 | 3 |

**Baseline Models**:
- Transformer (standard attention)
- Mamba (SSM with selective scan)
- S4 (structured state space)
- ANA-backprop (same architecture, trained with backprop)

**Tasks**:
- [ ] Train all models on identical data splits
- [ ] Run full benchmark suite (5 seeds each)
- [ ] Record training time, memory, energy
- [ ] Statistical analysis:
  - Paired t-tests (Bio-ANA vs each baseline)
  - Cohen's d for effect sizes
  - Bootstrap confidence intervals (95%)
- [ ] Ablation studies (single seed, full sweep):
  - Bio-ANA vs ANA-backprop
  - EqProp vs hybrid (EqProp + backprop)
  - With/without HoloLink
  - With/without Dale's law
  - Varying relaxation iterations: [5, 10, 20, 50]
  - Varying sparsity: [5%, 10%, 20%, 40%]
  - Varying spectral radius: [0.90, 0.95, 0.99, 0.999]

**Statistical Validation Criteria**:
- **Significance**: p < 0.05 (Bonferroni-corrected for multiple comparisons)
- **Effect size**: d > 0.8 for "large" benefit claims
- **Confidence**: 95% CI must exclude zero for primary metrics

**Deliverables**:
- `experiments/exp_bio_benchmarks.py` - Main benchmark runner
- `experiments/exp_ablations.py` - Ablation sweep
- `experiments/analysis.py` - Statistical analysis
- `results/benchmarks/` - Full results (JSON, CSV, plots)
- Statistical report (PDF/Markdown)

**Success Criteria**:
| Criterion | Target | Status |
|-----------|--------|--------|
| Significance | p<0.05 on ≥50% metrics | ✓ Required |
| Effect size | d>0.8 on ≥2 metrics | ✓ Required |
| Reproducibility | SD < 0.02 across seeds | ✓ Required |

---

### Phase 6: Edge Deployment

**Objective**: Real-world deployment validation

**Hardware Targets**:
| Platform | RAM | Compute | Target |
|----------|-----|---------|--------|
| Raspberry Pi 4 | 4GB | ARM Cortex | Nano (10M) |
| Jetson Nano | 4GB | 128 CUDA cores | Nano (10M) |
| Laptop CPU | 8GB | 8-core | Small (125M) |
| Laptop GPU | 8GB | RTX 3050 | Small (125M) |

**Tasks**:
- [ ] Export to ONNX format:
  - Static graph optimization
  - Opset version compatibility
- [ ] PyTorch Mobile conversion:
  - iOS/Android support
  - CoreML/TFLite fallbacks
- [ ] FPGA synthesis (optional):
  - HLS generation
  - Resource utilization analysis
- [ ] Demo applications:
  - On-device text generation
  - Real-time sequence classification
  - Batch processing pipeline
- [ ] Power profiling:
  - Battery drain on mobile
  - Thermal throttling analysis
- [ ] Latency benchmarking:
  - Cold start vs warm
  - Batch size sweep (1, 4, 16)

**Deliverables**:
- `deploy/export_onnx.py` - ONNX export
- `deploy/mobile_setup/` - Mobile scripts
- `deploy/fpga/` - FPGA synthesis (optional)
- `demos/text_gen.py` - Text generation demo
- `demos/edge_app.py` - Edge application
- Deployment guide (README_DEPLOY.md)

**Success Criteria**:
| Platform | Metric | Target | Status |
|----------|--------|--------|--------|
| Raspberry Pi | RAM | <2GB | ✓ Required |
| Jetson Nano | Latency | <100ms | ✓ Required |
| Mobile app | Battery | <5% per 1000 tokens | ✓ Required |
| ONNX export | Compatibility | All ops supported | ✓ Required |

---

## Data Pipeline

### Data Collection & Preprocessing

**Datasets**:

| Dataset | Size | Tokens | Format | Source |
|---------|------|--------|--------|--------|
| WikiText-103 | 103M | 103M | Pre-tokenized | HuggingFace |
| The Pile | 825GB | 300B (subset) | JSONL | EleutherAI |
| Synthetic AR | 10K | Variable | Generated on-demand | - |
| Char-level LM | 1MB | 1M characters | ASCII | Local files |

**Data Pipeline Implementation**:
```python
# ana/bio_training/data.py
class DataManager:
    def __init__(self, config):
        self.cache_dir = config.cache_dir
        self.hash_db = HashDatabase()
    
    def load_dataset(self, name, split='train'):
        path = self._get_cached_path(name, split)
        if not path:
            path = self._download_and_cache(name, split)
        return torch.load(path)
    
    def _download_and_cache(self, name, split):
        # Download, verify hash, save
        pass
    
    def validate_integrity(self, dataset):
        # Compute and verify hash
        pass
```

**Data Splits**:
- Train: 80%
- Validation: 10%
- Test: 10%
- Fixed splits across all experiments

**Data Augmentation**:
- Synthetic: noise injection (Gaussian, salt-pepper)
- Text: back-translation (optional for robustness)

**Deliverables**:
- `ana/bio_training/data.py` - Data manager
- `data/registry.json` - Dataset registry with hashes
- `scripts/prepare_datasets.py` - Download/preprocess script

---

## Progressive Compute Investment

### Milestone-Based Compute Allocation

| Milestone | GPU Hours | Cumulative | Decision Point | Convincing Result |
|-----------|-----------|------------|----------------|-------------------|
| **M0: Proof of Concept** | 1 | 1 | Continue/abort | EqProp converges on simple task |
| **M1: Core Validation** | 9 | 10 | Continue/abort/adjust | EqProp beats baseline on AR |
| **M2: Architecture Integration** | 40 | 50 | Continue/abort/adjust | Bio-ANA trains on curriculum |
| **M3: Training Scale-up** | 50 | 100 | Full evaluation plan | Bio-ANA matches backprop accuracy |
| **M4: Optimization** | 150 | 250 | Finalize architecture | Efficiency targets met |
| **M5: Full Evaluation** | 400 | 650 | Deployment decision | Statistical significance achieved |
| **M6: Edge Validation** | 0 | 650 | Project complete | Deployment functional |
| **Contingency** | 350 | 1000 | Risk buffer | Iterations, retries |

### Milestone Details

#### M0: Proof of Concept (~1 GPU hour)

**Objective**: Validate EqProp core mechanism

**Experiments**:
- XOR problem: 500 iterations × 3 random seeds
- Simple regression: 1000 iterations × 3 seeds
- Energy monitoring: confirm monotonic decrease

**Success Criteria** (All required):
- XOR convergence: <1000 iterations with accuracy >95%
- Energy decrease: Monotonic on >95% of iterations
- Gradient accuracy: Finite difference error <1e-6

**Deliverables**:
- `results/m0/xor_convergence.json`
- `results/m0/energy_plots.png`

**Decision Point**: If convergence fails → adjust hyperparameters (2 hours retry). If still fails → reconsider approach or use hybrid training.

---

#### M1: Core Validation (~9 GPU hours)

**Objective**: EqProp vs Backprop on synthetic tasks

**Experiments** (3 seeds each):
- Associative Recall (single KV, 10-50 noise): 2 hours
- Energy landscape analysis: 1 hour
- Gradient accuracy sweep: 1 hour
- Relaxation iteration sensitivity: 1 hour
- Spectral radius sweep: 1 hour
- Dale's law ablation: 1 hour
- Documentation & analysis: 2 hours

**Success Criteria** (≥2 of 3 required):
| Metric | EqProp | Backprop | Required |
|--------|--------|----------|----------|
| AR accuracy (10 noise) | >95% | 98% | Within 5% |
| AR accuracy (50 noise) | >90% | 95% | Within 10% |
| Energy monotonicity | >90% | N/A | ✓ Required |
| Convergence time | <50 iters | N/A | ✓ Required |

**Deliverables**:
- `results/m1/ar_comparison.json`
- `results/m1/energy_landscapes/`
- `results/m1/m1_report.md`

**Decision Point**: If accuracy gap >15% → implement hybrid training (EqProp + backprop layers). If convergence >100 iterations → add momentum acceleration.

---

#### M2: Architecture Integration (~40 GPU hours)

**Objective**: Integrate EqProp with ANA architecture

**Experiments**:
- Nano model (10M params) training:
  - Stage 0 curriculum (AR): 8 hours × 3 seeds = 24 hours
- Track ablation (2 vs 3 tracks): 4 hours
- HoloLink ablation (on/off): 4 hours
- Controller ablation (static vs dynamic): 4 hours
- Integration tests: 4 hours

**Success Criteria**:
| Stage | Metric | Target |
|-------|--------|--------|
| Stage 0 (AR, 15 noise) | Accuracy | >95% |
| Stage 0 (AR, 30 noise) | Accuracy | >90% |
| Multi-track benefit | Δ vs single-track | >+3% |
| HoloLink benefit | Δ vs no-HoloLink | >+5% |

**Deliverables**:
- `results/m2/nano_model_ckpt.pt`
- `results/m2/ablation_results.json`
- `results/m2/architecture_report.md`

**Decision Point**: If Stage 0 accuracy <90% → increase relaxation iterations or adjust track dimensions. If HoloLink shows no benefit → redesign Hebbian update rule.

---

#### M3: Training Scale-up (~50 GPU hours)

**Objective**: Small model (125M) curriculum validation

**Experiments**:
- Small model training:
  - Stage 0 (AR): 10 hours × 2 seeds = 20 hours
  - Stage 1 (AR + Stack): 8 hours × 2 seeds = 16 hours
  - Stage 2 (MQAR + Text): 6 hours × 2 seeds = 12 hours
- Baseline comparison (Mamba backprop): 2 hours

**Success Criteria** (≥3 of 4 required):
| Metric | Bio-ANA | Mamba (Backprop) | Required |
|--------|---------|------------------|----------|
| Stage 0 AR accuracy | >95% | 98% | Within 5% |
| Stage 1 accuracy | >90% | 93% | Within 5% |
| Stage 2 MQAR (16 pairs) | >85% | 72% | Beat baseline |
| MQAR (64 pairs) | >80% | 60% | Beat baseline |
| Training memory | <4GB | 3GB | Within 2× |

**Deliverables**:
- `results/m3/small_model_stage0.pt`
- `results/m3/small_model_stage1.pt`
- `results/m3/small_model_stage2.pt`
- `results/m3/comparison_report.md`

**Decision Point**: If Stage 2 MQAR <70% → curriculum too aggressive, return to Stage 1. If memory >6GB → implement gradient checkpointing or reduce batch size.

**Go/No-Go Decision**: At 100 GPU hours total, if:
- ✅ M0, M1, M2, M3 success criteria met → Proceed to full optimization and evaluation
- ⚠️ Some criteria met with issues → Adjust approach, retry critical experiments (use contingency)
- ❌ Multiple criteria failed → Reconsider approach or pivot to hybrid training

---

#### M4: Optimization (~150 GPU hours)

**Objective**: Maximize efficiency and tune hyperparameters

**Experiments**:
- Hyperparameter sweep (small model):
  - Relaxation iterations: [5, 10, 20, 50] → 20 hours
  - Spectral radius: [0.90, 0.95, 0.99, 0.999] → 15 hours
  - Sparsity: [5%, 10%, 20%, 40%] → 15 hours
  - Learning rate: log sweep 1e-4 to 1e-2 → 10 hours
- Quantization experiments:
  - INT8 static + dynamic → 20 hours
  - Ternary quantization → 15 hours
  - Accuracy-accuracy tradeoff analysis → 10 hours
- Efficiency profiling:
  - Memory profiling @ batch 8/16/32 → 10 hours
  - Inference speed @ seq 512/2048/8192 → 15 hours
  - Power consumption analysis → 10 hours
- Mixed precision validation → 10 hours

**Success Criteria**:
| Metric | Before | After Optimization | Required |
|--------|--------|-------------------|----------|
| Inference speed | 20K tok/s | >40K tok/s | 2× speedup |
| Training memory | 4GB | <3GB | 25% reduction |
| INT8 accuracy loss | - | <2% | ✓ Required |
| Power efficiency | - | <180W @ 40K tok/s | ✓ Required |
| Best hyperparams found | - | Yes | ✓ Required |

**Deliverables**:
- `results/m4/best_config.yaml`
- `results/m4/quantization_results.json`
- `results/m4/efficiency_report.md`
- `results/m4/optimized_model.pt`

**Decision Point**: If efficiency targets not met → explore event-driven updates or early stopping. If INT8 accuracy loss >5% → use mixed quantization (weights INT8, activations FP16).

---

#### M5: Full Evaluation (~400 GPU hours)

**Objective**: Comprehensive benchmark suite vs all baselines

**Experiments**:
- Train full model family (3 seeds each):
  - Nano (10M): 30 hours
  - Small (125M): 80 hours
  - Base (360M): 150 hours
  - Large (1.4B): 140 hours (on A100)
- Baseline models:
  - Transformer, Mamba, S4, ANA-backprop: 30 hours
- Benchmark suite:
  - Synthetic benchmarks (all models): 20 hours
  - Language benchmarks (WikiText, Pile): 30 hours
  - Downstream benchmarks (MMLU, HellaSwag): 20 hours
- Statistical analysis:
  - Significance testing, effect sizes: 10 hours
  - Ablation studies (full sweep): 20 hours

**Success Criteria** (statistical validation):
| Category | Metric | Bio-ANA vs Transformer | Bio-ANA vs Mamba | Required |
|----------|--------|------------------------|------------------|----------|
| Synthetic | MQAR (64 pairs) | d > 0.8, p < 0.05 | d > 0.5, p < 0.05 | ✓ Required |
| Language | WikiText PPL | Within 5% | Within 5% | ✓ Required |
| Efficiency | Memory (8K ctx) | 10× less | 2× less | ✓ Required |
| Bio-fidelity | Noise tolerance | d > 0.8, p < 0.05 | N/A | ✓ Required |

**Deliverables**:
- `results/m5/model_family/*.pt`
- `results/m5/benchmarks_full.json`
- `results/m5/statistical_analysis.pdf`
- `results/m5/evaluation_report.md`

**Decision Point**: If statistical significance not achieved → increase seed count or sample size. If efficiency gains <3× → investigate bottlenecks or architectural simplifications.

---

#### M6: Edge Validation (~0 GPU hours, on-device)

**Objective**: Validate deployment on edge hardware

**Experiments**:
- Export to ONNX (CPU time)
- PyTorch Mobile conversion (CPU time)
- Test on Raspberry Pi 4 (hardware time)
- Test on Jetson Nano (hardware time)
- Power and latency profiling (hardware time)

**Success Criteria**:
| Platform | Metric | Target |
|----------|--------|--------|
| Raspberry Pi 4 | RAM usage | <2GB |
| Jetson Nano | Latency | <100ms |
| ONNX export | Compatibility | All ops supported |

**Deliverables**:
- `results/m6/onnx_models/`
- `results/m6/edge_report.md`

---

### Contingency Budget (350 hours)

Allocated for:
- Failed experiment retries: 100 hours
- Hyperparameter search expansion: 100 hours
- Additional ablation studies: 50 hours
- Model iteration/improvement: 100 hours

**Contingency Triggers**:
| Situation | Trigger | Action | Budget |
|-----------|---------|--------|--------|
| EqProp divergence | Convergence fails at M1 | Adjust hyperparameters | +20 hours |
| Accuracy gap | >10% gap at M3 | Curriculum tuning | +30 hours |
| Memory overflow | >8GB at M4 | Gradient checkpointing | +15 hours |
| Statistical failure | p > 0.05 at M5 | Increase seeds to 5 | +25 hours |

---

### Cost Estimation (Progressive)

| Milestone | GPU Hours | Platform | Cost | Cumulative Cost |
|-----------|-----------|----------|------|-----------------|
| M0 | 1 | RTX 3080 | $0.30 | $0.30 |
| M1 | 9 | RTX 3080 | $2.70 | $3.00 |
| M2 | 40 | RTX 3080 | $12.00 | $15.00 |
| M3 | 50 | RTX 3080 | $15.00 | $30.00 |
| M4 | 150 | RTX 3080 | $45.00 | $75.00 |
| M5 | 400 | A100 | $1,224.00 | $1,299.00 |
| M6 | 0 | Edge | $0.00 | $1,299.00 |
| Contingency | 350 | Mixed | $350.00 | $1,649.00 |
| **Total** | **1000** | | | **~$1,650** |

**Progressive Investment Justification**:
- **$3 after 10 hours**: Proof that EqProp can train basic tasks
- **$30 after 100 hours**: Proof that Bio-ANA integrates successfully and beats baselines on key metrics
- **$75 after 250 hours**: Optimized model meeting efficiency targets
- **$1,300 after 650 hours**: Fully validated model family with statistical significance

**Stop Conditions**:
- Stop at M0: If XOR fails → EqProp implementation issue, fix before proceeding
- Stop at M1: If accuracy gap >20% → Fundamental limitation, pivot to hybrid
- Stop at M3 (100 hours): If Stage 2 MQAR <70% → Architecture issue, redesign tracks/HoloLink
- Stop at M5 (650 hours): If no statistical significance → Report findings, document limitations

---

## Risk Mitigation

| Risk | Likelihood | Impact | Probability | Mitigation Strategy |
|------|------------|--------|-------------|---------------------|
| EqProp instability on SSMs | Medium | Critical | 40% | Spectral normalization; fallback to hybrid training |
| Slow convergence | High | Medium | 70% | Early stopping; adaptive iterations; momentum acceleration |
| Accuracy gap vs backprop | Medium | Critical | 30% | Curriculum learning; architectural tuning; hyperparameter search |
| Memory overhead | Low | Medium | 20% | Gradient checkpointing; mixed precision; sparsity |
| CUDA kernel complexity | Medium | Low | 25% | Use PyTorch native ops first; defer to optional |
| Dataset licensing | Low | Low | 10% | Verify licenses; use permissive datasets only |
| Edge deployment failures | Medium | Medium | 35% | Test on emulator first; provide CPU fallback |

**Fallback Strategy**:
1. **Hybrid Training**: EqProp for first N layers, backprop for remainder
2. **Approximate EqProp**: Fixed number of iterations vs convergence-based
3. **Architecture Simplification**: Remove tracks, reduce depth
4. **Cloud Bursting**: Scale to A100 for final validation

---

## Technical Specifications

### Model Configurations

```python
# ana/bio_config.py
@dataclass
class BioANAConfig:
    # Model size
    variant: str = "small"  # nano, small, base, large
    d_model: int = 512
    num_layers: int = 4
    
    # Track configuration
    syntax_dim: int = 64
    semantic_dim: int = 128
    logic_dim: int = 64
    num_tracks: int = 3
    
    # Stack
    stack_depth: int = 5
    stack_dim: int = 64
    
    # HoloLink
    hololink_key_dim: int = 128
    hololink_capacity: int = 1000  # Max KV pairs
    
    # HyperController
    cortex_hidden_dim: int = 128
    cortex_layers: int = 2
    
    # EqProp parameters
    relaxation_iterations: int = 20
    nudge_strength: float = 0.1
    learning_rate: float = 1e-3
    spectral_radius: float = 0.99
    
    # Regularization
    sparsity: float = 0.1
    dale_constraint: bool = True
    noise_injection: float = 0.05
    
    # Training
    batch_size: int = 16
    epochs_per_stage: int = 30
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    
    # Quantization
    quantize_weights: bool = False
    quantize_activations: bool = False
    ternary_threshold: float = 0.7
```

### Hyperparameter Search Space

| Parameter | Range | Type | Prior |
|-----------|-------|------|-------|
| relaxation_iterations | [5, 50] | int | 20 |
| nudge_strength | [0.05, 0.5] | float | 0.1 |
| learning_rate | [1e-4, 1e-2] | log | 1e-3 |
| spectral_radius | [0.90, 0.999] | float | 0.99 |
| sparsity | [0.01, 0.4] | float | 0.1 |
| syntax_dim | [32, 128] | int | 64 |
| semantic_dim | [64, 256] | int | 128 |

---

## File Structure (Planned)

```
ana/
├── eqprop/
│   ├── energy.py          # Energy function base class
│   ├── relaxation.py      # Free/nudged phase
│   ├── update.py          # Local update rules
│   ├── constraints.py     # Dale's law, spectral norm
│   └── __init__.py
├── bio_ana/
│   ├── __init__.py
│   ├── model.py           # Bio-ANA main model
│   ├── tracks.py          # Track-specific energy functions
│   ├── hololink.py        # HoloLink with Hebbian updates
│   └── controller.py      # HyperController
├── bio_training/
│   ├── __init__.py
│   ├── trainer.py         # Main training loop
│   ├── curriculum.py      # Stage management
│   ├── scheduler.py       # Relaxation/LR scheduling
│   └── data.py            # Data pipeline
├── quantization/
│   ├── __init__.py
│   ├── static.py          # Static quantization
│   └── dynamic.py         # Dynamic quantization
├── profiling.py
└── bio_config.py

experiments/
├── exp_bio_benchmarks.py  # Main benchmark runner
├── exp_ablations.py       # Ablation sweep
├── exp_efficiency.py      # Efficiency benchmarks
└── analysis.py            # Statistical analysis

deploy/
├── export_onnx.py
├── mobile/
│   ├── ios.py
│   └── android.py
└── fpga/
    └── synthesis.py

demos/
├── text_generation.py
├── edge_demo.py
└── interactive.py

tools/
├── visualize_energy.py
└── plot_benchmarks.py

tests/
├── test_eqprop.py
├── test_bio_ana.py
└── test_training.py

results/
├── benchmarks/
│   ├── synthetic/
│   ├── language/
│   └── downstream/
├── ablations/
└── efficiency/

run_bio_experiment.py       # CLI interface (extends run_experiment.py)
PLAN.md                     # This file
README.md                   # Project README
AGENTS.md                   # Agent instructions
```

---

## Success Metrics Summary

| Tier | Category | Metric | Minimum | Target | Stretch |
|------|----------|--------|---------|--------|---------|
| **Proof** | Synthetic | AR accuracy | 95% | 98% | 99% |
| | | MQAR (64 pairs) | 80% | 90% | 95% |
| | EqProp | Energy monotonicity | 90% | 95% | 99% |
| | | Convergence < 50 iters | 70% | 90% | 100% |
| **Validation** | Language | WikiText PPL | <35 | <32 | <30 |
| | | Pile PPL | <11.5 | <10.5 | <9.5 |
| | Efficiency | Training speed | 0.3× | 0.8× | 1.2× backprop |
| | | Memory savings | 3× | 10× | 20× |
| **Production** | Deployment | Edge functional | Yes | Yes | Yes |
| | | INT8 accuracy loss | <5% | <2% | <1% |
| | | Power <5W | Yes | Yes | Yes |
| **Bio-Fidelity** | Robustness | Noise tolerance | +2% | +5% | +10% |
| | | Continual learning | <15% | <10% | <5% |

---

## Next Actions (Priority Order)

| Priority | Action | Owner | Dependencies |
|----------|--------|-------|--------------|
| 1 | Implement `ana/eqprop/energy.py` base class | Developer | None |
| 2 | Implement `ana/eqprop/relaxation.py` with early stopping | Developer | #1 |
| 3 | Validate XOR convergence in `tests/test_eqprop.py` | Developer | #2 |
| 4 | Implement `ana/bio_ana/tracks.py` with track-specific energy | Developer | #3 |
| 5 | Integrate with existing `model_v3.py` architecture | Developer | #4 |
| 6 | Implement Stage 0 curriculum on synthetic AR tasks | Developer | #5 |
| 7 | Run baseline benchmarks for comparison | Developer | #6 |

---

## References

- **Equilibrium Propagation**: Scellier & Bengio, "Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation", NeurIPS 2017
- **Selective SSMs**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- **Structured SSMs**: Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", NeurIPS 2021
- **Bio-plausible Learning**: Sacramento et al., "Deep Learning with Event-Driven Backpropagation", 2018
- **Holographic Memory**: Plate, "Holographic Reduced Representations", IEEE Trans. Neural Networks 1995
- **Oja's Rule**: Oja, "Simplified neuron model as a principal component analyzer", JMLR 1982
- **Dale's Law**: Dale, "The significance of the suprarenal capsules", 1901

---

## Appendix: Integration with Existing Codebase

### Compatible Files

| File | Purpose | Integration Point |
|------|---------|-------------------|
| `ana/model_v3.py` | Existing ANA model | Extend with EqProp methods |
| `ana/training_v2.py` | Existing training | Adapt for EqProp curriculum |
| `ana/config_v2.py` | Configuration | Extend with BioANAConfig |
| `ana/benchmarks.py` | Benchmark suite | Use directly |
| `ana/data.py` | Datasets | Extend for data pipeline |
| `tests/conftest.py` | Test fixtures | Use seed/seed fixtures |

### Extension Pattern

```python
# Example: extending model_v3.py
from ana.eqprop import EnergyFunction, RelaxationEngine

class BioANAModel(ANAv2Model):
    def __init__(self, config: BioANAConfig):
        super().__init__(config)
        self.relaxation = RelaxationEngine(config)
        self.track_energy = {
            'syntax': SSMEnergyFunction('syntax', config.syntax_dim),
            'semantic': SSMEnergyFunction('semantic', config.semantic_dim),
            'logic': SSMEnergyFunction('logic', config.logic_dim)
        }
    
    def forward(self, input_ids, return_info=False):
        # Add EqProp relaxation to existing forward pass
        pass
```

---

**Document Version**: 2.0
**Last Updated**: 2026-02-10
**Status**: Ready for Implementation
