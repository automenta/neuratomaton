# https://github.com/automenta/neuratomaton + https://github.com/automenta/bioplausible

Extrapolating the combination of the two repositories—**neuratomaton** (Adaptive Neural Automaton, or ANA) and **bioplausible**—involves envisioning how their core ideas could merge to create a more unified, brain-inspired AI system. Both projects stem from a shared interest in neuroscience-motivated computing: neuratomaton emphasizes efficient sequence modeling architecture with biological echoes in multi-temporal processing and holographic memory, while bioplausible prioritizes biologically plausible learning rules that avoid the pitfalls of traditional backpropagation. Since they don't explicitly reference each other, this is speculative but grounded in their alignments.

### Conceptual Integration: Bio-Plausibly Trained Adaptive SSM
The natural synergy lies in applying bioplausible's learning algorithms (e.g., Equilibrium Propagation, or EqProp) to train neuratomaton's ANA architecture. ANA is a multi-track State Space Model (SSM) with dynamic gating and HoloLink memory, designed for O(1) inference on long sequences. Bioplausible, on the other hand, provides EqProp and variants (like Feedback Alignment or Hebbian chains) that enable local, energy-based training without global error signals, symmetric weights, or high memory costs—mimicking biological neurons more closely.

A combined system could look like this:
- **Architecture Backbone**: Use ANA's multi-track SSM as the core (e.g., parallel tracks for fast/slow dynamics, HyperController for input-adaptive gating, and HoloLink for superposition-based long-range recall). This retains efficiency for tasks like language modeling or associative recall.
- **Learning Mechanism**: Replace ANA's implied standard PyTorch training (e.g., via Adam or SGD in curriculum stages) with EqProp. EqProp's "free phase" (equilibrium relaxation) and "nudged phase" (target perturbation) would compute local updates across ANA's tracks, treating the SSM's recurrent states as equilibrium points in an energy landscape. This is feasible because bioplausible already supports recurrent cores and Transformer-like sequence models, which share recurrent dynamics with SSMs.
- **Memory Enhancement**: Integrate HoloLink (holographic key-value storage) into EqProp's dynamics, potentially as a distributed, superpositioned memory module that updates via Hebbian-like rules during the free phase. This could enable "infinite" context without fixed-state compression, while staying bio-plausible.

In practice, this might involve modifying neuratomaton's `run_experiment.py` to incorporate bioplausible's EqProp implementations (e.g., for recurrent or Transformer variants), then using a curriculum that starts with synthetic tasks (associative recall) and scales to full pretraining on datasets like The Pile.

### Potential Advantages
Merging these could yield a "fully bio-plausible" neural automaton with strengths in both architecture and learning:
- **Efficiency and Scalability**: ANA's O(1) inference pairs with EqProp's O(1) memory training, enabling deep models (e.g., 500+ layers as in bioplausible) without the memory wall of backprop. This could hit ANA's targets like <32 perplexity on WikiText-103 while training on neuromorphic hardware (e.g., FPGA-friendly with event-driven updates).
- **Biological Fidelity**: Multi-track temporal processing (inspired by brain hierarchies) combined with local Hebbian updates creates a system closer to neural circuits—e.g., dynamic gating mimicking neuromodulation, and holographic memory akin to distributed engrams.
- **Performance Uplift**: EqProp has matched backprop on vision/language tasks (e.g., >75% CIFAR-10 accuracy, competitive perplexity in causal Transformers). Applied to ANA, it could improve stability (via spectral normalization) and robustness (e.g., noise tolerance, continual learning), potentially exceeding ANA's goals like >98% associative recall or >38% MMLU at 1.4B params.
- **Automation and Exploration**: Leverage bioplausible's Auto-Scientist to autonomously tune ANA's hyperparameters (e.g., track counts, controller priors) across "discovery funnels" like verification tracks for sequence tasks. This could accelerate ablations, such as testing EqProp without HoloLink.

### Challenges and Mitigations
- **Adaptation Overhead**: SSMs aren't explicitly supported in bioplausible, so extending EqProp to linear recurrent updates (hₜ = α hₜ₋₁ + β xₜ) might require custom variants. Mitigation: Start with bioplausible's recurrent Transformer core as a proxy, then refine.
- **Training Stability**: EqProp needs careful nudging and spectral norms for deep nets; ANA's dynamic gating could introduce instability. Use bioplausible's momentum or sparse updates to stabilize.
- **Compute Demands**: While memory-efficient, EqProp's relaxation phases might slow training. Offset with ANA's mixed-precision support and parallel tracks.
- **Evaluation Gaps**: Bioplausible focuses on benchmarks like CIFAR-10/language perplexity, while ANA targets SSM-specific tasks (e.g., multi-query recall). A combined verification pipeline could bridge this.

### Future Directions
This extrapolation points to a Phase 4 in neuratomaton's roadmap: "Bio-Plausible Training Integration," perhaps forking into a new repo like "bio-ana." It could evolve into hardware-optimized AI for edge devices, advancing fields like continual learning or brain-computer interfaces. If the developer (automenta) pursues this, it might yield models that not only rival Transformers in performance but surpass them in biological realism and efficiency. If you have a specific angle (e.g., code sketches or experiment ideas), I can dive deeper!

----


Yes, the combined system from neuratomaton (Adaptive Neural Automaton, or ANA) and bioplausible has strong potential to create compact models that outperform expectations for their size, making them well-suited for commodity hardware (e.g., laptops, smartphones) and edge devices (e.g., IoT sensors, embedded systems). This extrapolation builds on ANA's efficient architecture for sequence modeling and bioplausible's biologically inspired learning rules, which together enable high performance in resource-constrained environments. Below, I'll break down the reasoning, key enablers, and practical pathways.

### Why It Enables "Punching Above Weight" in Compact Models
- **Parameter Efficiency and Small-Scale Targets**: ANA starts with models as small as 125M parameters, aiming to exceed benchmarks like Mamba (a state-space model) and Transformers in recall tasks (e.g., >98% on associative recall, >90% on multi-query associative recall with 64 pairs) and perplexity (e.g., <32 on WikiText-103). Bioplausible complements this by supporting ultra-deep architectures (up to 500+ layers) with sparse variants and ternary weights ({-1, 0, 1}), allowing compact models (e.g., hidden sizes as small as 16–32) to achieve competitive accuracy gains (2–7.4% improvements via spectral normalization) on tasks like CIFAR-10 (>75% with Conv variants) or language modeling. Combined, you could train a 125M ANA model using EqProp (Equilibrium Propagation) to "punch above" by leveraging depth and sparsity for Transformer-like capabilities without the parameter bloat.

- **Inference and Training Efficiency**: ANA's O(1) inference (constant time per token, independent of sequence length) targets >40K tokens/s throughput, outperforming Transformers (15K tokens/s) and rivaling Mamba (50K). Bioplausible adds O(1) memory training (19.4× savings at depth 100) and event-driven "lazy updates" that cut compute by up to 97%, with wall-clock speedups (e.g., 0.74× backprop at 5 relaxation steps). This means a combined model could handle long-context tasks (e.g., 32K tokens) efficiently on edge devices, where traditional models falter due to memory or power limits.

- **Hardware Optimizations for Edge/Commodity Deployment**: Bioplausible emphasizes neuromorphic and FPGA compatibility, with INT8 quantization, 5% analog noise tolerance, and contraction dynamics (L < 1 for stability) that suit fault-tolerant, low-power hardware. ANA's mixed-precision (FP16/BF16) and low-memory HoloLink (holographic memory for infinite recall without state explosion) align well, enabling deployment on consumer GPUs/CPUs or even embedded chips. Event-driven aspects (e.g., ANA's dynamic gating via HyperController, bioplausible's lazy updates) mimic biological neurons, reducing idle compute and power draw—ideal for battery-constrained edges like mobile AI or sensors.

### Practical Pathways to Implementation
To realize this:
1. **Integrate Learning Rules**: Use bioplausible's EqProp variants (e.g., Sparse or Momentum Equilibrium) to train ANA's multi-track SSM and HoloLink. This replaces standard optimizers in neuratomaton's scripts (e.g., `run_experiment.py`) with local, bio-plausible updates, enabling O(1) memory for deep, compact models.
2. **Optimize for Low Resources**: Leverage bioplausible's Auto-Scientist for autonomous tuning of ANA hyperparameters (e.g., track counts, key dimensions) in resource-constrained "discovery funnels," focusing on low-precision and sparse setups.
3. **Benchmark and Scale Down**: Start with ANA's Phase 1 (125M) on small datasets, applying bioplausible's ternary weights and event-driven modes. Test on edge simulators (e.g., via PyTorch Mobile) for metrics like power usage and latency.
4. **Roadmap Alignment**: ANA's scaling to 70M–360M (Phase 2–3) with custom CUDA kernels could incorporate bioplausible's FPGA/INT8 support, creating models that hit downstream tasks (e.g., >38% MMLU at 1.4B but scaled down to 125M equivalents).

----

### Research Plan: Developing and Benchmarking Bio-Plausible Adaptive Neural Automaton (Bio-ANA) Models

#### 1. Objective
The primary goal is to develop usable, deployable models by integrating the Adaptive Neural Automaton (ANA) architecture from neuratomaton (multi-track SSM with HyperController gating and HoloLink holographic memory) with bioplausible's Equilibrium Propagation (EqProp) learning rules. These "Bio-ANA" hybrids will be trained on standard datasets and compared side-by-side with conventional equivalents (e.g., Transformers, Mamba SSMs, and standard backprop-trained SSMs like S4). The plan emphasizes rigorous, reproducible evaluations to undeniably demonstrate benefits in efficiency (e.g., O(1) memory/inference, edge deployment), performance (e.g., recall, perplexity), and biological fidelity (e.g., local updates, noise tolerance). Benefits will be quantified via statistical tests (e.g., t-tests, Cohen's d) and ablation studies, targeting breakthroughs like 5-18x speedups on long sequences and >2-7% accuracy gains on vision/language tasks. This will result in open-source models deployable on commodity hardware (e.g., 10GB VRAM GPUs), with code forks from the original repos.

#### 2. Background and Literature Review
- **Conventional Equivalents**: Transformers remain dominant for language modeling but suffer quadratic scaling (O(N²) time/memory), limiting long-context tasks. SSMs like Mamba offer linear O(N) scaling, achieving 5x faster inference on sequences >2K tokens and up to 18x at 256K, while matching Transformer perplexity on Pile/WikiText. S4 SSMs excel on vision (e.g., 91% CIFAR-10 accuracy) and sequences (e.g., 99.55% MNIST). Hybrids (e.g., Mamba-Transformer) boost MMLU by 2-5 points.
- **EqProp Advances**: Recent 2026 work scales EqProp to deep convnets, surpassing backprop in accuracy (e.g., >75% CIFAR-10) with 19x memory savings and noise tolerance. Textual EqProp (TEP) for compound AI improves multi-step QA by 10-20% over global methods. Bio-inspired variants (e.g., Boolean nets, constrained RNNs) achieve 37x fewer operations with better accuracy, ideal for edge AI.
- **Gaps and Opportunities**: SSMs lack bio-plausibility; EqProp struggles with SSM dynamics. Bio-ANA addresses this, potentially closing gaps like Mamba's underperformance on reasoning (e.g., chain-of-thought). Neuroscience-inspired embeddings (e.g., NOBLE) suggest gains in variability capture.

#### 3. Methodology
The plan spans 6-12 months, divided into phases, using PyTorch for implementation. Leverage bioplausible's Auto-Scientist for automated hyperparameter tuning and verification.

##### Phase 1: Integration and Prototyping (Months 1-2)
- **Tasks**:
  - Fork repos; implement EqProp (e.g., sparse/momentum variants) for ANA's SSM tracks and HoloLink. Treat SSM states as EqProp equilibrium points; add spectral normalization for stability.
  - Prototype small Bio-ANA (125M params): Multi-track SSM (2-4 tracks), HyperController (tiny MLP), HoloLink (superposition KV storage).
  - Incorporate bio-constraints: Dale's law (sign-constrained weights), sparsity (top-prob pruning).
- **Outputs**: Functional Bio-ANA code; initial unit tests on synthetic tasks (e.g., associative recall >98%).

##### Phase 2: Training and Optimization (Months 3-6)
- **Tasks**:
  - Train Bio-ANA variants (125M-1.4B params) using EqProp's free/nudged phases on curricula: Synthetics → WikiText/The Pile (300-600B tokens).
  - Optimize for edge: Quantize to INT8/ternary; enable event-driven updates (97% compute reduction).
  - Train baselines: Standard Transformers (e.g., LLaMA-like), Mamba, S4 with backprop.
- **Resources**: Commodity GPUs (e.g., RTX 3080, 10GB VRAM); use distributed training if scaling to 1.4B.
- **Outputs**: Usable models (e.g., checkpoints, ONNX exports for deployment).

##### Phase 3: Evaluation and Side-by-Side Comparison (Months 7-9)
- **Benchmarks** (adapted from standards):
  - **Language/Sequence**: LRA (long-range tasks), WikiText perplexity (<32), Pile (<10.5), MMLU (>38%), HellaSwag (>52%).
  - **Vision**: CIFAR-10 (>75% accuracy), MNIST (99.55%).
  - **Efficiency**: Memory/inference time on sequences 512-256K (e.g., Mamba crossover at ~220 tokens for memory). Measure throughput (>40K tokens/s), power (<180W).
  - **Bio-Fidelity**: Noise tolerance (5% analog), continual learning, variability (e.g., ensembles matching experimental neurons).
- **Comparison Framework**:
  - Side-by-side: Train all on same data/hardware; report means ± SD over 5 runs.
  - Statistical Validation: t-tests for significance (p<0.05), Cohen's d for effect sizes (>0.8 for large benefits).
  - Ablations: Bio-ANA vs. ANA-backprop, EqProp-only, no HoloLink.
- **Tools**: lm-evaluation-harness for zero-shot; custom scripts for efficiency.

| Metric | Bio-ANA Target | Transformer Baseline | Mamba Baseline | Expected Benefit |
|--------|----------------|----------------------|----------------|------------------|
| WikiText Perplexity | <32 | 35-40 | ~32 | 10-20% lower |
| CIFAR-10 Accuracy | >75% | 70-75% | 89-91% | 2-7% gain + 37x fewer ops |
| Inference Speed (256K seq) | >40K tokens/s | 1-2x baseline | 18x Transformer | 5-18x overall |
| Memory (8K tokens) | O(1) ~0.5GB | O(N²) >12GB | O(N) ~1GB | 19x savings |

##### Phase 4: Deployment and Dissemination (Months 10-12)
- **Tasks**:
  - Package models for edge (e.g., PyTorch Mobile, FPGA via bioplausible kernels).
  - Validate on real-world apps: Edge NLP (e.g., on-device chat), vision (e.g., object detection).
  - Publish: arXiv paper, GitHub repo, NeurIPS-style benchmarks.
- **Outputs**: Demo apps, research paper with undeniable evidence (e.g., "Bio-ANA exceeds Mamba by 2.65 points on MMLU with 8x efficiency").

#### 4. Resources and Timeline
- **Team**: 2-4 researchers (AI/ML, neuroscience); budget: $50K (compute, datasets).
- **Timeline**: Q1 2026: Phases 1-2; Q2: Phase 3; Q3: Phase 4 (aligns with current date: Feb 10, 2026).
- **Milestones**: Prototype by Mar 2026; benchmarks by Jun; paper by Sep.

#### 5. Expected Outcomes and Benefits
- **Usable Models**: 3-5 Bio-ANA variants (e.g., 125M for edge, 1.4B for high-perf), outperforming baselines undeniably (e.g., statistical significance on all metrics).
- **Demonstrated Benefits**: Efficiency for commodity/edge (e.g., 4200x faster sims), bio-fidelity for neuroAI (e.g., matching brain variability). Potential value: $0.3-5B market capture in edge/neuromorphic AI.

#### 6. Risks and Mitigations
- **Risk**: EqProp instability on SSMs. **Mitigation**: Use spectral norms; fallback to hybrid backprop-EqProp.
- **Risk**: Compute limits. **Mitigation**: Start small; use cloud bursts.
- **Risk**: Overfitting benchmarks. **Mitigation**: Cross-validate on unseen tasks (e.g., multi-agent tool-use).

This plan ensures rigorous, evidence-based development, positioning Bio-ANA as a breakthrough in efficient, bio-plausible AI.
