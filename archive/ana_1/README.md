# ANA-1 Production Implementation

This directory contains the production-ready implementation of the **Adaptive Neural Automaton (ANA)** architecture, validated through the PoC experiments in `/ana/`.

## Architecture Overview

ANA combines three key innovations:
1. **Dual-Track State Space Model**: Separate "Reflex" (fast decay) and "Reasoning" (slow decay) tracks
2. **Holo-Link Associative Memory**: Learned key-value memory with normalized projections
3. **Hyper-Controller**: Meta-network that dynamically modulates SSM gates based on input

## Available Configurations

### Micro (1M params) - `python3 train.py micro`
**Purpose:** GPU smoke testing on shared resources
- Layers: 2
- Model Dim: 128
- Batch Size: 4
- Seq Length: 128
- Steps: 20
- Memory: ~500MB GPU

### Mini (10M params) - `python3 train.py mini`
**Purpose:** Rapid validation and ablation studies
- Layers: 6
- Model Dim: 256
- Batch Size: 2
- Seq Length: 128
- Steps: 50
- Memory: ~2GB GPU

### Small (125M params) - `python3 train.py`
**Purpose:** Production training (Pythia-125M equivalent)
- Layers: 12
- Model Dim: 768
- Batch Size: 4
- Seq Length: 1024
- Dataset: SlimPajama-627B (streaming)
- Target: Match Pythia-125M perplexity

## Quick Start

```bash
# Smoke test (fastest, GPU-friendly)
python3 train.py micro

# Validation run (CPU or small GPU)
python3 train.py mini

# Full training (requires dedicated GPU + SlimPajama dataset)
python3 train.py
```

## Key Files

- `config.py`: Model configurations (Micro, Mini, Small)
- `model/layers.py`: Core architecture (DualTrackBlock, HoloLink, HyperController)
- `model/modeling_ana.py`: Main model class with Multi-Objective Loss
- `data.py`: Data loading (Wikitext-2, SlimPajama)
- `train.py`: Training script

## Multi-Objective Loss

The model optimizes:
- **L_LM**: Standard next-token prediction
- **L_Sparsity**: L1 penalty on retrieval gate (encourages selective memory use)

Future versions will add **L_Retrieval** (auxiliary reconstruction task).

## Results from PoC

Phase 1-3 validation demonstrated:
- **13% improvement** over static baseline on mixed-mode tasks
- **Track specialization** confirmed (Reflex α~0.5, Reasoning α~0.95)
- **Holo-Link convergence** achieved with weighted loss strategy

See `/ana/walkthrough.md` for detailed results.

## Citation

If you use this code, please reference the ANA architecture specification:
```
Adaptive Neural Automaton (ANA): A Dual-Track State Space Model with Dynamic Gating
```
