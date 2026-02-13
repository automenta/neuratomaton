# ANA Research Infrastructure

This document outlines the infrastructure designed to scientifically reveal the potential of the Adaptive Neural Automaton (ANA) architecture. The process is designed to be **maximally automated** and provide **quick, frequent rich feedback**.

## Tools Overview

### 1. Automated Research Agent (`run_auto_research.py`)
The primary entry point for scientific discovery. It executes a staged research pipeline with built-in contingencies.

**Usage:**
```bash
python run_auto_research.py
python run_auto_research.py --quick  # For smoketesting / CI
```

**Stages:**
- **Stage 1: Sanity Check (Fail Fast).** Runs a small Associative Recall task. If accuracy < 90%, the pipeline aborts.
- **Stage 2: Scaling Probe (Probe Trend).** Runs a scaling benchmark on $N \in \{128, 512\}$. Checks if ANA outperforms the Baseline. If trend is negative, massive scaling is skipped.
- **Stage 3: Deep Dive.** Runs full benchmarks:
    - **Scaling:** Up to $N=4096$.
    - **Ablation:** Tests component contributions (HoloLink, Controller).
    - **Throughput:** Measures efficiency.

### 2. Fast Breakthrough Demo (`src/ana/experiments/demo.py`)
Designed for **immediate feedback**. Use this to watch the model "learn" in real-time.

**Usage:**
```bash
python src/ana/experiments/demo.py --steps 500
```
- Updates plots in `results/demo/latest/` every 50 steps.
- Prints "Epiphany" messages when accuracy thresholds are crossed.
- Visualizes internal dynamics (Retrieval Gates, Mixing Weights).

### 3. Comprehensive Manual Runner (`src/ana/experiments/run_comprehensive.py`)
For manual execution of specific benchmarks.

**Usage:**
```bash
python src/ana/experiments/run_comprehensive.py --task scaling
python src/ana/experiments/run_comprehensive.py --task ablation
```

## Visualization & Analysis
Internal model states are captured and visualized in `src/ana/utils/plotting.py`.
- **Track Mixing:** Heatmap of how the model mixes between Fast (Reflex) and Slow (Reasoning) tracks over time.
- **Gating Dynamics:** Line plots of Retrieval Gate (Memory Access) and Halt Gate (Thinking Steps).

## Directory Structure
- `src/ana/experiments/automated_researcher.py`: Logic for staged execution.
- `src/ana/experiments/comprehensive.py`: Core `ComparisonRunner` implementation.
- `src/ana/models/core.py`: Instrumented ANA model.
- `results/automated/`: Output directory for automated runs (timestamped).

## Research Philosophy
1.  **Fail Fast:** Don't waste compute on broken models.
2.  **Probe First:** Verify trends on small N before committing to large N.
3.  **Rich Feedback:** Always visualize internal dynamics to understand *why* it works (or fails).
