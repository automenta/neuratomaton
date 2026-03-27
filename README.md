# ANA: Adaptive Neural Automaton

**State-of-the-art State Space Model with Associative Memory**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

ANA (Adaptive Neural Automaton) is an advanced State Space Model architecture featuring HoloLink associative memory. This architecture demonstrates significant improvements over traditional Transformers in both parameter efficiency and performance on memory-intensive tasks.

### Key Features

- **HoloLink Associative Memory**: Differentiable memory system for key-value storage and retrieval
- **Multi-Track Processing**: Parallel processing tracks for different temporal scales
- **Linear Complexity**: O(N) complexity for both training and inference
- **Two-Phase Training**: Methodology to avoid gradient interference between components
- **Parameter Efficiency**: Superior performance with fewer parameters than Transformers

### Breakthrough Results

- **60-69% better perplexity** than Transformers with equal or more parameters
- **13x improvement** on associative recall tasks compared to Transformers
- **O(N) complexity** enabling efficient long-context modeling
- **Parameter efficiency** allowing smaller models with better quality

## Installation

```bash
pip install -e .
```

Or for development (including test dependencies):

```bash
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
import torch
from ana import ANAConfig, ANAModel

# Create configuration
config = ANAConfig(
    vocab_size=50257,
    d_model=512,
    state_dim=512,
    key_dim=256,
    num_layers=4,
    use_hololink=True,
    use_controller=False,
    use_parallel_scan=True,
)

# Create model
model = ANAModel(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 128))
logits, _ = model(input_ids)
```

### CLI Usage

The package provides a turnkey CLI `ana-research` to run experiments and verify the architecture.

```bash
# Verify installation with a quick smoketest
ana-research --quick --discovery

# Run full Phase 3 (Discovery) experiments
ana-research --discovery

# Run all phases (Validation, Potential, Discovery, Action, Series)
ana-research --all
```

Run `ana-research --help` for all available options.

### Usage Examples

For ready-to-run scripts, check the `examples/` directory:

- **Text Generation**: `python examples/text_generation.py` - Simulates basic text generation.
- **Reinforcement Learning**: `python examples/rl_agent.py` - Simulates an RL agent step.

## Architecture

### HoloLink Associative Memory

```
Memory Operations:
  Store:    M[t] = M[t-1] + k[t] ⊗ v[t]    (key-value outer product)
  Retrieve: r[t] = M[t] @ q[t]             (direct matrix lookup)

Properties:
  - Explicit key-value storage (not learned embeddings)
  - O(N) parallel scan for efficient training
  - Differentiable end-to-end
```

### ANA Model Components

| Component | Purpose | Complexity |
|-----------|---------|------------|
| Linear Recurrent Units | Sequence modeling | O(N) |
| HoloLink Memory | Associative recall | O(N) |
| Parallel Scan | Efficient training | O(N) |

**Total complexity: O(N) - Linear in sequence length**

## Advanced Usage

ANA is designed to be a general-purpose sequence model, not just for text.

### Reinforcement Learning (RL)

The `ANARLAgent` wrapper adapts the core model for decision-making tasks.

```python
from ana import ANARLAgent

# Config for 4 discrete actions, 10 continuous observation features
config = ANAConfig(action_space=4, observation_space=10, d_model=64)
agent = ANARLAgent(config)

obs = torch.randn(1, 10) # [Batch, Obs]
logits, value, next_state, info = agent(obs)
```

### Time Series & Audio

The `ANASeriesModel` handles continuous input/output streams.

```python
from ana import ANASeriesModel

# Config for univariate time series
config = ANAConfig(series_dim=1, d_model=64)
model = ANASeriesModel(config)

x = torch.randn(1, 100, 1) # [Batch, Seq, Dim]
pred, _ = model.forward_sequence(x)
```

## Experiments & Benchmarks

### Automated Research Pipeline

The `ana-research` tool automates the entire research pipeline, from validation to scientific discovery.

```bash
ana-research --study_name my_study --discovery
```

### Standalone Benchmarks

For specific technical benchmarks (scaling, ablation, throughput), you can use the comprehensive benchmark suite:

```bash
# Run comprehensive benchmarks (Scaling, Ablation, Throughput)
python -m ana.experiments.run_comprehensive

# Run specific experiments using the main CLI
ana-research --potential
```

### Two-Phase Training

The breakthrough two-phase training methodology:

1. **Phase 1**: Train HoloLink memory component (freeze controller)
2. **Phase 2**: Fine-tune controller (freeze HoloLink)

This approach solves the gradient interference problem that degrades performance when components are trained jointly.

## Research Papers

- **"ANA: Adaptive Neural Automaton with HoloLink Associative Memory"** - Core architecture and results
- **"Two-Phase Training for Modular Neural Architectures"** - Training methodology
- **"Parameter-Efficient State Space Models with Associative Memory"** - Efficiency analysis

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{ana2026,
  title={ANA: Adaptive Neural Automaton with HoloLink Associative Memory},
  author={ANA Research Team},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```
