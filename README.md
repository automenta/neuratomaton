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

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

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

## Experiments

### Reproduce Results

```bash
# Run comprehensive experiments
python -m ana.experiments.comprehensive

# Run associative recall benchmark
python -m ana.experiments.associative_recall

# Run parameter efficiency study
python -m ana.experiments.parameter_efficiency
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