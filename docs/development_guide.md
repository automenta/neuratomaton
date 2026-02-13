# ANA: Adaptive Neural Automaton - Industrial/Academic Research Platform

## Project Structure

```
ana/
├── src/
│   ├── ana/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── config.py      # Configuration classes
│   │   │   └── core.py        # Core model implementations
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   └── utils.py       # Training utilities and trainers
│   │   ├── experiments/
│   │   │   ├── __init__.py
│   │   │   └── main.py        # Experiment runners
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── datasets.py    # Dataset utilities
│   ├── setup.py
│   └── pyproject.toml
├── tests/
│   └── test_models.py         # Unit tests
├── results/                   # Experiment results
├── data/                      # Datasets
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── README.md                  # Main documentation
├── run_experiments.py         # Entry point for experiments
└── .gitignore
```

## Key Features

### 1. Clean Architecture
- Well-organized package structure
- Separation of concerns (models, training, experiments, utils)
- Type hints and comprehensive docstrings
- Proper error handling

### 2. Core Components
- **ANAConfig**: Centralized configuration management
- **ANAModel**: Main model with HoloLink associative memory
- **BaselineSSM**: Baseline for comparisons
- **HoloLink**: Associative memory module
- **LinearRecurrentUnit**: Core SSM component

### 3. Training Utilities
- **Trainer**: Generic training loop
- **TwoPhaseTrainer**: Specialized trainer for two-phase methodology
- Support for masked datasets
- Gradient clipping and stability features

### 4. Experiment Framework
- Systematic comparison between ANA and baseline
- Associative recall evaluation
- Two-phase training validation
- Parameter efficiency studies

## Getting Started

### Installation
```bash
pip install -e .
```

### Running Experiments
```bash
python run_experiments.py
```

### Running Tests
```bash
python -m pytest tests/ -v
```

## Key Research Validations

### 1. Two-Phase Training
- Phase 1: Train HoloLink memory (freeze controller)
- Phase 2: Fine-tune controller (freeze HoloLink)
- Solves gradient interference problem

### 2. Associative Memory
- HoloLink provides explicit key-value storage
- O(N) complexity for training and inference
- Significant improvements on memory-intensive tasks

### 3. Parameter Efficiency
- ANA achieves better performance with fewer parameters
- Validated through systematic comparisons
- Particularly effective for associative recall tasks

## Development Guidelines

### Code Standards
- Follow PEP 8 style guide
- Include type hints
- Write comprehensive docstrings
- Add unit tests for new functionality

### Experiment Design
- Use controlled experimental conditions
- Match parameter counts for fair comparisons
- Document all hyperparameters
- Save results for reproducibility

## Publications & Citations

This codebase supports the following research:

1. "ANA: Adaptive Neural Automaton with HoloLink Associative Memory"
2. "Two-Phase Training for Modular Neural Architectures" 
3. "Parameter-Efficient State Space Models with Associative Memory"

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.