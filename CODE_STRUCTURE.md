# Ana Code Structure

This document describes the organization of the `ana/` module.

## Directory Overview

```
ana/
├── core/              # Core shared components (WIP - to be created)
├── bio_ana/           # Bio-plausible ANA (EqProp integration) ✅
├── eqprop/            # bioplausible library (cloned) ✅
├── models.py          # ANA v1 (original) - DEPRECATED
├── models_v2.py       # ANA v2 (external memory) - DEPRECATED  
├── models_v3.py       # ANA v3 components (SpecializedTracks, etc.) ✅ ACTIVE
├── model_v3.py        # ANA v3 main model wrapper ✅ ACTIVE
├── config.py          # ANA v1 config - DEPRECATED
├── config_v2.py       # ANA v2/v3 config ✅ ACTIVE
├── training_v2.py     # Training utilities ✅ ACTIVE
├── model_factory.py   # Model factory for systematic exploration
├── model_space.py     # Architecture space definitions
├── benchmark.py       # Single-KV associative recall
├── benchmarks.py      # Full benchmark suite
├── data.py            # Dataset utilities
├── eval.py            # Evaluation tasks
└── analysis.py        # Result visualization
```

## Version History

### ANA v1 (DEPRECATED - `models.py`, `config.py`)
- Original implementation
- Basic multi-track SSM
- Simple HyperController
- Direct HoloLink binding
- **Status**: Replaced by v2/v3

### ANA v2 (DEPRECATED - `models_v2.py`)
- Enhanced with external memory bank
- Selective attention mechanism
- Query-gated routing
- **Status**: Replaced by v3

### ANA v3 (ACTIVE - `models_v3.py`, `model_v3.py`, `config_v2.py`)
Current production implementation with:
- **GumbelSoftmax**: Soft sampling for discrete operations
- **StackFrame**: Stack frame with opcodes
- **MetaStateStack**: Gumbel-controlled stack with bind/gate/shift/recurse
- **LinearRecurrentTrack**: Track with α, β modulation
- **FaultTraceBuffer**: Holographic fault memory
- **CortexController**: HyperController with α/β modulation
- **SpecializedTracks**: Syntax (τ=0.5), Semantic (τ=2.0), Logic (τ=1.0)
- **ANAInterpreter**: Executes opcodes

### Bio-ANA (ACTIVE - `bio_ana/`)
Integration with bioplausible EqProp:
- **BioSyntaxTrack/BioSemanticTrack/BioLogicTrack**: Track-specific energy functions
- **HoloLinkHebbian**: Oja's rule memory updates
- **BioANAModel**: Full model with energy tracking
- **BioANAConfig**: 4 variants (nano/small/base/large)

## Module Dependencies

### v3 Components
```
model_v3.py
├── config_v2.py (ANAv2Config)
└── models_v3.py
    ├── GumbelSoftmax
    ├── StackFrame
    ├── MetaStateStack
    ├── parallel_scan_hillis_steele_v2
    ├── LinearRecurrentTrack
    ├── FaultTraceBuffer
    ├── CortexController
    └── SpecializedTracks
```

### Bio-ANA
```
bio_ana/
├── config.py
│   └── config_v2.py (ANAv2Config - base class)
├── tracks.py
│   └── eqprop/bioplausible/models/eqprop_base.py (EqPropModel)
├── hololink.py
└── model.py
```

## Test Organization

```
tests/
├── test_models.py        # v1 model tests
├── test_models_v3.py     # v3 component tests (ACTIVE)
├── test_model_v3.py      # v3 integration tests (ACTIVE)
├── test_bio_ana.py       # Bio-ANA tests (ACTIVE)
├── test_eqprop.py        # EqProp tests (ACTIVE)
├── test_benchmarks.py    # Benchmark suite tests
├── test_training.py      # Training pipeline tests
├── test_data.py          # Dataset tests
└── test_ablation.py      # Ablation tests
```

## Component Mapping

| Component | v1 | v2 | v3 | Bio-ANA |
|-----------|----|----|----|---------|
| SSM Tracks | LinearRecurrentUnit | LinearRecurrentUnit | LinearRecurrentTrack | BioTrackEnergy |
| Controller | HyperController | HyperController | CortexController | - |
| Stack | - | - | MetaStateStack | - |
| Memory | HoloLink | ExternalMemory | FaultTraceBuffer | HoloLinkHebbian |
| Training | train.py | - | training_v2.py | - |

## Usage Patterns

### Using ANA v3 (current)
```python
from ana.config_v2 import ANAv2Config
from ana.model_v3 import ANAv2Model

config = ANAv2Config()
model = ANAv2Model(config)
logits = model(input_ids)
```

### Using Bio-ANA
```python
from ana.bio_ana import create_bio_ana

model = create_bio_ana('nano')
logits = model(input_ids, return_energy=True)
```

### Using bioplausible directly
```python
import sys
sys.path.insert(0, 'ana/eqprop')
from bioplausible.sklearn_interface import EqPropClassifier

clf = EqPropClassifier(model_name="EqProp MLP", hidden_dim=64)
clf.fit(X, y)
```

## Future Refactoring

### Recommended Changes
1. **Deprecate v1/v2 files** - Add deprecation warnings
2. **Create `ana/core/`** - Extract common utilities
3. **Consolidate configs** - Single config hierarchy
4. **Standardize naming** - Consistent Track/Controller naming
5. **Add docstrings** - Complete API documentation
6. **Type hints** - Full type annotation coverage

### Proposed Structure
```
ana/
├── core/
│   ├── attention.py
│   ├── memory.py
│   ├── stacks.py
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── tracks.py
│   ├── controllers.py
│   └── stack.py
├── config/
│   ├── __init__.py
│   └── configs.py
├── training/
│   ├── __init__.py
│   └── trainer.py
└── bio_ana/
    └── (existing)
```

## Migration Guide

### From v1 to v3
```python
# OLD (v1)
from ana.models import ANAModel, HyperController
config = ANAConfig(d_model=128)

# NEW (v3)
from ana.model_v3 import ANAv2Model
from ana.models_v3 import CortexController
config = ANAv2Config(d_model=128)
```

### From v3 to Bio-ANA
```python
# OLD (v3)
from ana.model_v3 import ANAv2Model
model = ANAv2Model(config)

# NEW (Bio-ANA)
from ana.bio_ana import BioANAModel
model = BioANAModel(config)  # Extends ANAv2Config
```

## Notes

- `eqprop/` is a submodule (bioplausible) and should not be modified
- Tests primarily target v3 and Bio-ANA
- v1/v2 files kept for historical compatibility
- Use `model_factory.py` for systematic architecture exploration
