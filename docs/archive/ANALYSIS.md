# Analysis of ANA's Failure to Learn Sequence Reversal

## Summary

ANA, a state space model-based architecture, fails to learn the sequence reversal algorithm. Instead, it learns position-specific token mappings. This is a manifestation of the "Reversal Curse" - autoregressive models learn simple token-to-position mappings rather than complex algorithmic operations.

## Key Findings

### Performance Comparison

| Task | Model | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L10 | L12 |
|------|-------|----|----|----|----|----|----|----|-----|-----|
| Copy | ANA | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 99.8% | 99.7% |
| Reversal | ANA | 56.6% | 50% | 35% | 28% | 25% | 19.3% | 19.8% | 15.3% | 12.3% |
| Reversal | ReverseNet | 90% | 75% | 60% | 50% | 45% | 42.3% | 35.7% | 28.6% | 25.4% |

### What ANA Actually Learned

From our analysis, the model learned:
- For any input of length ≥1, position 0 always predicts 2
- Position 1 always predicts 1
- All positions ≥2 always predict 1

This is a purely heuristic, position-based strategy that fails completely on unseen lengths.

## What We Tried

1. **Basic Training:** Standard training on random sequences
2. **Hyperparameter Tuning:**
   - Learning rates: 1e-2, 1e-3, 5e-4
   - Batch sizes: 16, 32
   - State dimensions: 32, 64, 128, 256
   - Layers: 1, 2, 3, 4
3. **Curriculum Learning:**
   - Progressive lengths (2→6)
   - Warmup strategies
   - Direct training on target lengths
4. **Data Augmentation:**
   - Training on both forward and backward sequences
   - Variable vocab datasets
   - Position encoding
5. **Architecture Modifications:**
   - Added position encoding
   - Changed track counts (1, 2, 3 tracks)
   - Enabled/disabled HoloLink
6. **Specialized Architectures:** Created ReverseNet (bidirectional LSTM)

## Why ANA Fails

### 1. Reversal Curse
The Reversal Curse is a known limitation of autoregressive models. They learn simple token-to-position mappings rather than complex algorithmic operations.

### 2. Causal Structure
ANA's autoregressive nature limits bidirectional reasoning. Each token only has access to previous tokens, making it hard to model reversal.

### 3. Task Complexity
Reversal requires understanding the entire sequence and mapping from position i to position n-1-i, which is more complex than copy task's simple token prediction.

## Solution Approaches

### 1. Modify the ANA Architecture
- Add bidirectional processing capabilities
- Incorporate position-specific attention mechanisms
- Develop track interactions that enable bidirectional reasoning

### 2. Redesign the Training Methodology
- Create more challenging training datasets that require algorithmic learning
- Develop curriculum strategies that force generalization
- Explore meta-learning approaches

### 3. Reformulate the Task
- Frame reversal as a problem that requires algorithmic thinking
- Explore intermediate supervision signals

## Model Architecture Details

ANA Model Parameters:
```
Total params: 54,035

Layer breakdown:
- embedding: 640
- position_encoding: 6,400
- controller.net: 8,320
- controller.head: 520
- tracks.input_proj: 8,320
- tracks.output_proj: 8,320
- holo.q_proj: 4,096
- holo.k_proj: 8,192
- holo.v_proj: 8,192
- norm: 128
- output_head: 650
```

## Visualizations

### Task Performance
![Task Comparison](ana_task_comparison.png)

### Copy Task Performance
![Copy Task](ana_copy_performance.png)

## Conclusion

ANA works well for simple tasks like copy, but it fails to learn algorithmic tasks like reversal. This is due to the Reversal Curse - autoregressive models learn token-to-position mappings rather than algorithms.

To achieve human-like algorithmic reasoning, we need:
1. Better architectures with appropriate inductive biases for bidirectional reasoning
2. Training methods that encourage generalization over memorization
3. Curricula that systematically build algorithmic skills

This investigation has provided valuable insights into the limitations of ANA and state space models for algorithmic learning tasks.
