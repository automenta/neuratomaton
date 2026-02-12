# ANA Experimental Findings

## Summary of Results (Feb 2026)

### Experiment: KV Associative Recall Scaling

**Hypothesis**: Two-phase training prevents interference between HoloLink and Controller that occurs during joint training.

**Results**:

| Method | Params | 12 KV pairs | 20 KV pairs |
|--------|--------|-------------|-------------|
| Two-Phase ANA | 562K | 92.5% | 72.5% |
| Joint Training ANA | 562K | 6.5% | 0.0% |
| SSM (no HoloLink) | 844K | 4.5% | 0.0% |
| SSM (no HoloLink) | 1.2M | 2.0% | 3.0% |

### Key Finding 1: Training Order Matters
Joint backpropagation destroys HoloLink performance (96% → 6.5%). Two-phase training preserves it (96% → 92.5%).

### Key Finding 2: Architecture > Parameters
Tiny ANA (562K params) beats 2x larger SSM (1.2M params) on associative recall by 70% absolute. The right inductive bias is worth 2x generic parameters.

---

## What This Is NOT

- NOT a breakthrough language model
- NOT beating GPT-2, TinyLLaMA, or any real LM
- NOT demonstrating practical value for real applications
- NOT a publishable result in its current form

## Why It Falls Short

1. **Synthetic task**: KV recall is not a real-world benchmark
2. **No text generation**: Can't show actual language capabilities
3. **No comparison to real models**: Only comparing to ourselves
4. **No practical application**: Solves an artificial problem

---

## What Would Be A Breakthrough

### Option A: Small LM with In-Context Learning

Train a ~5-10M param ANA on small corpus, demonstrate:
- Few-shot learning on real tasks (sentiment, QA, translation)
- Beat GPT-2 small (117M params) on specific benchmarks despite being 10x smaller
- Show actual text generation quality differences

### Option B: Long-Context Retrieval

Train on documents, demonstrate:
- Retrieve facts from 50K+ token context
- Beat retrieval-augmented baselines
- "Needle in haystack" with real documents

### Option C: Rapid Adaptation / Few-Shot

Demonstrate:
- Learn new tasks from 10-100 examples
- Compare to fine-tuning larger models
- Show efficiency gains

### Option D: On-Device Assistant

Build a tiny assistant that:
- Runs on mobile/embedded
- Maintains conversation context for hours
- Actually useful for something

---

## Current Limitations

1. **No tokenizer**: Using raw tokens, not real text
2. **No real data**: Only synthetic tasks
3. **No training pipeline**: Manual scripts, not scalable
4. **No evaluation framework**: Can't compare to baselines
5. **Position encoding limit**: 8K max, need longer for long-context claims

---

## Next Steps for Real Breakthrough

1. **Add real tokenizer** (sentencepiece/tiktoken)
2. **Get small corpus** (TinyStories, OpenWebText subset)
3. **Implement proper training loop** with logging, checkpoints
4. **Add evaluation** on real benchmarks (MMLU subset, Hellaswag, etc.)
5. **Compare to baselines** (GPT-2 small, TinyLLaMA)
6. **Show actual text outputs** side-by-side

The goal: "This 5M param model generates better text than GPT-2 small (117M params) on X task."
