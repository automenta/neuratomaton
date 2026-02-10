#!/usr/bin/env python3
# Experiment 4: Character-level Language Modeling
import torch
from ana.config import ANAConfig, TrainingConfig, DataConfig
from ana.models import ANAModel
from ana.train import run_training
import json
import os

print('='*70)
print('CHARACTER-LEVEL LANGUAGE MODELING')
print('='*70)
print()

# Generate corpus
os.makedirs('data', exist_ok=True)
corpus_path = 'data/corpus.txt'
with open(corpus_path, 'w') as f:
    # Write code files
    import glob
    for py_file in glob.glob('ana/*.py'):
        with open(py_file, 'r') as pf:
            f.write(pf.read() + '\n\n')
    # Write README
    with open('README.md', 'r') as rf:
        f.write(rf.read())

print(f'Corpus generated: {corpus_path}')
print(f'Size: {os.path.getsize(corpus_path)} bytes')
print()

config = ANAConfig(
    d_model=128,
    state_dim=128,
    num_layers=3,
    track_count=2,
    key_dim=64,
    vocab_size=256,  # ASCII character set
)

train_config = TrainingConfig(
    batch_size=8,
    epochs=20,
    stage='2b',  # Text warmup stage
    save_checkpoints=False,
    learning_rate=1e-3,
)

data_config = DataConfig(
    vocab_size=256,
    seq_len=64,
    dataset_path=corpus_path,
    dataset_size=None,  # Use all available data
)

print(f'Model config: {config.d_model}d x {config.state_dim} state x {config.num_layers} layers')
print(f'Training: {train_config.epochs} epochs, batch {train_config.batch_size}')
print()

history = run_training(config, train_config, data_config, model_type='ana', num_workers=0)

results = {
    'final_loss': history['loss'][-1],
    'final_ppl': history['ppl'][-1],
    'final_acc': history['acc'][-1],
    'loss_curve': history['loss'],
    'ppl_curve': history['ppl'],
    'acc_curve': history['acc'],
}

print()
print('='*70)
print('CHARACTER-LEVEL LM RESULTS')
print('='*70)
print(f'Final Loss: {results["final_loss"]:.4f}')
print(f'Final Perplexity: {results["final_ppl"]:.2f}')
print(f'Final Accuracy: {results["final_acc"]:.2%}')
print()

with open('archive/phase5_char_lm.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to archive/phase5_char_lm.json')
