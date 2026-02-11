#!/usr/bin/env python3
"""
ANA v2: Curriculum tasks that PROVE it works.

The test: Can ANA learn algorithms from ONE example and generalize?

Tasks:
1. Copy Task - Learn to repeat input sequence
2. Reverse Task - Learn to reverse input sequence  
3. Associative Recall - Learn to retrieve value based on key
4. Arithmetic - Learn to perform calculations from examples
5. Sorting - Learn to sort sequences

The key: Train on SHORT sequences, test on LONGER sequences.
If it generalizes, it learned the ALGORITHM, not memorized patterns.
"""

import torch
import random
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class Task:
    name: str
    train_seqs: torch.Tensor
    train_targets: torch.Tensor
    test_seqs: torch.Tensor
    test_targets: torch.Tensor
    vocab_size: int


def generate_copy_task(num_train: int = 1000, 
                       num_test: int = 200,
                       train_len: Tuple[int, int] = (5, 10),
                       test_len: Tuple[int, int] = (11, 30),
                       vocab_size: int = 10) -> Task:
    """
    Copy Task: Input [a, b, c] → Output [a, b, c]
    
    Simple baseline - if this fails, nothing works.
    """
    def generate_batch(num, length_range):
        seqs = []
        targets = []
        for _ in range(num):
            length = random.randint(*length_range)
            seq = [random.randint(1, vocab_size) for _ in range(length)]
            target = seq.copy()
            seqs.append(seq)
            targets.append(target)
        return seqs, targets
    
    train_seqs, train_targets = generate_batch(num_train, train_len)
    test_seqs, test_targets = generate_batch(num_test, test_len)
    
    return Task(
        name="Copy",
        train_seqs=_pad_sequences(train_seqs),
        train_targets=_pad_sequences(train_targets),
        test_seqs=_pad_sequences(test_seqs),
        test_targets=_pad_sequences(test_targets),
        vocab_size=vocab_size + 1
    )


def generate_reverse_task(num_train: int = 1000,
                          num_test: int = 200,
                          train_len: Tuple[int, int] = (3, 7),
                          test_len: Tuple[int, int] = (8, 20),
                          vocab_size: int = 10) -> Task:
    """
    Reverse Task: Input [a, b, c] → Output [c, b, a]
    
    The classic "learn algorithm from examples" test.
    Requires working memory to store and retrieve in reverse order.
    """
    def generate_batch(num, length_range):
        seqs = []
        targets = []
        for _ in range(num):
            length = random.randint(*length_range)
            seq = [random.randint(1, vocab_size) for _ in range(length)]
            target = seq[::-1]
            seqs.append(seq)
            targets.append(target)
        return seqs, targets
    
    train_seqs, train_targets = generate_batch(num_train, train_len)
    test_seqs, test_targets = generate_batch(num_test, test_len)
    
    return Task(
        name="Reverse",
        train_seqs=_pad_sequences(train_seqs),
        train_targets=_pad_sequences(train_targets),
        test_seqs=_pad_sequences(test_seqs),
        test_targets=_pad_sequences(test_targets),
        vocab_size=vocab_size + 1
    )


def generate_associative_recall_task(num_train: int = 1000,
                                      num_test: int = 200,
                                      train_pairs: Tuple[int, int] = (2, 4),
                                      test_pairs: Tuple[int, int] = (5, 10),
                                      vocab_size: int = 20) -> Task:
    """
    Associative Recall: Input [k1, v1, k2, v2, QUERY] → Output [v_for_QUERY]
    
    Tests holographic memory capacity. The model must learn to:
    1. Store key-value pairs in memory
    2. Retrieve the correct value for a query key
    
    Vocab split: 1-10 for keys, 11-20 for values
    """
    key_range = (1, vocab_size // 2 + 1)
    value_range = (vocab_size // 2 + 1, vocab_size + 1)
    
    def generate_batch(num, pairs_range):
        seqs = []
        targets = []
        for _ in range(num):
            num_pairs = random.randint(*pairs_range)
            
            keys = [random.randint(*key_range) for _ in range(num_pairs)]
            values = [random.randint(*value_range) for _ in range(num_pairs)]
            
            kv_pairs = []
            for k, v in zip(keys, values):
                kv_pairs.extend([k, v])
            
            query_key = random.choice(keys)
            target_value = values[keys.index(query_key)]
            
            seq = kv_pairs + [query_key + vocab_size]  # Offset query to distinguish
            target = [target_value]
            
            seqs.append(seq)
            targets.append(target)
        return seqs, targets
    
    train_seqs, train_targets = generate_batch(num_train, train_pairs)
    test_seqs, test_targets = generate_batch(num_test, test_pairs)
    
    return Task(
        name="AssociativeRecall",
        train_seqs=_pad_sequences(train_seqs),
        train_targets=_pad_sequences(train_targets),
        test_seqs=_pad_sequences(test_seqs),
        test_targets=_pad_sequences(test_targets),
        vocab_size=vocab_size * 2 + 1
    )


def generate_arithmetic_task(num_train: int = 1000,
                             num_test: int = 200,
                             train_digits: int = 2,
                             test_digits: int = 3,
                             vocab_size: int = 12) -> Task:
    """
    Arithmetic Task: Input "a + b =" → Output sum
    
    Tests algorithmic reasoning. Model must learn:
    1. Parse the operation
    2. Perform calculation
    3. Format output
    
    Vocab: 0-9 for digits, 10 for +, 11 for =
    """
    def generate_batch(num, num_digits):
        seqs = []
        targets = []
        for _ in range(num):
            a = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)
            b = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)
            result = a + b
            
            seq = [int(d) for d in str(a)] + [10] + [int(d) for d in str(b)] + [11]
            target = [int(d) for d in str(result)]
            
            seqs.append(seq)
            targets.append(target)
        return seqs, targets
    
    train_seqs, train_targets = generate_batch(num_train, train_digits)
    test_seqs, test_targets = generate_batch(num_test, test_digits)
    
    return Task(
        name="Arithmetic",
        train_seqs=_pad_sequences(train_seqs),
        train_targets=_pad_sequences(train_targets),
        test_seqs=_pad_sequences(test_seqs),
        test_targets=_pad_sequences(test_targets),
        vocab_size=12
    )


def generate_sorting_task(num_train: int = 1000,
                          num_test: int = 200,
                          train_len: Tuple[int, int] = (3, 5),
                          test_len: Tuple[int, int] = (6, 10),
                          vocab_size: int = 10) -> Task:
    """
    Sorting Task: Input [c, a, b] → Output [a, b, c]
    
    Tests complex algorithm learning. Requires:
    1. Compare elements
    2. Maintain working state
    3. Output sorted order
    """
    def generate_batch(num, length_range):
        seqs = []
        targets = []
        for _ in range(num):
            length = random.randint(*length_range)
            seq = [random.randint(1, vocab_size) for _ in range(length)]
            target = sorted(seq)
            seqs.append(seq)
            targets.append(target)
        return seqs, targets
    
    train_seqs, train_targets = generate_batch(num_train, train_len)
    test_seqs, test_targets = generate_batch(num_test, test_len)
    
    return Task(
        name="Sorting",
        train_seqs=_pad_sequences(train_seqs),
        train_targets=_pad_sequences(train_targets),
        test_seqs=_pad_sequences(test_seqs),
        test_targets=_pad_sequences(test_targets),
        vocab_size=vocab_size + 1
    )


def _pad_sequences(sequences: List[List[int]], pad_value: int = 0) -> torch.Tensor:
    """Pad sequences to the same length."""
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded.append(seq + [pad_value] * (max_len - len(seq)))
    return torch.tensor(padded, dtype=torch.long)


def evaluate_task(model, task: Task) -> Dict[str, float]:
    """
    Evaluate a model on a task.
    
    Returns:
        - accuracy: Exact match accuracy
        - token_accuracy: Per-token accuracy
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(task.test_seqs)
        predictions = logits.argmax(dim=-1)
        
        exact_match = 0
        total_tokens = 0
        correct_tokens = 0
        
        for i in range(len(task.test_seqs)):
            pred = predictions[i]
            target = task.test_targets[i]
            
            if torch.equal(pred, target):
                exact_match += 1
            
            mask = target != 0
            total_tokens += mask.sum().item()
            correct_tokens += ((pred == target) & mask).sum().item()
    
    return {
        'exact_accuracy': exact_match / len(task.test_seqs),
        'token_accuracy': correct_tokens / total_tokens
    }


def get_all_tasks() -> Dict[str, Task]:
    """Get all curriculum tasks."""
    return {
        'copy': generate_copy_task(),
        'reverse': generate_reverse_task(),
        'ar_recall': generate_associative_recall_task(),
        'arithmetic': generate_arithmetic_task(),
        'sorting': generate_sorting_task()
    }


def run_curriculum(model, tasks: Dict[str, Task], epochs_per_task: int = 20):
    """
    Run curriculum learning through tasks.
    
    Each task is trained sequentially. Success on a task unlocks the next.
    """
    results = {}
    
    for task_name, task in tasks.items():
        print(f"\n{'='*60}")
        print(f"Training on task: {task_name}")
        print(f"{'='*60}")
        
        from .train import Trainer, SimpleDataset, DataLoader
        
        dataset = SimpleDataset(task.train_seqs, task.train_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        trainer = Trainer(task.config)
        history = trainer.train(loader, num_epochs=epochs_per_task)
        
        eval_results = evaluate_task(model, task)
        results[task_name] = eval_results
        
        print(f"Results for {task_name}:")
        print(f"  Exact Accuracy: {eval_results['exact_accuracy']:.2%}")
        print(f"  Token Accuracy: {eval_results['token_accuracy']:.2%}")
        
        if eval_results['exact_accuracy'] < 0.5:
            print(f"WARNING: {task_name} performance below 50%. May need more training.")
    
    return results
