import pytest
import os
import shutil
import torch
from src.ana.utils.datasets import InductionHeadTask, MultiQueryAssociativeRecall, PointerChainTask
from src.ana.experiments.potential_reveal import PotentialRevealer

def test_induction_head_task():
    task = InductionHeadTask(num_samples=10, seq_len=32)
    x, y, mask = task[0]
    assert x.shape[0] == 31
    assert y.shape[0] == 31
    assert mask.shape[0] == 31
    assert mask.sum() == 1.0

def test_multi_query_task():
    task = MultiQueryAssociativeRecall(num_samples=10, num_pairs=4, num_queries=2)
    x, y, mask = task[0]
    # mask sum should be num_queries
    assert mask.sum() == 2.0

def test_pointer_chain_task():
    task = PointerChainTask(num_samples=10, chain_len=3)
    x, y, mask = task[0]
    # chain_len 3 means 4 nodes. Pairs: n1->n2, n2->n3, n3->n4.
    # Input seq len: 3 pairs * 2 * 2 + 2 = 14? No.
    # 3 pairs: K V K V K V (6 tokens).
    # Plus noise pairs (0).
    # Plus Query [Q] [n1] (2 tokens).
    # Total input context: 8 tokens.
    # Plus chain completion [n2] [n3] [n4] (3 tokens, except last is in target).
    # x: Context (8) + [n2] [n3] = 10?
    # Let's check logic:
    # seq = pairs(6) + query(2) = 8.
    # suffix_input = chain_rest[:-1] (length 2: n2, n3).
    # x = seq + suffix_input = 10.
    # y = seq[1:] + ...
    # mask covers chain prediction.
    assert len(x) > 0
    assert len(x) == len(y)
    assert mask.sum() > 0

def test_potential_revealer_instantiation():
    revealer = PotentialRevealer(output_dir="tests/results/potential_test")
    assert os.path.exists(revealer.output_dir)
    # Cleanup
    shutil.rmtree(revealer.output_dir)
