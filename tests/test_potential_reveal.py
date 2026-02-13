import pytest
import os
import shutil
import torch
from src.ana.utils.datasets import InductionHeadTask, MultiQueryAssociativeRecall
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

def test_potential_revealer_instantiation():
    revealer = PotentialRevealer(output_dir="tests/results/potential_test")
    assert os.path.exists(revealer.output_dir)
    # Cleanup
    shutil.rmtree(revealer.output_dir)
