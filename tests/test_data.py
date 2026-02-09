import pytest
import torch
from ana.data import AssociativeRecallDataset, TextDataset
from ana.eval import CopyTaskDataset, ReverseTaskDataset, AdditionTaskDataset, SortTaskDataset, run_eval_task
from ana.config import ANAConfig
from ana.models import ANAModel
import tempfile
import os

class TestAssociativeRecallDataset:
    def test_basic(self):
        dataset = AssociativeRecallDataset(size=100, vocab_size=30, min_noise=5, max_noise=10)
        
        assert len(dataset) == 100
        
        x, y, mask = dataset[0]
        
        assert x.shape == y.shape
        assert mask.shape == y.shape
        assert mask[-1] == 1.0
    
    def test_vocab_range(self):
        dataset = AssociativeRecallDataset(size=10, vocab_size=30, min_noise=2, max_noise=5)
        
        for i in range(10):
            x, y, mask = dataset[i]
            assert x.max() < 30
            assert x.min() >= 0

class TestTextDataset:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello world! This is a test. " * 100)
            temp_path = f.name
        
        try:
            dataset = TextDataset(temp_path, seq_len=32)
            
            assert len(dataset) > 0
            
            x, y = dataset[0]
            assert x.shape == (32,)
            assert y.shape == (32,)
        finally:
            os.unlink(temp_path)

class TestEvalDatasets:
    def test_copy_task(self):
        dataset = CopyTaskDataset(size=50, vocab_size=20, seq_len=5)
        
        assert len(dataset) == 50
        
        x, y, mask = dataset[0]
        assert mask.sum() > 0
    
    def test_reverse_task(self):
        dataset = ReverseTaskDataset(size=50, vocab_size=20, seq_len=5)
        
        assert len(dataset) == 50
        
        x, y, mask = dataset[0]
        assert mask.sum() > 0
    
    def test_addition_task(self):
        dataset = AdditionTaskDataset(size=50, max_digits=2)
        
        assert len(dataset) == 50
        
        x, y, mask = dataset[0]
        assert mask.sum() > 0
    
    def test_sort_task(self):
        dataset = SortTaskDataset(size=50, vocab_size=20, seq_len=5)
        
        assert len(dataset) == 50
        
        x, y, mask = dataset[0]
        assert mask.sum() > 0

class TestRunEvalTask:
    def test_eval_runs(self):
        config = ANAConfig(d_model=16, state_dim=8, num_layers=1, vocab_size=15, track_count=1, use_hololink=False)
        model = ANAModel(config)
        
        copy_ds = CopyTaskDataset(size=20, vocab_size=15, seq_len=3)
        
        score = run_eval_task(model, copy_ds, torch.device('cpu'))
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
