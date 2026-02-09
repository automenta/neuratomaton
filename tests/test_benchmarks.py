import pytest
import torch
from ana.config import ANAConfig
from ana.models import ANAModel
from ana.benchmarks import (
    MultiQueryARDataset, InductionHeadDataset, LongContextARDataset,
    BenchmarkEvaluator
)

@pytest.fixture
def device():
    return torch.device('cpu')

@pytest.fixture
def small_model(device):
    config = ANAConfig(
        d_model=16, state_dim=8, num_layers=1, vocab_size=100,
        track_count=1, use_hololink=False, use_controller=False
    )
    return ANAModel(config).to(device)

class TestBenchmarkDatasets:
    def test_multi_query_ar(self):
        ds = MultiQueryARDataset(size=10, vocab_size=100, num_kv_pairs=4, noise_multiplier=1)
        
        assert len(ds) == 10
        
        x, y, mask = ds[0]
        assert x.shape == y.shape
        assert mask[-1] == 1.0
    
    def test_induction_head(self):
        ds = InductionHeadDataset(size=10, vocab_size=100, seq_len=16, pattern_len=3)
        
        assert len(ds) == 10
        
        x, y, mask = ds[0]
        assert x.shape == y.shape
    
    def test_long_context(self):
        ds = LongContextARDataset(size=5, vocab_size=100, context_len=100, kv_position='start')
        
        assert len(ds) == 5
        
        x, y, mask = ds[0]
        assert x.size(0) > 100

class TestBenchmarkEvaluator:
    def test_evaluator_basic(self, small_model, device):
        evaluator = BenchmarkEvaluator(small_model, device, vocab_size=100)
        
        ds = MultiQueryARDataset(size=5, vocab_size=100, num_kv_pairs=2, noise_multiplier=1)
        acc = evaluator.evaluate_task(ds, "test_mqar")
        
        assert 0.0 <= acc <= 1.0
        assert "test_mqar" in evaluator.results
    
    def test_associative_recall_sweep(self, small_model, device):
        evaluator = BenchmarkEvaluator(small_model, device, vocab_size=100)
        evaluator.run_associative_recall_sweep()
        
        assert len(evaluator.results) > 0
        
        for name, acc in evaluator.results.items():
            assert 0.0 <= acc <= 1.0
    
    def test_mqar_sweep(self, small_model, device):
        evaluator = BenchmarkEvaluator(small_model, device, vocab_size=100)
        
        ds = MultiQueryARDataset(size=5, vocab_size=100, num_kv_pairs=4, noise_multiplier=1)
        acc = evaluator.evaluate_task(ds, "test_mqar_4")
        
        assert 0.0 <= acc <= 1.0
        assert "test_mqar_4" in evaluator.results
