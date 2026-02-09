import pytest
import torch
import torch.nn as nn
from ana.config import ANAConfig, TrainingConfig, DataConfig
from ana.models import ANAModel, BaselineSSM
from ana.train import evaluate, col_fn
from ana.data import AssociativeRecallDataset
from torch.utils.data import DataLoader

@pytest.fixture
def device():
    return torch.device('cpu')

class TestAblations:
    def test_no_controller(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=2, vocab_size=20,
            use_controller=False, use_hololink=True
        )
        model = ANAModel(config).to(device)
        
        dataset = AssociativeRecallDataset(size=50, vocab_size=20, min_noise=2, max_noise=5)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=col_fn)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        loss, stats, acc, needle_acc, ppl = evaluate(model, dataloader, criterion, device)
        
        assert loss > 0
        assert stats['ga_0'] == 0.0
    
    def test_no_hololink(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=2, vocab_size=20,
            use_controller=True, use_hololink=False
        )
        model = ANAModel(config).to(device)
        
        dataset = AssociativeRecallDataset(size=50, vocab_size=20, min_noise=2, max_noise=5)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=col_fn)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        loss, stats, acc, needle_acc, ppl = evaluate(model, dataloader, criterion, device)
        
        assert loss > 0
    
    def test_no_controller_no_hololink(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=2, vocab_size=20,
            use_controller=False, use_hololink=False
        )
        model = ANAModel(config).to(device)
        
        dataset = AssociativeRecallDataset(size=50, vocab_size=20, min_noise=2, max_noise=5)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=col_fn)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        loss, stats, acc, needle_acc, ppl = evaluate(model, dataloader, criterion, device)
        
        assert loss > 0
    
    def test_single_track(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=2, vocab_size=20,
            track_count=1, use_hololink=True
        )
        model = ANAModel(config).to(device)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)
    
    def test_four_tracks(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=2, vocab_size=20,
            track_count=4, use_hololink=True
        )
        model = ANAModel(config).to(device)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)
    
    def test_baseline_vs_ana(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=2, vocab_size=20,
            use_parallel_scan=True
        )
        
        baseline = BaselineSSM(config).to(device)
        ana = ANAModel(config).to(device)
        
        dataset = AssociativeRecallDataset(size=50, vocab_size=20, min_noise=2, max_noise=5)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=col_fn)
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        bl_loss, _, _, _, _ = evaluate(baseline, dataloader, criterion, device)
        ana_loss, _, _, _, _ = evaluate(ana, dataloader, criterion, device)
        
        assert bl_loss > 0
        assert ana_loss > 0

class TestThinkingSteps:
    def test_zero_thinking(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=1, vocab_size=20,
            max_thinking_steps=0, use_parallel_scan=False
        )
        model = ANAModel(config).to(device)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, info = model(input_ids, return_info=True)
        
        assert logits.shape == (2, 5, 20)
    
    def test_multiple_thinking(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=1, vocab_size=20,
            max_thinking_steps=3, use_parallel_scan=False
        )
        model = ANAModel(config).to(device)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, info = model(input_ids, return_info=True)
        
        assert logits.shape == (2, 5, 20)
        if len(info) > 0 and 'thinking_steps' in info[0]:
            assert info[0]['thinking_steps'] == 4

class TestHoloLinkVariants:
    def test_with_decay(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=1, vocab_size=20,
            hololink_decay=0.9
        )
        model = ANAModel(config).to(device)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)
    
    def test_with_orthogonal_init(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=1, vocab_size=20,
            orthogonal_init=True
        )
        model = ANAModel(config).to(device)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)
    
    def test_without_learned_binding(self, device):
        config = ANAConfig(
            d_model=32, state_dim=16, num_layers=1, vocab_size=20,
            use_learned_binding=False
        )
        model = ANAModel(config).to(device)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)

class TestParallelScan:
    def test_parallel_runs(self, device):
        config_par = ANAConfig(
            d_model=32, state_dim=16, num_layers=1, vocab_size=20,
            use_parallel_scan=True, use_hololink=False, use_controller=False
        )
        
        model_par = ANAModel(config_par).to(device)
        
        torch.manual_seed(123)
        input_ids = torch.randint(0, 20, (2, 8))
        
        logits_par, _ = model_par(input_ids)
        
        assert logits_par.shape == (2, 8, 20)
