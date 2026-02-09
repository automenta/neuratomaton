import pytest
import torch
import torch.nn as nn
from ana.config import ANAConfig
from ana.models import ANAModel
from ana.train import train_one_epoch, evaluate, col_fn
from ana.data import AssociativeRecallDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tempfile
import os

@pytest.fixture
def device():
    return torch.device('cpu')

class TestTrainingPipeline:
    def test_train_one_step(self, device):
        config = ANAConfig(d_model=32, state_dim=16, num_layers=1, vocab_size=20, track_count=2)
        model = ANAModel(config).to(device)
        
        dataset = AssociativeRecallDataset(size=10, vocab_size=20, min_noise=2, max_noise=4)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=col_fn)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)
            loss, _ = train_one_epoch(model, dataloader, optimizer, criterion, device, writer, 0)
            writer.close()
        
        assert loss > 0
        assert loss < 10  # Sanity check
    
    def test_evaluate_function(self, device):
        config = ANAConfig(d_model=32, state_dim=16, num_layers=1, vocab_size=20, track_count=2)
        model = ANAModel(config).to(device)
        
        dataset = AssociativeRecallDataset(size=10, vocab_size=20, min_noise=2, max_noise=4)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=col_fn)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        loss, stats, acc, needle_acc, ppl = evaluate(model, dataloader, criterion, device)
        
        assert loss > 0
        assert 0 <= acc <= 1
        assert 0 <= needle_acc <= 1
        assert ppl > 0
    
    def test_force_prob_affects_training(self, device):
        config = ANAConfig(d_model=32, state_dim=16, num_layers=1, vocab_size=20, track_count=2)
        
        dataset = AssociativeRecallDataset(size=10, vocab_size=20, min_noise=2, max_noise=4)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=col_fn)
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without force
            model1 = ANAModel(config).to(device)
            opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
            writer1 = SummaryWriter(tmpdir)
            loss1, _ = train_one_epoch(model1, dataloader, opt1, criterion, device, writer1, 0, force_prob=0.0)
            writer1.close()
            
            # With force
            model2 = ANAModel(config).to(device)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            writer2 = SummaryWriter(tmpdir)
            loss2, _ = train_one_epoch(model2, dataloader, opt2, criterion, device, writer2, 0, force_prob=1.0)
            writer2.close()
        
        # Both should complete without error
        assert loss1 > 0 and loss2 > 0
