import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ana" / "eqprop"))

from ana.bio_ana import (
    BioANAModel,
    BioANAConfig,
    get_bio_config,
    BioSpecializedTracks,
    BioHoloLink,
    create_bio_ana,
)


class TestBioANAConfig:
    def test_default_config(self):
        config = BioANAConfig()
        assert config.d_model == 128
        assert config.syntax_dim == 64
        assert config.relaxation_iterations == 20
        assert config.spectral_radius == 0.99
    
    def test_variant_configs(self):
        nano = get_bio_config('nano')
        assert nano.d_model == 128
        
        small = get_bio_config('small')
        assert small.d_model == 512
        
        base = get_bio_config('base')
        assert base.d_model == 768
    
    def test_custom_overrides(self):
        config = get_bio_config('nano', relaxation_iterations=50, d_model=256)
        assert config.relaxation_iterations == 50
        assert config.d_model == 256


class TestBioTracks:
    def test_track_shapes(self):
        tracks = BioSpecializedTracks(
            d_model=128,
            syntax_dim=32,
            semantic_dim=64,
            logic_dim=32,
        )
        
        x = torch.randn(4, 128)
        output, states = tracks(x, steps=10)
        
        assert output.shape == (4, 128)
        assert states['syntax'].shape == (4, 32)
        assert states['semantic'].shape == (4, 64)
        assert states['logic'].shape == (4, 32)
    
    def test_energy_computation(self):
        tracks = BioSpecializedTracks(
            d_model=64,
            syntax_dim=16,
            semantic_dim=32,
            logic_dim=16,
        )
        
        x = torch.randn(2, 64)
        output, states = tracks(x, steps=10)
        
        energy = tracks.compute_energy(
            states['syntax'],
            states['semantic'],
            states['logic'],
            x
        )
        
        assert 'syntax' in energy
        assert 'semantic' in energy
        assert 'logic' in energy
        assert 'total' in energy
        assert energy['syntax'].shape == (2,)
    
    def test_track_convergence(self):
        tracks = BioSpecializedTracks(
            d_model=32,
            syntax_dim=8,
            semantic_dim=16,
            logic_dim=8,
        )
        
        x = torch.randn(2, 32)
        
        h_syntax = None
        h_semantic = None
        h_logic = None
        diff = 0.0
        
        for _ in range(50):
            output, h_new = tracks(x, h_syntax=h_syntax, h_semantic=h_semantic, h_logic=h_logic, steps=1)
            if h_syntax is not None:
                diff = sum(
                    torch.abs(h_new[k] - prev).max().item()
                    for k, prev in [('syntax', h_syntax), ('semantic', h_semantic), ('logic', h_logic)]
                )
            h_syntax = h_new['syntax']
            h_semantic = h_new['semantic']
            h_logic = h_new['logic']
        
        assert diff < 0.1, f"Tracks should converge, got diff={diff}"


class TestBioHoloLink:
    def test_hololink_forward(self):
        hololink = BioHoloLink(input_dim=128, key_dim=128, capacity=100)
        
        h = torch.randn(4, 128)
        output, info = hololink(h, write_mode=True)
        
        assert output.shape == (4, 128)
        assert 'weights' in info
    
    def test_hololink_memory_stats(self):
        hololink = BioHoloLink(input_dim=64, key_dim=32, capacity=100)
        
        stats = hololink.get_memory_stats()
        assert stats['capacity'] == 100
        assert stats['utilization'] >= 0.0
    
    def test_hebbian_update(self):
        hololink = BioHoloLink(input_dim=128, key_dim=128, capacity=50, hebbian_lr=0.1)
        
        h = torch.randn(4, 128)
        
        for _ in range(10):
            output, info = hololink(h, write_mode=True)
        
        stats = hololink.get_memory_stats()
        assert stats['avg_norm'] > 0


class TestBioANAModel:
    def test_forward_pass(self):
        config = get_bio_config('nano')
        model = BioANAModel(config)
        
        input_ids = torch.randint(0, 50, (2, 16))
        logits = model(input_ids)
        
        assert logits.shape == (2, 16, 50)
    
    def test_forward_with_energy(self):
        config = get_bio_config('nano', relaxation_iterations=5)
        model = BioANAModel(config)
        
        input_ids = torch.randint(0, 50, (2, 8))
        logits, energy = model(input_ids, return_energy=True)
        
        assert logits.shape == (2, 8, 50)
        assert len(energy) == 8
        assert 'total' in energy[0]
    
    def test_forward_with_info(self):
        config = get_bio_config('nano', relaxation_iterations=5)
        model = BioANAModel(config)
        
        input_ids = torch.randint(0, 50, (2, 8))
        logits, info = model(input_ids, return_info=True)
        
        assert logits.shape == (2, 8, 50)
        assert len(info) == 8
        assert 'track_states' in info[0]
    
    def test_backward_pass(self):
        config = get_bio_config('nano', relaxation_iterations=5)
        model = BioANAModel(config)
        
        input_ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))
        
        logits = model(input_ids)
        loss = model.compute_loss(logits, targets)
        
        loss['total'].backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and 'memory' not in name:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_create_bio_ana_factory(self):
        model = create_bio_ana('nano')
        assert isinstance(model, BioANAModel)
        
        model = create_bio_ana('small')
        assert model.config.d_model == 512
    
    def test_memory_stats(self):
        model = create_bio_ana('nano', use_hebbian_memory=True)
        
        input_ids = torch.randint(0, 50, (2, 8))
        model(input_ids)
        
        stats = model.get_memory_stats()
        assert 'hololink' in stats


class TestBioANAIntegration:
    def test_training_step(self):
        model = create_bio_ana('nano')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        input_ids = torch.randint(0, 50, (4, 16))
        targets = torch.randint(0, 50, (4, 16))
        
        for step in range(5):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = model.compute_loss(logits, targets)
            loss['total'].backward()
            optimizer.step()
            
            assert torch.isfinite(loss['total']), f"Loss became non-finite at step {step}"
    
    def test_energy_decreases(self):
        config = get_bio_config('nano', relaxation_iterations=20)
        model = BioANAModel(config)
        model.eval()
        
        input_ids = torch.randint(0, 50, (1, 4))
        
        with torch.no_grad():
            _, energy = model(input_ids, return_energy=True)
        
        total_energies = [e['total'].mean().item() for e in energy]
        
        decreasing_count = sum(
            1 for i in range(1, len(total_energies))
            if total_energies[i] <= total_energies[i-1] + 1.0
        )
        
        ratio = decreasing_count / (len(total_energies) - 1)
        assert ratio > 0.25, f"Energy should mostly decrease, got ratio={ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
