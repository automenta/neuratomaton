"""
Basic tests for ANA models
"""

import torch
import pytest
from ana.models.config import ANAConfig
from ana.models.core import ANAModel, BaselineSSM


def test_ana_config_creation():
    """Test that ANAConfig can be created with default values"""
    config = ANAConfig()
    assert config.d_model == 64
    assert config.vocab_size == 40
    assert config.use_hololink == True


def test_ana_model_creation():
    """Test that ANAModel can be created and run forward pass"""
    config = ANAConfig(
        vocab_size=50,
        d_model=32,
        state_dim=32,
        key_dim=16,
        num_layers=2,
        use_hololink=True,
        use_controller=False
    )
    
    model = ANAModel(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, info = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert isinstance(info, list)


def test_baseline_ssm_creation():
    """Test that BaselineSSM can be created and run forward pass"""
    config = ANAConfig(
        vocab_size=50,
        d_model=32,
        state_dim=32,
        num_layers=2,
        use_hololink=False,
        use_controller=False
    )
    
    model = BaselineSSM(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, info = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert isinstance(info, list)


def test_parameter_counts():
    """Test that parameter counts are reasonable"""
    # ANA with HoloLink should have more parameters than baseline
    ana_config = ANAConfig(
        vocab_size=50,
        d_model=32,
        state_dim=32,
        key_dim=16,
        num_layers=2,
        use_hololink=True,
        use_controller=False
    )
    
    baseline_config = ANAConfig(
        vocab_size=50,
        d_model=32,
        state_dim=32,
        num_layers=2,
        use_hololink=False,
        use_controller=False
    )
    
    ana_model = ANAModel(ana_config)
    baseline_model = BaselineSSM(baseline_config)
    
    ana_params = sum(p.numel() for p in ana_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    assert ana_params > baseline_params  # HoloLink adds parameters


def test_model_device_compatibility():
    """Test that models work on both CPU and GPU (if available)"""
    config = ANAConfig(
        vocab_size=30,
        d_model=16,
        state_dim=16,
        key_dim=8,
        num_layers=1,
        use_hololink=True,
        use_controller=False
    )
    
    # Test on CPU
    model_cpu = ANAModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    
    logits_cpu, _ = model_cpu(input_ids)
    assert logits_cpu.device.type == 'cpu'
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model_gpu = ANAModel(config).cuda()
        input_ids_gpu = input_ids.cuda()
        
        logits_gpu, _ = model_gpu(input_ids_gpu)
        assert logits_gpu.device.type == 'cuda'


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_ana_config_creation()
    test_ana_model_creation()
    test_baseline_ssm_creation()
    test_parameter_counts()
    test_model_device_compatibility()
    print("All tests passed!")