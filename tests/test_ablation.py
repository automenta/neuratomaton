
import pytest
import torch
import torch.nn as nn
from ana.config import ANAConfig
from ana.models import ANAModel

def test_ablation_hololink():
    config = ANAConfig(use_hololink=False)
    model = ANAModel(config)

    # Check submodule
    assert not hasattr(model.layers[0], 'holo')

    # Check forward
    x = torch.randint(0, 10, (2, 5))
    logits, _ = model(x)
    assert logits.shape == (2, 5, config.vocab_size)

def test_ablation_controller():
    config = ANAConfig(use_controller=False)
    model = ANAModel(config)

    assert not hasattr(model.layers[0], 'controller')

    x = torch.randint(0, 10, (2, 5))
    logits, _ = model(x)
    assert logits.shape == (2, 5, config.vocab_size)

def test_ablation_tracks():
    config = ANAConfig(num_tracks=1)
    model = ANAModel(config)

    assert hasattr(model.layers[0], 'lru_0')
    assert not hasattr(model.layers[0], 'lru_1')

    x = torch.randint(0, 10, (2, 5))
    logits, _ = model(x)
    assert logits.shape == (2, 5, config.vocab_size)

    config3 = ANAConfig(num_tracks=3)
    model3 = ANAModel(config3)
    assert hasattr(model3.layers[0], 'lru_2')

    logits, _ = model3(x)
    assert logits.shape == (2, 5, config.vocab_size)

if __name__ == "__main__":
    test_ablation_hololink()
    test_ablation_controller()
    test_ablation_tracks()
