import pytest
import torch
import numpy as np

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "cuda: marks tests that require CUDA")

@pytest.fixture(scope="session")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    np.random.seed(42)
