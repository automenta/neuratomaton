
import torch
import sys
import os

from ana.models.config import ANAConfig
from ana.models.core import ANARLAgent, ANASeriesModel

def test_extensions():
    print("Testing Extensions...")
    config = ANAConfig(
        d_model=16,
        state_dim=16,
        num_layers=2,
        vocab_size=10,
        track_count=2,
        observation_space=4,
        action_space=3,
        series_dim=2
    )

    # Test RL Agent
    print("Test RL Agent")
    agent = ANARLAgent(config)
    obs = torch.randn(2, 4) # Batch, Obs
    logits, value, state, info = agent(obs) # Now expecting 4 returns

    assert logits.shape == (2, 3) # Batch, Action
    assert value.shape == (2, 1)  # Batch, 1
    # Check state structure
    assert isinstance(state, tuple)
    assert len(state) == 2 # h, m
    assert 'layers' in info
    print("RL Agent Passed.")

    # Test Series Model
    print("Test Series Model")
    series_model = ANASeriesModel(config)

    # Step
    x = torch.randn(2, 2) # Batch, SeriesDim
    pred, state, info = series_model(x) # Now expecting 3 returns
    assert pred.shape == (2, 2)
    assert 'layers' in info
    print("Series Step Passed.")

    # Sequence
    x_seq = torch.randn(2, 5, 2) # Batch, Seq, SeriesDim
    pred_seq, info_seq = series_model.forward_sequence(x_seq) # Now expecting 2 returns
    assert pred_seq.shape == (2, 5, 2)
    assert 'layers' in info_seq
    print("Series Sequence Passed.")

    print("All tests passed!")

if __name__ == "__main__":
    test_extensions()
