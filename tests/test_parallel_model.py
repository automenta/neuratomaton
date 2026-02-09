
import pytest
import torch
from ana.config import ANAConfig
from ana.models import ANAModel

def test_parallel_vs_sequential_model():
    # Config
    config_seq = ANAConfig(d_model=16, state_dim=8, num_layers=1, use_parallel_scan=False)
    config_par = ANAConfig(d_model=16, state_dim=8, num_layers=1, use_parallel_scan=True)

    # Init models with SAME weights
    model_seq = ANAModel(config_seq)
    model_par = ANAModel(config_par)

    model_par.load_state_dict(model_seq.state_dict())

    # Input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config_seq.vocab_size, (batch_size, seq_len))

    # Forward
    model_seq.eval()
    model_par.eval()

    with torch.no_grad():
        out_seq, _ = model_seq(input_ids)
        out_par, _ = model_par(input_ids)

    # Compare
    diff = (out_seq - out_par).abs().max()
    print(f"Max difference (Model): {diff.item()}")

    assert torch.allclose(out_seq, out_par, atol=1e-4, rtol=1e-3)

if __name__ == "__main__":
    test_parallel_vs_sequential_model()
