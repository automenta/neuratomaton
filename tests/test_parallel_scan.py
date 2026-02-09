
import pytest
import torch
import torch.nn.functional as F
from ana.models import parallel_scan_lru

def test_parallel_scan_correctness():
    batch_size = 2
    seq_len = 10
    dim = 4

    # Random inputs
    u = torch.randn(batch_size, seq_len, dim)
    # Logits for alpha
    alpha_logits = torch.randn(batch_size, seq_len, dim)
    alpha = torch.sigmoid(alpha_logits)
    log_alpha = F.logsigmoid(alpha_logits)

    # Beta in (0, 1)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, dim))

    # 1. Parallel Scan
    h_parallel = parallel_scan_lru(u, log_alpha, beta)

    # 2. Sequential Scan
    h_seq_list = []
    h_prev = torch.zeros(batch_size, dim)

    for t in range(seq_len):
        # h_t = alpha_t * h_{t-1} + beta_t * u_t
        h_t = alpha[:, t, :] * h_prev + beta[:, t, :] * u[:, t, :]
        h_seq_list.append(h_t)
        h_prev = h_t

    h_sequential = torch.stack(h_seq_list, dim=1)

    # Compare
    diff = (h_parallel - h_sequential).abs().max()
    print(f"Max difference: {diff.item()}")

    assert torch.allclose(h_parallel, h_sequential, atol=1e-4, rtol=1e-3)

if __name__ == "__main__":
    test_parallel_scan_correctness()
