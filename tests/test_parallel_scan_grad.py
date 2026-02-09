
import pytest
import torch
import torch.nn.functional as F
from ana.models import parallel_scan_lru

def test_parallel_scan_grad():
    batch_size = 2
    seq_len = 20
    dim = 4

    # Inputs with grad
    u = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    alpha_logits = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    beta = torch.rand(batch_size, seq_len, dim, requires_grad=True)

    # Use logsigmoid
    log_alpha = F.logsigmoid(alpha_logits)

    h = parallel_scan_lru(u, log_alpha, beta)

    loss = h.sum()
    loss.backward()

    print("Grad u max:", u.grad.abs().max().item())
    print("Grad alpha max:", alpha_logits.grad.abs().max().item())

    assert not torch.isnan(u.grad).any()
    assert not torch.isnan(alpha_logits.grad).any()

if __name__ == "__main__":
    test_parallel_scan_grad()
