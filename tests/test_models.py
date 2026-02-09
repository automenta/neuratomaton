
import pytest
import torch
import torch.nn as nn
from ana.config import ANAConfig
from ana.models import ANAModel, LinearRecurrentUnit, HyperController, HoloLink

@pytest.fixture
def config():
    return ANAConfig(d_model=32, state_dim=16, num_layers=2, vocab_size=20)

def test_lru(config):
    batch_size = 4
    seq_len = 10
    lru = LinearRecurrentUnit(config)

    x = torch.randn(batch_size, config.d_model)
    h_prev = torch.zeros(batch_size, config.state_dim)

    # Test step
    y, h_next, (alpha, beta) = lru(x, h_prev)

    assert y.shape == (batch_size, config.d_model)
    assert h_next.shape == (batch_size, config.state_dim)
    assert alpha.shape == (batch_size, config.state_dim) or alpha.shape == (config.state_dim,)
    assert beta.shape == (batch_size, config.state_dim) or beta.shape == (config.state_dim,)

def test_controller(config):
    batch_size = 4
    ctl = HyperController(config)

    x = torch.randn(batch_size, config.d_model)

    gates = ctl(x)

    assert isinstance(gates, dict)
    assert 'alpha_0' in gates
    assert 'beta_0' in gates
    assert 'ret' in gates

    # Check shapes
    assert gates['alpha_0'].shape == (batch_size, 1)

    # Check force behavior
    ctl.train()
    gates_forced = ctl(x, force_prob=1.0)
    g_ret_forced = gates_forced['ret']
    # g_ret should be roughly 5.0
    assert torch.allclose(g_ret_forced, torch.ones_like(g_ret_forced) * 5.0, atol=1e-5)

def test_controller_deeper():
    # Test deeper controller config
    config = ANAConfig(d_model=32, controller_hidden_dim=16, controller_layers=3)
    ctl = HyperController(config)

    # Check network depth:
    # Input -> (Linear, SiLU) -> (Linear, SiLU) -> (Linear, SiLU) -> Head
    # net should have 2 * 3 = 6 layers?
    # Original code: input layer (2) + (layers-1)*2 hidden
    # 2 + 2*2 = 6. Correct.

    assert len(ctl.net) == 2 + (config.controller_layers - 1) * 2

    x = torch.randn(4, 32)
    gates = ctl(x)
    assert 'alpha_0' in gates

def test_hololink(config):
    batch_size = 4
    # HoloLink takes concatenated state from 2 tracks
    input_state_dim = config.state_dim * config.num_tracks
    holo = HoloLink(config, input_state_dim=input_state_dim)

    x = torch.randn(batch_size, config.d_model)
    h = torch.randn(batch_size, input_state_dim)
    m_prev = None

    retrieved, m_next = holo(x, h, m_prev)

    assert retrieved.shape == (batch_size, config.d_model)
    assert m_next.shape == (batch_size, holo.key_dim, config.d_model)

def test_hololink_decay():
    config = ANAConfig(hololink_decay=0.5)
    holo = HoloLink(config, input_state_dim=32)

    assert holo.decay == 0.5

    batch_size = 1
    x = torch.randn(batch_size, config.d_model)
    h = torch.zeros(batch_size, 32) # Zero input state -> zero update

    # M_prev is all ones
    M_prev = torch.ones(batch_size, holo.key_dim, config.d_model)

    # Zero update means M_t = 0.5 * M_prev + 0
    _, M_next = holo(x, h, M_prev)

    assert torch.allclose(M_next, M_prev * 0.5)

def test_hololink_ortho_init():
    config = ANAConfig(orthogonal_init=True)
    holo = HoloLink(config, input_state_dim=32)
    assert isinstance(holo.k_proj.weight, torch.Tensor)

def test_anamodel_forward(config):
    batch_size = 2
    seq_len = 5
    model = ANAModel(config)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, info_log = model(input_ids, return_info=True)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    # Check info log structure
    if len(info_log) > 0:
        assert 'ga_A' in info_log[0]
        # 'ret_gate' might be there if holo enabled

def test_anamodel_loss(config):
    batch_size = 2
    seq_len = 5
    model = ANAModel(config)
    criterion = nn.CrossEntropyLoss()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, _ = model(input_ids)

    loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

    assert loss > 0

    # Backward pass
    loss.backward()

    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
