import pytest
import torch
import torch.nn as nn
from ana.config import ANAConfig
from ana.models import ANAModel, BaselineSSM, LinearRecurrentUnit, HyperController, HoloLink

@pytest.fixture
def config():
    return ANAConfig(d_model=32, state_dim=16, num_layers=2, vocab_size=20, track_count=2)

@pytest.fixture
def config_parallel():
    return ANAConfig(d_model=32, state_dim=16, num_layers=2, vocab_size=20, use_parallel_scan=True)

class TestLinearRecurrentUnit:
    def test_step_forward(self, config):
        batch_size = 4
        lru = LinearRecurrentUnit(config)
        
        x = torch.randn(batch_size, config.d_model)
        h_prev = torch.zeros(batch_size, config.state_dim)
        
        y, h_next = lru(x, h_prev)
        
        assert y.shape == (batch_size, config.d_model)
        assert h_next.shape == (batch_size, config.state_dim)
    
    def test_sequence_forward(self, config):
        batch_size = 4
        seq_len = 10
        lru = LinearRecurrentUnit(config)
        
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        y, h_seq = lru(x)
        
        assert y.shape == (batch_size, seq_len, config.d_model)
        assert h_seq.shape == (batch_size, seq_len, config.state_dim)
    
    def test_dynamic_gates(self, config):
        batch_size = 4
        seq_len = 10
        lru = LinearRecurrentUnit(config)
        
        x = torch.randn(batch_size, seq_len, config.d_model)
        g_alpha = torch.randn(batch_size, seq_len, 1)
        g_beta = torch.randn(batch_size, seq_len, 1)
        
        y, h_seq = lru(x, dynamic_gates=(g_alpha, g_beta))
        
        assert y.shape == (batch_size, seq_len, config.d_model)
    
    def test_parallel_vs_sequential(self, config, config_parallel):
        batch_size = 2
        seq_len = 8
        
        torch.manual_seed(42)
        lru_seq = LinearRecurrentUnit(config)
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        torch.manual_seed(42)
        lru_par = LinearRecurrentUnit(config_parallel)
        x_par = x.clone()
        
        y_seq, _ = lru_seq(x)
        y_par, _ = lru_par(x_par)
        
        assert torch.allclose(y_seq, y_par, atol=1e-4)

class TestHyperController:
    def test_forward(self, config):
        batch_size = 4
        ctl = HyperController(config)
        
        x = torch.randn(batch_size, config.d_model)
        
        track_outputs, g_ret, g_halt = ctl(x)
        
        assert len(track_outputs) == config.track_count
        alpha, beta, mix = track_outputs[0]
        assert alpha.shape == (batch_size, 1)
        assert beta.shape == (batch_size, 1)
        assert mix.shape == (batch_size, 1)
        assert g_ret.shape == (batch_size, 1)
        assert g_halt.shape == (batch_size, 1)
    
    def test_force_prob(self, config):
        batch_size = 4
        ctl = HyperController(config)
        ctl.train()
        
        x = torch.randn(batch_size, config.d_model)
        
        _, g_ret_normal, _ = ctl(x, force_prob=0.0)
        _, g_ret_forced, _ = ctl(x, force_prob=1.0)
        
        assert torch.allclose(g_ret_forced, torch.ones_like(g_ret_forced) * 5.0, atol=1e-5)
    
    def test_controller_depth(self, config):
        config_deep = ANAConfig(d_model=32, controller_hidden_dim=16, controller_layers=3)
        ctl = HyperController(config_deep)
        
        expected_layers = 2 + (config_deep.controller_layers - 1) * 2
        assert len(ctl.net) == expected_layers

class TestHoloLink:
    def test_forward(self, config):
        batch_size = 4
        input_dim = config.state_dim * config.track_count
        holo = HoloLink(config, input_dim=input_dim)
        
        x = torch.randn(batch_size, config.d_model)
        h = torch.randn(batch_size, input_dim)
        
        retrieved, M_next = holo(x, h, None)
        
        assert retrieved.shape == (batch_size, config.d_model)
        assert M_next.shape == (batch_size, config.key_dim, config.d_model)
    
    def test_decay(self, config):
        config_decay = ANAConfig(hololink_decay=0.5, key_dim=16, d_model=32)
        holo = HoloLink(config_decay, input_dim=32)
        
        batch_size = 1
        x = torch.randn(batch_size, 32)
        h = torch.zeros(batch_size, 32)
        M_prev = torch.ones(batch_size, 16, 32)
        
        _, M_next = holo(x, h, M_prev)
        
        assert torch.allclose(M_next, M_prev * 0.5, atol=1e-5)
    
    def test_sequence_forward(self, config):
        batch_size = 4
        seq_len = 10
        input_dim = config.state_dim * config.track_count
        holo = HoloLink(config, input_dim=input_dim)
        
        x = torch.randn(batch_size, seq_len, config.d_model)
        h = torch.randn(batch_size, seq_len, input_dim)
        
        retrieved, M_seq = holo.forward_sequence(x, h)
        
        assert retrieved.shape == (batch_size, seq_len, config.d_model)
    
    def test_learned_binding(self, config):
        config_binding = ANAConfig(use_learned_binding=True, key_dim=16, d_model=32)
        holo = HoloLink(config_binding, input_dim=32)
        
        assert hasattr(holo, 'binding_strength')
        assert isinstance(holo.binding_strength, nn.Parameter)

class TestANAModel:
    def test_forward_sequential(self, config):
        batch_size = 2
        seq_len = 5
        model = ANAModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, info = model(input_ids, return_info=True)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert isinstance(info, list)
    
    def test_forward_parallel(self, config_parallel):
        batch_size = 2
        seq_len = 5
        model = ANAModel(config_parallel)
        
        input_ids = torch.randint(0, config_parallel.vocab_size, (batch_size, seq_len))
        
        logits, info = model(input_ids, return_info=True)
        
        assert logits.shape == (batch_size, seq_len, config_parallel.vocab_size)
    
    def test_ablation_no_controller(self, config):
        config_no_ctl = ANAConfig(d_model=32, state_dim=16, num_layers=2, vocab_size=20, use_controller=False)
        model = ANAModel(config_no_ctl)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)
    
    def test_ablation_no_hololink(self, config):
        config_no_holo = ANAConfig(d_model=32, state_dim=16, num_layers=2, vocab_size=20, use_hololink=False)
        model = ANAModel(config_no_holo)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)
    
    def test_variable_track_count(self, config):
        config_3tracks = ANAConfig(d_model=32, state_dim=16, num_layers=2, vocab_size=20, track_count=3)
        model = ANAModel(config_3tracks)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, _ = model(input_ids)
        
        assert logits.shape == (2, 5, 20)
    
    def test_thinking_steps(self, config):
        config_think = ANAConfig(d_model=32, state_dim=16, num_layers=2, vocab_size=20, max_thinking_steps=2, use_parallel_scan=False)
        model = ANAModel(config_think)
        
        input_ids = torch.randint(0, 20, (2, 5))
        logits, info = model(input_ids, return_info=True)
        
        assert logits.shape == (2, 5, 20)
    
    def test_backward_pass(self, config):
        batch_size = 2
        seq_len = 5
        model = ANAModel(config)
        criterion = nn.CrossEntropyLoss()
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, _ = model(input_ids)
        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
        
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

class TestBaselineSSM:
    def test_forward(self, config):
        batch_size = 2
        seq_len = 5
        model = BaselineSSM(config)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, info = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert info == []
    
    def test_forward_parallel(self, config_parallel):
        batch_size = 2
        seq_len = 5
        model = BaselineSSM(config_parallel)
        
        input_ids = torch.randint(0, config_parallel.vocab_size, (batch_size, seq_len))
        logits, _ = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config_parallel.vocab_size)

class TestConfig:
    def test_default_config(self):
        config = ANAConfig()
        
        assert config.d_model == 64
        assert config.state_dim == 64
        assert config.num_layers == 2
        assert config.track_count == 2
        assert config.use_hololink == True
        assert config.use_controller == True
    
    def test_custom_config(self):
        config = ANAConfig(d_model=128, state_dim=256, num_layers=4, track_count=3)
        
        assert config.d_model == 128
        assert config.state_dim == 256
        assert config.num_layers == 4
        assert config.track_count == 3
