import pytest
import torch
import torch.nn as nn
from ana.config_v2 import ANAv2Config
from ana.models_v3 import (
    GumbelSoftmax, StackFrame, MetaStateStack,
    LinearRecurrentTrack, SpecializedTracks,
    parallel_scan_hillis_steele_v2
)
from ana.model_v3 import (
    FaultTraceBuffer, CortexController, ANAv2Model,
    ANAv2Interpreter
)


@pytest.fixture
def config_v2():
    return ANAv2Config(
        d_model=128,
        vocab_size=30,
        syntax_dim=32,
        semantic_dim=64,
        logic_dim=32,
        stack_dim=32,
        fault_dim=128,
        cortex_hidden_dim=64,
        cortex_layers=2
    )


class TestGumbelSoftmax:
    def test_sample_shape(self, config_v2):
        batch_size = 4
        num_opcodes = 4
        logits = torch.randn(batch_size, num_opcodes)
        
        samples = GumbelSoftmax.sample(logits, temperature=1.0)
        
        assert samples.shape == (batch_size, num_opcodes)
        assert torch.allclose(samples.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    
    def test_hard_sample(self, config_v2):
        batch_size = 2
        num_opcodes = 3
        logits = torch.randn(batch_size, num_opcodes)
        
        samples = GumbelSoftmax.sample(logits, temperature=1.0, hard=True)
        
        for i in range(batch_size):
            assert samples[i].sum().item() == pytest.approx(1.0, abs=1e-5)
            assert torch.any(samples[i] == 1.0) or torch.any(samples[i] == 0.0)
    
    def test_zero_temperature(self, config_v2):
        batch_size = 4
        num_opcodes = 4
        logits = torch.randn(batch_size, num_opcodes)
        
        samples = GumbelSoftmax.sample(logits, temperature=0.0)
        
        for i in range(batch_size):
            assert samples[i].sum().item() == pytest.approx(1.0, abs=1e-5)
            max_idx = logits[i].argmax().item()
            assert samples[i, max_idx].item() == pytest.approx(1.0, abs=1e-5)


class TestStackFrame:
    def test_frame_creation(self, config_v2):
        vector = torch.randn(32)
        opcode_logits = torch.randn(4)
        
        frame = StackFrame(vector, opcode_logits, temperature=1.0)
        
        assert frame.vector.shape == (32,)
        assert frame.opcode_logits.shape == (4,)
        assert frame.temperature == 1.0
    
    def test_opcode_sample(self, config_v2):
        vector = torch.randn(32)
        opcode_logits = torch.randn(4)
        
        frame = StackFrame(vector, opcode_logits, temperature=1.0)
        
        assert frame.opcode.shape == (4,)
        assert torch.allclose(frame.opcode.sum(), torch.tensor(1.0), atol=1e-5)


class TestLinearRecurrentTrack:
    def test_step_forward(self, config_v2):
        input_dim = 32
        state_dim = 16
        
        track = LinearRecurrentTrack(input_dim, state_dim)
        
        x = torch.randn(4, input_dim)
        h_prev = torch.zeros(4, state_dim)
        
        y, h_next = track._step(x, h_prev, None, None)
        
        assert y.shape == (4, input_dim)
        assert h_next.shape == (4, state_dim)
    
    def test_sequence_forward(self, config_v2):
        input_dim = 32
        state_dim = 16
        
        track = LinearRecurrentTrack(input_dim, state_dim)
        
        x = torch.randn(4, 10, input_dim)
        
        y, h = track._sequence(x, None, None)
        
        assert y.shape == (4, 10, input_dim)
        assert h.shape == (4, 10, state_dim)
    
    def test_dynamic_mods(self, config_v2):
        input_dim = 32
        state_dim = 16
        
        track = LinearRecurrentTrack(input_dim, state_dim)
        
        x = torch.randn(4, 10, input_dim)
        alpha_mod = torch.randn(4, 10, 1)
        beta_mod = torch.randn(4, 10, 1)
        
        y, h = track._sequence(x, alpha_mod, beta_mod)
        
        assert y.shape == (4, 10, input_dim)


class TestSpecializedTracks:
    def test_tracks_forward(self, config_v2):
        tracks = SpecializedTracks(config_v2)
        
        x = torch.randn(4, 10, config_v2.d_model)
        
        y, states = tracks(x)
        
        assert y.shape == (4, 10, config_v2.total_track_dim)
        assert 'syntax' in states
        assert 'semantic' in states
        assert 'logic' in states
    
    def test_state_dim(self, config_v2):
        tracks = SpecializedTracks(config_v2)
        
        assert tracks.get_state_dim() == config_v2.total_track_dim


class TestFaultTraceBuffer:
    def test_init(self, config_v2):
        buffer = FaultTraceBuffer(config_v2)
        
        assert buffer.buffer_dim == config_v2.fault_dim
        assert buffer.max_size == config_v2.fault_buffer_size
    
    def test_store_error(self, config_v2):
        buffer = FaultTraceBuffer(config_v2)
        
        error = torch.randn(1, 128) * 5
        token_ids = torch.tensor([0])
        summary = buffer(error, token_ids)
        
        assert summary.shape == (1, config_v2.d_model)
    
    def test_batch_store(self, config_v2):
        buffer = FaultTraceBuffer(config_v2)
        
        errors = torch.randn(4, 128) * 5
        token_ids = torch.tensor([0, 1, 2, 3])
        summaries = buffer(errors, token_ids)
        
        assert summaries.shape == (4, config_v2.d_model)
    
    def test_get_summary(self, config_v2):
        buffer = FaultTraceBuffer(config_v2)
        
        error = torch.randn(1, 128) * 3
        token_ids = torch.tensor([0])
        buffer(error, token_ids)
        
        summary = buffer.get_summary()
        
        assert summary.shape == (1, config_v2.d_model)
    
    def test_reset(self, config_v2):
        buffer = FaultTraceBuffer(config_v2)
        
        error = torch.randn(1, 128) * 5
        token_ids = torch.tensor([0])
        buffer(error, token_ids)
        
        buffer.reset()
        
        assert torch.allclose(buffer.buffer, torch.zeros_like(buffer.buffer))


class TestCortexController:
    def test_forward(self, config_v2):
        cortex = CortexController(config_v2)
        
        batch_size = 4
        x = torch.randn(batch_size, config_v2.d_model)
        top_stack = torch.randn(batch_size, config_v2.stack_dim)
        fault_summary = torch.randn(batch_size, config_v2.fault_dim)
        
        result = cortex(x, top_stack, fault_summary)
        
        assert 'opcode_logits' in result
        assert 'delta' in result
        assert 'alpha_mods' in result
        assert 'beta_mods' in result
        
        assert result['opcode_logits'].shape == (batch_size,4)
        assert result['delta'].shape == (batch_size, config_v2.stack_dim)
        assert len(result['alpha_mods']) == 3
        assert len(result['beta_mods']) == 3


class TestMetaStateStack:
    def test_init(self, config_v2):
        stack = MetaStateStack(config_v2)
        
        assert stack.stack_dim == config_v2.stack_dim
        assert stack.max_depth == config_v2.stack_depth
        assert stack.num_opcodes == config_v2.num_opcodes
    
    def test_forward_empty_stack(self, config_v2):
        stack = MetaStateStack(config_v2)
        
        x = torch.randn(4, config_v2.d_model)
        fault_summary = torch.randn(4, config_v2.d_model)
        empty_stack = []
        
        result = stack(x, fault_summary, empty_stack)
        
        assert 'opcodes' in result
        assert 'stack' in result
        assert len(result['stack']) == 4
    
    def test_temperature_update(self, config_v2):
        stack = MetaStateStack(config_v2)
        
        initial_temp = stack.gumbel_temp
        
        for _ in range(100):
            stack.update_temperature()
        
        updated_temp = stack.gumbel_temp
        
        assert updated_temp < initial_temp


class TestANAv2Model:
    def test_forward(self, config_v2):
        model = ANAv2Model(config_v2)
        
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, config_v2.vocab_size, (batch_size, seq_len))
        
        logits, rule_logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config_v2.vocab_size)
        assert rule_logits.shape == (batch_size, seq_len, 2)
    
    def test_forward_with_info(self, config_v2):
        model = ANAv2Model(config_v2)
        
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, config_v2.vocab_size, (batch_size, seq_len))
        
        logits, rule_logits, info = model(input_ids, return_info=True)
        
        assert len(info) == seq_len
        assert 'opcode' in info[0]
        assert 'stack_depth' in info[0]
    
    def test_compute_loss(self, config_v2):
        model = ANAv2Model(config_v2)
        
        batch_size = 2
        seq_len = 5
        logits = torch.randn(batch_size, seq_len, config_v2.vocab_size)
        rule_logits = torch.randn(batch_size, seq_len, 2)
        targets = torch.randint(0, config_v2.vocab_size, (batch_size, seq_len))
        
        loss_dict = model.compute_loss(logits, rule_logits, targets)
        
        assert 'total' in loss_dict
        assert 'ce' in loss_dict
        assert 'rule' in loss_dict
        assert 'density' in loss_dict
        assert loss_dict['total'].requires_grad
    
    def test_backward_pass(self, config_v2):
        model = ANAv2Model(config_v2)
        
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, config_v2.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config_v2.vocab_size, (batch_size, seq_len))
        
        logits, rule_logits = model(input_ids)
        loss_dict = model.compute_loss(logits, rule_logits, targets)
        
        loss_dict['total'].backward()
        
        grad_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_params += 1
        
        assert grad_params > 0
    
    def test_parallel_forward(self, config_v2):
        model = ANAv2Model(config_v2)
        model.config.use_parallel_scan = True
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config_v2.vocab_size, (batch_size, seq_len))
        
        logits, rule_logits = model.forward_parallel(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config_v2.vocab_size)
        assert rule_logits.shape == (batch_size, seq_len, 2)
    
    def test_fault_buffer_update(self, config_v2):
        model = ANAv2Model(config_v2)
        
        batch_size = 2
        predictions = torch.randn(batch_size, config_v2.d_model)
        targets = torch.randn(batch_size, config_v2.d_model)
        token_ids = torch.randint(0, config_v2.vocab_size, (batch_size,))
        
        model.update_fault_buffer(predictions, targets, token_ids)
        
        assert torch.any(model.fault_buffer.usage_counts > 0)


class TestParallelScanV2:
    def test_scan_forward(self, config_v2):
        batch = 2
        seq = 8
        dim = 16
        
        u = torch.randn(batch, seq, dim)
        alpha = torch.sigmoid(torch.randn(batch, seq, 1)).expand(-1, -1, dim)
        beta = torch.sigmoid(torch.randn(batch, seq, 1)).expand(-1, -1, dim)
        h_init = torch.randn(batch, dim)
        
        h = parallel_scan_hillis_steele_v2(u, alpha, beta, h_init)
        
        assert h.shape == (batch, seq, dim)
    
    def test_scan_single_step(self, config_v2):
        batch = 2
        seq = 1
        dim = 16
        
        u = torch.randn(batch, seq, dim)
        alpha = torch.sigmoid(torch.randn(batch, seq, 1))
        beta = torch.sigmoid(torch.randn(batch, seq, 1))
        h_init = torch.randn(batch, dim)
        
        h = parallel_scan_hillis_steele_v2(u, alpha, beta, h_init)
        
        assert h.shape == (batch, seq, dim)
        assert torch.allclose(h, alpha * h_init.unsqueeze(1) + beta * u, atol=1e-4)


class TestConfigV2:
    def test_default_config(self):
        config = ANAv2Config()
        
        assert config.d_model == 128
        assert config.syntax_dim == 64
        assert config.semantic_dim == 128
        assert config.logic_dim == 64
        assert config.stack_depth == 5
    
    def test_custom_config(self):
        config = ANAv2Config(
            d_model=256,
            syntax_dim=128,
            semantic_dim=256,
            stack_depth=3
        )
        
        assert config.d_model == 256
        assert config.syntax_dim == 128
        assert config.semantic_dim == 256
        assert config.stack_depth == 3
    
    def test_total_track_dim(self):
        config = ANAv2Config(
            syntax_dim=32,
            semantic_dim=64,
            logic_dim=32
        )
        
        assert config.total_track_dim == 128
