import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config_v2 import ANAv2Config


class GumbelSoftmax:
    @staticmethod
    def sample(logits, temperature=1.0, hard=True):
        if temperature == 0:
            return F.one_hot(logits.argmax(-1), logits.size(-1)).float()
        
        gumbels = -torch.empty_like(logits).exponential_().log()
        y = logits + gumbels
        y_soft = F.softmax(y / temperature, dim=-1)
        
        if hard:
            index = y_soft.argmax(dim=-1)
            y_hard = F.one_hot(index, logits.size(-1)).float()
            return (y_hard - y_soft).detach() + y_soft
        return y_soft


class StackFrame:
    def __init__(self, vector, opcode_logits, temperature=1.0):
        self.vector = vector
        self.opcode_logits = opcode_logits
        self.opcode = GumbelSoftmax.sample(opcode_logits, temperature)
        self.temperature = temperature


class MetaStateStack(nn.Module):
    def __init__(self, config: ANAv2Config):
        super().__init__()
        self.config = config
        self.gumbel_temp = config.gumbel_temp_init
        self.gumbel_temp_min = config.gumbel_temp_min
        self.gumbel_decay_steps = config.gumbel_decay_steps
        self.global_step = 0
        self.stack_dim = config.stack_dim
        self.max_depth = config.stack_depth
        self.num_opcodes = config.num_opcodes
        
        self.input_dim = config.d_model + config.stack_dim + config.d_model
        self.delta_proj = nn.Linear(self.input_dim, config.stack_dim)
        self.opcode_head = nn.Linear(self.input_dim, config.num_opcodes)
        
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.xavier_uniform_(self.opcode_head.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.opcode_head.bias)
        
        self.opcode_names = ['bind', 'gate', 'shift', 'recurse']
    
    def update_temperature(self):
        if self.global_step < self.gumbel_decay_steps:
            progress = self.global_step / self.gumbel_decay_steps
            self.gumbel_temp = self.config.gumbel_temp_init * (1 - progress) + self.gumbel_temp_min * progress
        self.global_step += 1
    
    def push(self, vector, opcode_logits, stack):
        if len(stack) >= self.max_depth:
            return stack, False
        frame = StackFrame(vector, opcode_logits, self.gumbel_temp)
        stack.append(frame)
        return stack, True
    
    def pop(self, stack):
        if len(stack) == 0:
            return stack, None
        return stack[:-1], stack[-1]
    
    def update_top(self, stack, delta_vector, new_opcode_logits):
        if len(stack) == 0:
            return stack
        frame = stack[-1]
        new_vector = frame.vector + delta_vector
        new_frame = StackFrame(new_vector, new_opcode_logits, self.gumbel_temp)
        stack[-1] = new_frame
        return stack
    
    def forward(self, x, fault_summary, stack, return_opcodes=False):
        batch_size = x.size(0)
        
        top_vector = x.new_zeros(batch_size, self.stack_dim)
        
        fault_summary_expanded = fault_summary
        if fault_summary.size(0) != batch_size:
            fault_summary_expanded = fault_summary.expand(batch_size, -1)
        
        combined = torch.cat([x, top_vector, fault_summary_expanded], dim=-1)
        delta = self.delta_proj(combined)
        opcode_logits = self.opcode_head(combined)
        
        self.update_temperature()
        
        opcodes = GumbelSoftmax.sample(opcode_logits, self.gumbel_temp)
        
        recurse_prob = opcodes[:, 3]
        should_recurse = recurse_prob > 0.5
        
        new_stack = []
        for b in range(batch_size):
            if len(stack) > b:
                frame_b = stack[b] if isinstance(stack[b], StackFrame) else None
                vec_b = frame_b.vector if frame_b is not None else top_vector[b]
            else:
                vec_b = top_vector[b]
            
            new_vec = vec_b + delta[b]
            new_frame = StackFrame(new_vec, opcode_logits[b], self.gumbel_temp)
            new_stack.append(new_frame)
        
        should_pop = len(new_stack) > 0 and torch.all(recurse_prob < 0.1)
        
        result = {
            'stack': new_stack,
            'opcodes': opcodes,
            'opcode_logits': opcode_logits,
            'recurse': should_recurse,
            'should_pop': should_pop
        }
        
        if return_opcodes:
            return result, opcodes
        return result
    
    def get_top_frame(self, stack):
        if len(stack) == 0:
            return None
        return stack[-1]
    
    def get_stack_depth(self, stack):
        return len(stack)


def parallel_scan_hillis_steele_v2(u, alpha, beta, h_init):
    batch, seq, dim = u.shape
    
    if seq <= 1:
        h = h_init.unsqueeze(1).expand(-1, seq, -1) * alpha + beta * u
        return h
    
    n = 1 << (seq - 1).bit_length()
    
    if n > seq:
        pad = n - seq
        alpha_pad = u.new_ones(batch, pad, dim)
        beta_pad = u.new_zeros(batch, pad, dim)
        u_pad = u.new_zeros(batch, pad, dim)
        alpha = torch.cat([alpha, alpha_pad], dim=1)
        beta = torch.cat([beta, beta_pad], dim=1)
        u = torch.cat([u, u_pad], dim=1)
        curr_a = alpha.clone()
        curr_b = beta.clone()
    else:
        curr_a = alpha.clone()
        curr_b = beta.clone()
    
    log_n = int(math.log2(n))
    
    for i in range(log_n):
        d = 1 << i
        a_shifted = torch.cat([u.new_ones(batch, d, dim), curr_a[:, :-d]], dim=1)
        b_shifted = torch.cat([u.new_zeros(batch, d, dim), curr_b[:, :-d]], dim=1)
        next_a = curr_a * a_shifted
        next_b = curr_a * b_shifted + curr_b
        curr_a = next_a
        curr_b = next_b
    
    h = curr_a[:, :seq] * h_init.unsqueeze(1) + curr_b[:, :seq]
    return h
    
    n = 1 << (seq - 1).bit_length()
    
    if n > seq:
        pad = n - seq
        alpha_pad = torch.ones(batch, pad, dim, device=u.device, dtype=u.dtype)
        beta_pad = torch.zeros(batch, pad, dim, device=u.device, dtype=u.dtype)
        u_pad = torch.zeros(batch, pad, dim, device=u.device, dtype=u.dtype)
        alpha = torch.cat([alpha, alpha_pad], dim=1)
        beta = torch.cat([beta, beta_pad], dim=1)
        u = torch.cat([u, u_pad], dim=1)
        curr_a = alpha.clone()
        curr_b = beta.clone()
        curr_u = u.clone()
    else:
        curr_a = alpha.clone()
        curr_b = beta.clone()
        curr_u = u.clone()
    
    log_n = int(math.log2(n))
    
    for i in range(log_n):
        d = 1 << i
        a_shifted = torch.cat([torch.ones(batch, d, dim, device=u.device, dtype=u.dtype), curr_a[:, :-d]], dim=1)
        b_shifted = torch.cat([torch.zeros(batch, d, dim, device=u.device, dtype=u.dtype), curr_b[:, :-d]], dim=1)
        next_a = curr_a * a_shifted
        next_b = curr_a * b_shifted + curr_b
        curr_a = next_a
        curr_b = next_b
    
    h = curr_a[:, :seq] * h_init.unsqueeze(1) + curr_b[:, :seq]
    return h


class LinearRecurrentTrack(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim=None, decay_init=-3.0):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.output_proj = nn.Linear(state_dim, self.output_dim)
        
        self.alpha_logit = nn.Parameter(torch.full((state_dim,), decay_init))
        self.beta_logit = nn.Parameter(torch.full((state_dim,), 0.0))
    
    def forward(self, x, h_prev=None, alpha_mod=None, beta_mod=None):
        if x.dim() == 2:
            return self._step(x, h_prev, alpha_mod, beta_mod)
        return self._sequence(x, alpha_mod, beta_mod)
    
    def _step(self, x, h_prev, alpha_mod, beta_mod):
        batch = x.size(0)
        if h_prev is None:
            h_prev = x.new_zeros(batch, self.state_dim)
        
        u = self.input_proj(x)
        
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        
        if alpha_mod is not None:
            alpha = torch.sigmoid(self.alpha_logit + alpha_mod.squeeze(-1).unsqueeze(1))
        if beta_mod is not None:
            beta = torch.sigmoid(self.beta_logit + beta_mod.squeeze(-1).unsqueeze(1))
        
        h = alpha * h_prev + beta * u
        y = self.output_proj(h)
        return y, h
    
    def _sequence(self, x, alpha_mod=None, beta_mod=None):
        batch, seq, _ = x.shape
        u = self.input_proj(x)
        
        alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1).expand(batch, seq, -1)
        beta = torch.sigmoid(self.beta_logit).view(1, 1, -1).expand(batch, seq, -1)
        
        h_init = x.new_zeros(batch, self.state_dim)
        h = parallel_scan_hillis_steele_v2(u, alpha, beta, h_init)
        y = self.output_proj(h)
        return y, h


class FaultTraceBuffer(nn.Module):
    def __init__(self, config: ANAv2Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.buffer_dim = config.fault_dim
        self.fault_dim = config.fault_dim
        self.max_size = config.fault_buffer_size
        self.threshold = config.fault_threshold
        
        fft_dim = self.buffer_dim // 2 + 1
        self.register_buffer('phase_keys', torch.randn(config.vocab_size, fft_dim))
        self.register_buffer('buffer', torch.zeros(1, config.vocab_size, fft_dim, dtype=torch.complex64))
        self.register_buffer('usage_counts', torch.zeros(config.vocab_size))
        
        self.summary_proj = nn.Linear(self.fault_dim, self.d_model)
        nn.init.xavier_uniform_(self.summary_proj.weight)
        nn.init.zeros_(self.summary_proj.bias)
    
    def holographic_bind(self, content, key):
        batch, dim = content.shape
        key_dim = key.shape[-1]
        
        content_fft = torch.fft.rfft(content, dim=-1, n=dim)
        key_fft = torch.fft.rfft(key, dim=-1, n=dim)
        
        key_phase = torch.angle(key_fft)
        
        bound = content_fft * torch.exp(1j * key_phase)
        return torch.fft.irfft(bound, n=dim)
    
    def holographic_retrieve(self, query, bound_vectors):
        batch, dim = query.shape
        
        query_fft = torch.fft.rfft(query, dim=-1, n=dim)
        
        if isinstance(bound_vectors, torch.Tensor) and bound_vectors.is_complex():
            retrieved = bound_vectors * torch.exp(-1j * torch.angle(query_fft))
        else:
            bound_fft = torch.fft.rfft(bound_vectors, dim=-1, n=dim)
            retrieved = bound_fft * torch.exp(-1j * torch.angle(query_fft))
        
        return torch.fft.irfft(retrieved, n=dim)
    
    def forward(self, error_vector, token_ids=None):
        batch = error_vector.size(0)
        
        if error_vector.dim() == 1:
            error_vector = error_vector.unsqueeze(0)
        
        error_norm = torch.norm(error_vector, dim=-1, keepdim=True)
        should_store = error_norm > self.threshold
        
        for b in range(batch):
            if should_store[b]:
                error_b = error_vector[b:b+1]
                
                if token_ids is not None and token_ids.numel() > 1:
                    key_id = token_ids[b].item() if b < token_ids.numel() else 0
                    key_id = key_id % self.config.vocab_size
                elif token_ids is not None and token_ids.numel() == 1:
                    key_id = token_ids.item() % self.config.vocab_size
                else:
                    key_id = 0
                
                phase_key = self.phase_keys[key_id:key_id+1]
                
                bound = self.holographic_bind(error_b, phase_key)
                
                bound_fft = torch.fft.rfft(bound, dim=-1, n=self.buffer_dim)
                current_fft = self.buffer[0, key_id:key_id+1]
                
                alpha = 0.1
                new_fft = current_fft + alpha * bound_fft
                self.buffer[0, key_id:key_id+1] = new_fft
                self.usage_counts[key_id] += 1
        
        avg_fft = self.buffer[0].mean(dim=0)
        retrieved = torch.fft.irfft(avg_fft, n=self.buffer_dim)
        
        summary = self.summary_proj(retrieved.unsqueeze(0))
        
        if batch > 1:
            summary = summary.expand(batch, -1)
        
        return summary
    
    def get_summary(self):
        avg_fft = self.buffer[0].mean(dim=0)
        retrieved = torch.fft.irfft(avg_fft, n=self.buffer_dim)
        summary = self.summary_proj(retrieved)
        return summary.unsqueeze(0)
    
    def reset(self):
        self.buffer.zero_()
        self.usage_counts.zero_()


class CortexController(nn.Module):
    def __init__(self, config: ANAv2Config):
        super().__init__()
        self.config = config
        cortex_input_dim = config.d_model + config.stack_dim + config.d_model
        hidden_dim = config.cortex_hidden_dim
        
        self.input_dim = cortex_input_dim
        
        layers = []
        layers.append(nn.Linear(cortex_input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(config.cortex_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        self.net = nn.Sequential(*layers)
        
        self.opcode_head = nn.Linear(hidden_dim, config.num_opcodes)
        self.delta_head = nn.Linear(hidden_dim, config.stack_dim)
        
        self.A_head = nn.Linear(hidden_dim, 3)
        self.B_head = nn.Linear(hidden_dim, 3)
        
        nn.init.xavier_uniform_(self.opcode_head.weight)
        nn.init.zeros_(self.opcode_head.bias)
        nn.init.xavier_uniform_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.xavier_uniform_(self.A_head.weight)
        nn.init.zeros_(self.A_head.bias)
        nn.init.xavier_uniform_(self.B_head.weight)
        nn.init.zeros_(self.B_head.bias)
    
    def forward(self, x, top_stack, fault_summary):
        batch_size = x.size(0)
        
        fault_summary_expanded = fault_summary
        if fault_summary.size(0) != batch_size:
            fault_summary_expanded = fault_summary.expand(batch_size, -1)
        
        combined = torch.cat([x, top_stack, fault_summary_expanded], dim=-1)
        features = self.net(combined)
        
        opcode_logits = self.opcode_head(features)
        delta = self.delta_head(features)
        
        alpha_mods = self.A_head(features).view(batch_size, 3)
        beta_mods = self.B_head(features).view(batch_size, 3)
        
        return {
            'opcode_logits': opcode_logits,
            'delta': delta,
            'alpha_mods': [alpha_mods[:, i:i+1] for i in range(3)],
            'beta_mods': [beta_mods[:, i:i+1] for i in range(3)]
        }


class SpecializedTracks(nn.Module):
    def __init__(self, config: ANAv2Config):
        super().__init__()
        self.config = config
        
        self.syntax_track = LinearRecurrentTrack(config.d_model, config.syntax_dim, config.syntax_dim, decay_init=-5.0)
        self.semantic_track = LinearRecurrentTrack(config.d_model, config.semantic_dim, config.semantic_dim, decay_init=1.0)
        self.logic_track = LinearRecurrentTrack(config.d_model, config.logic_dim, config.logic_dim, decay_init=-2.0)
    
    def forward(self, x, h_syntax=None, h_semantic=None, h_logic=None, 
                alpha_mods=None, beta_mods=None):
        y_syntax, h_syntax = self.syntax_track(x, h_syntax, 
            alpha_mods[0] if alpha_mods else None,
            beta_mods[0] if beta_mods else None)
        
        y_semantic, h_semantic = self.semantic_track(x, h_semantic,
            alpha_mods[1] if alpha_mods else None,
            beta_mods[1] if beta_mods else None)
        
        y_logic, h_logic = self.logic_track(x, h_logic,
            alpha_mods[2] if alpha_mods else None,
            beta_mods[2] if beta_mods else None)
        
        outputs = torch.cat([y_syntax, y_semantic, y_logic], dim=-1)
        
        states = {
            'syntax': h_syntax,
            'semantic': h_semantic,
            'logic': h_logic
        }
        
        return outputs, states
    
    def get_state_dim(self):
        return self.config.syntax_dim + self.config.semantic_dim + self.config.logic_dim
