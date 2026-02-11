#!/usr/bin/env python3
"""
ANA v2: Adaptive Neural Automaton

THE BEAST - A savage self-bootstrapping SSM that recursively rewrites its own
program stack with differentiable Gumbel-Softmax opcodes, fault-trace holographic
memory, and a live interpreter brutally executing ops across parallel tracks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass
class ANAConfig:
    """Configuration for ANA v2."""
    d_model: int = 64
    vocab_size: int = 50
    
    track_dims: Tuple[int, int, int] = (32, 64, 32)
    stack_depth: int = 5
    stack_dim: int = 32
    
    num_opcodes: int = 4
    num_tracks: int = 3
    
    gumbel_temp_init: float = 1.0
    gumbel_temp_min: float = 0.01
    gumbel_decay: float = 0.9995
    
    num_layers: int = 2
    
    @property
    def total_track_dim(self):
        return sum(self.track_dims)


class GumbelSoftmax:
    """Differentiable discrete choice."""
    
    @staticmethod
    def sample(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
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


class HolographicMemory(nn.Module):
    """VSA memory using FFT circular convolution."""
    
    def __init__(self, dim: int, max_capacity: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_capacity = max_capacity
        self.fft_dim = dim // 2 + 1
        
        self.register_buffer('memory', torch.zeros(max_capacity, self.fft_dim, dtype=torch.complex64))
        self.register_buffer('keys', torch.zeros(max_capacity, dim))
        self.register_buffer('usage', torch.zeros(max_capacity))
        
        self.write_idx = 0
    
    def bind(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        key_fft = torch.fft.rfft(key, dim=-1, n=self.dim)
        value_fft = torch.fft.rfft(value, dim=-1, n=self.dim)
        return torch.fft.irfft(key_fft * value_fft, n=self.dim)
    
    def unbind(self, query: torch.Tensor, memory_chunk: torch.Tensor) -> torch.Tensor:
        query_fft = torch.fft.rfft(query, dim=-1, n=self.dim)
        if memory_chunk.is_complex():
            unbound = torch.conj(query_fft) * memory_chunk
        else:
            mem_fft = torch.fft.rfft(memory_chunk, dim=-1, n=self.dim)
            unbound = torch.conj(query_fft) * mem_fft
        return torch.fft.irfft(unbound, n=self.dim)
    
    def write(self, key: torch.Tensor, value: torch.Tensor):
        bound = self.bind(key, value)
        bound_fft = torch.fft.rfft(bound, dim=-1, n=self.dim)
        
        idx = self.write_idx % self.max_capacity
        self.memory[idx] = bound_fft.squeeze(0)[:self.fft_dim]
        self.keys[idx] = key.squeeze(0)
        self.usage[idx] = 1.0
        self.write_idx += 1
    
    def read(self, query: torch.Tensor) -> torch.Tensor:
        batch = query.shape[0]
        results = []
        
        for b in range(batch):
            query_b = query[b:b+1]
            best_match, best_score = None, float('-inf')
            
            for i in range(min(self.write_idx, self.max_capacity)):
                if self.usage[i] > 0:
                    key_i = self.keys[i:i+1]
                    score = (query_b * key_i).sum() / (query_b.norm() * key_i.norm() + 1e-8)
                    if score > best_score:
                        best_score, best_match = score, i
            
            if best_match is not None:
                mem = self.memory[best_match:best_match+1]
                results.append(self.unbind(query_b, torch.fft.irfft(mem, n=self.dim)))
            else:
                results.append(torch.zeros(1, self.dim, device=query.device))
        
        return torch.cat(results, dim=0)
    
    def reset(self):
        self.write_idx = 0
        self.memory.zero_()
        self.keys.zero_()
        self.usage.zero_()


class ProgramStack:
    """LIFO stack for program frames."""
    
    def __init__(self, dim: int, max_depth: int):
        self.dim = dim
        self.max_depth = max_depth
        self.frames: List[Dict] = []
    
    def push(self, state: torch.Tensor) -> bool:
        if len(self.frames) >= self.max_depth:
            return False
        self.frames.append({'state': state.clone()})
        return True
    
    def pop(self) -> Optional[Dict]:
        return self.frames.pop() if self.frames else None
    
    def depth(self) -> int:
        return len(self.frames)
    
    def reset(self):
        self.frames = []


class Interpreter(nn.Module):
    """Executes opcodes on the model state."""
    
    OPCODE_NAMES = ['PUSH', 'POP', 'BIND', 'CALL']
    
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.temperature = config.gumbel_temp_init
        self.temp_decay = config.gumbel_decay
        
        self.memory_proj = nn.Linear(config.d_model, config.stack_dim)
    
    def update_temperature(self):
        self.temperature = max(self.config.gumbel_temp_min, self.temperature * self.temp_decay)
    
    def execute(self, x: torch.Tensor, opcode_logits: torch.Tensor,
                stack: ProgramStack, hologram: HolographicMemory,
                h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        batch_size = x.shape[0]
        num_tracks = self.config.num_tracks
        
        opcode = GumbelSoftmax.sample(opcode_logits, self.temperature)
        self.update_temperature()
        
        alpha_mods = torch.zeros(batch_size, num_tracks, device=x.device)
        beta_mods = torch.zeros(batch_size, num_tracks, device=x.device)
        
        dominant_op = opcode.argmax(dim=-1)
        
        for b in range(batch_size):
            op = dominant_op[b].item()
            
            if op == 0:  # PUSH
                if stack.push(h_prev[b:b+1]):
                    alpha_mods[b, 1] = 1.0
            elif op == 1:  # POP
                frame = stack.pop()
                if frame is not None:
                    beta_mods[b, 0] = 1.0
                    h_prev[b] = h_prev[b] + frame['state'].view(-1)
            elif op == 2:  # BIND
                hologram.write(x[b:b+1], x[b:b+1])
            elif op == 3:  # CALL
                stack.push(h_prev[b:b+1])
                alpha_mods[b, 2] = 1.0
        
        retrieved = self.memory_proj(hologram.read(x))
        h_next = h_prev + 0.1 * retrieved
        
        return alpha_mods, beta_mods, h_next, {
            'opcode': opcode,
            'stack_depth': stack.depth(),
            'temperature': self.temperature
        }


class LinearRecurrentTrack(nn.Module):
    """Core SSM: h_t = α_t · h_{t-1} + β_t · x_t"""
    
    def __init__(self, input_dim: int, state_dim: int, decay_init: float = -3.0):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.output_proj = nn.Linear(state_dim, input_dim)
        
        self.alpha_logit = nn.Parameter(torch.full((state_dim,), decay_init))
        self.beta_logit = nn.Parameter(torch.full((state_dim,), 0.0))
    
    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None,
                alpha_mod: Optional[torch.Tensor] = None,
                beta_mod: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            return self._step(x, h_prev, alpha_mod, beta_mod)
        return self._sequence(x, alpha_mod, beta_mod)
    
    def _step(self, x: torch.Tensor, h_prev: Optional[torch.Tensor],
              alpha_mod: Optional[torch.Tensor], beta_mod: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        if h_prev is None:
            h_prev = torch.zeros(batch, self.state_dim, device=x.device)
        
        u = self.input_proj(x)
        
        alpha = torch.sigmoid(self.alpha_logit).unsqueeze(0).expand(batch, -1)
        beta = torch.sigmoid(self.beta_logit).unsqueeze(0).expand(batch, -1)
        
        if alpha_mod is not None:
            alpha = torch.sigmoid(alpha + alpha_mod)
        if beta_mod is not None:
            beta = torch.sigmoid(beta + beta_mod)
        
        h = alpha * h_prev + beta * u
        y = self.output_proj(h)
        return y, h
    
    def _sequence(self, x: torch.Tensor, alpha_mod: Optional[torch.Tensor],
                  beta_mod: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        u = self.input_proj(x)
        
        alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1).expand(batch, seq, -1)
        beta = torch.sigmoid(self.beta_logit).view(1, 1, -1).expand(batch, seq, -1)
        
        if alpha_mod is not None:
            alpha = torch.sigmoid(alpha + alpha_mod.unsqueeze(-1))
        if beta_mod is not None:
            beta = torch.sigmoid(beta + beta_mod.unsqueeze(-1))
        
        h = torch.zeros_like(u)
        h[:, 0] = beta[:, 0] * u[:, 0]
        for t in range(1, seq):
            h[:, t] = alpha[:, t] * h[:, t-1] + beta[:, t] * u[:, t]
        
        y = self.output_proj(h)
        return y, h[:, -1]


class ANALayer(nn.Module):
    """Complete ANA layer: Interpreter + Parallel Tracks + Holographic Memory."""
    
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        
        fast_dim, slow_dim, logic_dim = config.track_dims
        
        self.tracks = nn.ModuleList([
            LinearRecurrentTrack(config.d_model, fast_dim, decay_init=-5.0),
            LinearRecurrentTrack(config.d_model, slow_dim, decay_init=0.5),
            LinearRecurrentTrack(config.d_model, logic_dim, decay_init=-2.0)
        ])
        
        self.hologram = HolographicMemory(config.d_model)
        self.stack = ProgramStack(config.stack_dim, config.stack_depth)
        
        self.interpreter = Interpreter(config)
        self.opcode_head = nn.Linear(config.d_model, config.num_opcodes)
        
        # Each track outputs d_model, so mixer input is d_model * num_tracks
        self.mixer = nn.Linear(config.d_model * config.num_tracks, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor, track_states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        batch, seq, _ = x.shape
        
        if track_states is None:
            track_states = [None] * self.config.num_tracks
        
        track_outputs = []
        new_states = [None] * self.config.num_tracks
        all_info = []
        
        for t in range(seq):
            xt = x[:, t, :]
            
            opcode_logits = self.opcode_head(xt)
            h_stack = torch.zeros(batch, self.config.stack_dim, device=xt.device)
            
            alpha_mods, beta_mods, h_stack, info = self.interpreter.execute(
                xt, opcode_logits, self.stack, self.hologram, h_stack
            )
            
            track_out = []
            for i, track in enumerate(self.tracks):
                y, h = track._step(xt, track_states[i], alpha_mods[:, i:i+1], beta_mods[:, i:i+1])
                track_out.append(y)
                new_states[i] = h
            
            track_out_cat = torch.cat(track_out, dim=-1)
            layer_out = self.mixer(track_out_cat)
            
            xt = xt + layer_out
            track_outputs.append(xt)
            all_info.append(info)
            
            track_states = list(new_states)
        
        output = torch.stack(track_outputs, dim=1)
        output = self.norm(output)
        
        return output, new_states, all_info
    
    def reset_state(self):
        self.stack.reset()
        self.hologram.reset()


class ANAModel(nn.Module):
    """ANA v2 - The complete model."""
    
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            ANALayer(config) for _ in range(config.num_layers)
        ])
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        layer_states = None
        for layer in self.layers:
            x, layer_states, _ = layer(x, layer_states)
        
        logits = self.output_head(x)
        return logits
    
    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
