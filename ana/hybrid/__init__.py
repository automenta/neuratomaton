"""
Hybrid ANA-Transformer Architecture with Learnable Routing

This module implements a hybrid architecture that combines:
- ANA (Adaptive Neural Automaton) for associative memory
- Transformer attention for pattern matching
- Learnable router to select optimal processing per token

Research Questions:
1. Can learned routing select optimal processing per token?
2. Does hybrid beat both pure ANA and pure Transformer?
3. What patterns emerge in routing decisions?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
import numpy as np

from ..config_v2 import ANAv2Config
from ..models_v3 import (
    GumbelSoftmax, SpecializedTracks, FaultTraceBuffer, 
    CortexController, MetaStateStack
)
from ..model_v3 import ANAv2Model


class LearnableRouter(nn.Module):
    def __init__(self, d_model: int, num_routes: int = 2, 
                 initial_temp: float = 2.0, min_temp: float = 0.1,
                 decay_steps: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.num_routes = num_routes
        
        self.router = nn.Linear(d_model, num_routes)
        
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.decay_steps = decay_steps
        self.global_step = 0
        
        self.route_history = []
        self.entropy_history = []
        
    def forward(self, x: torch.Tensor, hard: bool = True) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        logits = self.router(x)
        
        route_weights = GumbelSoftmax.sample(logits, self.temperature, hard=hard)
        
        entropy = -(route_weights * torch.log(route_weights + 1e-10)).sum(dim=-1)
        
        if self.training:
            self.global_step += 1
            progress = min(self.global_step / self.decay_steps, 1.0)
            self.temperature = self.initial_temp * (1 - progress) + self.min_temp * progress
        
        if not self.training:
            self.route_history.append(route_weights.detach().cpu())
            self.entropy_history.append(entropy.detach().cpu())
        
        return route_weights
    
    def get_routing_stats(self) -> Dict:
        if len(self.route_history) == 0:
            return {}
        
        route_weights = torch.cat(self.route_history, dim=0)
        
        route_usage = route_weights.mean(dim=(0, 1)).tolist()
        route_entropy = torch.cat(self.entropy_history).mean().item()
        
        return {
            'route_usage': route_usage,
            'mean_entropy': route_entropy,
            'temperature': self.temperature
        }
    
    def reset_history(self):
        self.route_history = []
        self.entropy_history = []


class ANABranch(nn.Module):
    def __init__(self, config: ANAv2Config):
        super().__init__()
        self.config = config
        
        self.tracks = SpecializedTracks(config)
        self.fault_buffer = FaultTraceBuffer(config)
        self.cortex = CortexController(config)
        self.stack = MetaStateStack(config)
        
        self.mixer = nn.Linear(config.total_track_dim, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        track_states = {'syntax': None, 'semantic': None, 'logic': None}
        outputs = []
        
        fault_summary = self.fault_buffer.get_summary()
        if fault_summary.size(0) != batch_size:
            fault_summary = fault_summary.expand(batch_size, -1)
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            cortex_out = self.cortex(xt, torch.zeros_like(xt[:, :self.config.stack_dim]), fault_summary)
            stack_result = self.stack(xt, fault_summary, [[]])
            
            track_out, new_track_states = self.tracks(
                xt,
                h_syntax=track_states['syntax'],
                h_semantic=track_states['semantic'],
                h_logic=track_states['logic'],
                alpha_mods=cortex_out.get('alpha_mods'),
                beta_mods=cortex_out.get('beta_mods')
            )
            track_states = new_track_states
            
            layer_out = self.mixer(track_out)
            out = self.norm(xt + layer_out)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


class TransformerBranch(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward or d_model * 4
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class HybridANATransformer(nn.Module):
    def __init__(self, config: ANAv2Config, num_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if config.use_position_encoding:
            self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(config.max_seq_len, config.d_model))
        
        self.ana_branch = ANABranch(config)
        self.transformer_branch = TransformerBranch(
            d_model=config.d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=config.d_model * 4
        )
        
        self.router = LearnableRouter(config.d_model, num_routes=2)
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _add_position_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.use_position_encoding:
            return x
        seq_len = x.size(1)
        return x + self.pos_encoding[:, :seq_len, :]
    
    def _init_weights(self):
        for module in [self.ana_branch.mixer, self.output_head]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
            if hasattr(module, 'bias'):
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, return_routing: bool = False) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        
        ana_output = self.ana_branch(x)
        transformer_output = self.transformer_branch(x)
        
        route_weights = self.router(x, hard=True)
        
        combined = route_weights[:, :, 0:1] * ana_output + route_weights[:, :, 1:2] * transformer_output
        
        logits = self.output_head(combined)
        
        if return_routing:
            return logits, route_weights
        
        return logits
    
    def get_routing_stats(self) -> Dict:
        return self.router.get_routing_stats()
    
    def analyze_routing(self, input_ids: torch.Tensor, targets: torch.Tensor) -> Dict:
        with torch.no_grad():
            logits, route_weights = self.forward(input_ids, return_routing=True)
            
            predictions = logits.argmax(-1)
            errors = (predictions != targets).float()
            
            error_by_route = []
            for r in range(route_weights.size(-1)):
                route_error = (errors * route_weights[:, :, r]).sum() / (route_weights[:, :, r].sum() + 1e-10)
                error_by_route.append(route_error.item())
            
            position_usage = route_weights.mean(dim=0).mean(dim=0).tolist()
            
            return {
                'error_by_route': error_by_route,
                'position_usage': position_usage,
                'route_weights_std': route_weights.std(dim=(0, 1)).tolist(),
                'mean_error_rate': errors.mean().item()
            }


class HybridWithSpecialization(HybridANATransformer):
    def __init__(self, config: ANAv2Config, num_layers: int = 4, nhead: int = 8,
                 specialize_positions: Optional[List[int]] = None):
        super().__init__(config, num_layers, nhead)
        
        self.specialize_positions = specialize_positions or []
        
        if len(self.specialize_positions) > 0:
            self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
            nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, return_routing: bool = False) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        
        ana_output = self.ana_branch(x)
        transformer_output = self.transformer_branch(x)
        
        route_weights = self.router(x, hard=False)
        
        if len(self.specialize_positions) > 0:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            pos_bias = self.position_embeddings(positions)
            route_logits = self.router.router(x) + pos_bias
            route_weights = F.softmax(route_logits / self.router.temperature, dim=-1)
        
        combined = route_weights[:, :, 0:1] * ana_output + route_weights[:, :, 1:2] * transformer_output
        
        if self.training:
            route_weights = GumbelSoftmax.sample(
                self.router.router(x), 
                self.router.temperature, 
                hard=True
            )
            combined = route_weights[:, :, 0:1] * ana_output + route_weights[:, :, 1:2] * transformer_output
        
        logits = self.output_head(combined)
        
        if return_routing:
            return logits, route_weights
        
        return logits


class MultiRouterHybrid(nn.Module):
    def __init__(self, config: ANAv2Config, num_routers: int = 3, 
                 num_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_routers = num_routers
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if config.use_position_encoding:
            self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(config.max_seq_len, config.d_model))
        
        self.ana_branch = ANABranch(config)
        self.transformer_branch = TransformerBranch(
            d_model=config.d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        self.routers = nn.ModuleList([
            LearnableRouter(config.d_model, num_routes=2)
            for _ in range(num_routers)
        ])
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _add_position_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.use_position_encoding:
            return x
        seq_len = x.size(1)
        return x + self.pos_encoding[:, :seq_len, :]
    
    def forward(self, input_ids: torch.Tensor, return_routing: bool = False) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        
        ana_output = self.ana_branch(x)
        transformer_output = self.transformer_branch(x)
        
        combined = x
        all_route_weights = []
        
        for router in self.routers:
            route_weights = router(combined, hard=True)
            combined = route_weights[:, :, 0:1] * ana_output + route_weights[:, :, 1:2] * transformer_output
            all_route_weights.append(route_weights)
        
        logits = self.output_head(combined)
        
        if return_routing:
            return logits, all_route_weights
        
        return logits


def create_hybrid_model(config: ANAv2Config, variant: str = 'standard', **kwargs):
    if variant == 'standard':
        return HybridANATransformer(config, **kwargs)
    elif variant == 'specialized':
        return HybridWithSpecialization(config, **kwargs)
    elif variant == 'multi_router':
        return MultiRouterHybrid(config, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def load_pretrained_ana(path: str, config: ANAv2Config) -> ANAv2Model:
    ana_model = ANAv2Model(config)
    ana_model.load_state_dict(torch.load(path))
    return ana_model


def load_pretrained_transformer(path: str, config: ANAv2Config) -> TransformerBranch:
    xf_model = TransformerBranch(config.d_model)
    xf_model.load_state_dict(torch.load(path))
    return xf_model
