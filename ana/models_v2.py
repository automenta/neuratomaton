"""
ANA v2: Enhanced architecture with external memory and attention mechanisms.

Key improvements:
1. External Memory Bank: Differentiable key-value storage
2. Selective Attention: Sparse attention for long-range dependencies  
3. Query-Gated Routing: Explicit memory access on query tokens
4. Multi-KV Training: Force memory utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ana.config import ANAConfig


class ExternalMemory(nn.Module):
    """Differentiable external memory bank for key-value storage."""
    
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.state_dim * 4  # Expand memory
        self.key_dim = config.key_dim
        
        # Memory projections
        self.key_proj = nn.Linear(config.d_model, self.key_dim)
        self.val_proj = nn.Linear(config.d_model, self.memory_size)
        self.query_proj = nn.Linear(config.d_model, self.key_dim)
        self.output_proj = nn.Linear(self.memory_size, config.d_model)
        
        # Gating for memory access
        self.write_gate = nn.Linear(config.d_model, 1)
        self.read_gate = nn.Linear(config.d_model, 1)
        
        self.scale = 1.0 / math.sqrt(self.key_dim)
        
    def forward(self, x, memory_state=None, force_read=False):
        """
        x: [batch, seq_len, d_model]
        memory_state: tuple of (keys, values) or None
        Returns: (output, new_memory_state)
        """
        batch, seq_len, _ = x.shape
        
        # Initialize memory if needed
        if memory_state is None:
            # Use learned initialization
            keys = torch.zeros(batch, 1, self.key_dim, device=x.device)
            values = torch.zeros(batch, 1, self.memory_size, device=x.device)
        else:
            keys, values = memory_state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Compute write gate
            w_gate = torch.sigmoid(self.write_gate(x_t))
            
            # Compute key and value
            k_t = self.key_proj(x_t)
            v_t = self.val_proj(x_t)
            k_t = F.normalize(k_t, dim=-1)
            
            # Write to memory (append)
            new_key = k_t.unsqueeze(1)
            new_val = v_t.unsqueeze(1) * w_gate.unsqueeze(-1)
            keys = torch.cat([keys, new_key], dim=1)
            values = torch.cat([values, new_val], dim=1)
            
            # Read from memory
            r_gate = torch.sigmoid(self.read_gate(x_t))
            if force_read and t == seq_len - 1:
                r_gate = torch.ones_like(r_gate)
            
            q_t = self.query_proj(x_t)
            q_t = F.normalize(q_t, dim=-1)
            
            # Attention over memory
            attn_scores = torch.bmm(q_t.unsqueeze(1), keys.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Weighted sum of values
            read_val = torch.bmm(attn_weights, values).squeeze(1)
            read_out = self.output_proj(read_val)
            
            # Gate the read output
            output_t = r_gate * read_out
            outputs.append(output_t)
        
        output = torch.stack(outputs, dim=1)
        return output, (keys, values)
    
    def forward_parallel(self, x, memory_state=None):
        """Parallel version for efficiency."""
        batch, seq_len, _ = x.shape
        
        # Project all at once
        keys = F.normalize(self.key_proj(x), dim=-1)  # [batch, seq, key_dim]
        values = self.val_proj(x)  # [batch, seq, memory_size]
        queries = F.normalize(self.query_proj(x), dim=-1)
        
        # Compute write gates
        w_gates = torch.sigmoid(self.write_gate(x))  # [batch, seq, 1]
        values = values * w_gates
        
        # Causal attention mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Compute attention scores
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Read from memory
        read_val = torch.bmm(attn_weights, values)
        output = self.output_proj(read_val)
        
        # Gate with read gates
        r_gates = torch.sigmoid(self.read_gate(x))
        output = r_gates * output
        
        return output, (keys, values)


class SelectiveAttention(nn.Module):
    """Sparse attention for long-range dependencies."""
    
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.num_heads = 4
        self.head_dim = config.d_model // self.num_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x):
        """Apply sparse causal attention."""
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Local attention window + sparse global
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attn_out)


class QueryGatedRouter(nn.Module):
    """Routes to memory on query tokens."""
    
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        
        # Learn query detection
        self.query_detector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, special_token_mask=None):
        """
        Returns routing weights for memory access.
        special_token_mask: [batch, seq_len] where 1 indicates query position
        """
        base_gate = self.query_detector(x)  # [batch, seq, 1]
        
        if special_token_mask is not None:
            # Boost gate at query positions
            boost = special_token_mask.unsqueeze(-1) * 0.5
            base_gate = torch.clamp(base_gate + boost, 0, 1)
        
        return base_gate


class ANAModelV2(nn.Module):
    """ANA v2 with external memory and selective attention."""
    
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if config.use_position_encoding:
            self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(512, config.d_model))
        
        # Core SSM tracks
        from ana.models import LinearRecurrentUnit, HyperController
        self.tracks = nn.ModuleList([
            LinearRecurrentUnit(config) for _ in range(config.track_count)
        ])
        
        # New components
        self.external_memory = ExternalMemory(config)
        self.selective_attention = SelectiveAttention(config)
        self.router = QueryGatedRouter(config)
        
        if config.use_controller:
            self.controller = HyperController(config)
        
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
    def _create_sinusoidal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _add_position_encoding(self, x):
        if not self.config.use_position_encoding:
            return x
        seq_len = x.size(1)
        return x + self.pos_encoding[:, :seq_len, :]
    
    def forward(self, input_ids, return_info=False, query_token_id=3):
        """
        query_token_id: ID of the query marker token (default 3)
        """
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        batch, seq_len, _ = x.shape
        
        # Detect query positions
        query_mask = (input_ids == query_token_id).float()  # [batch, seq]
        
        # SSM tracks
        track_outputs = []
        track_states = []
        
        controller_out = None
        if self.config.use_controller:
            controller_out = self.controller(x, force_prob=0.0)
        
        for i, track in enumerate(self.tracks):
            gates = None
            if controller_out is not None:
                g_alpha, g_beta, _ = controller_out[0][i]
                gates = (g_alpha, g_beta)
            
            yt, ht = track._forward_sequence(x, dynamic_gates=gates)
            track_outputs.append(yt)
            track_states.append(ht)
        
        # Mix tracks
        stacked_tracks = torch.stack(track_outputs, dim=-1)  # [batch, seq, d, tracks]
        track_out = stacked_tracks.mean(dim=-1)  # Simple mean for now
        
        # External memory
        memory_out, memory_state = self.external_memory.forward_parallel(x)
        
        # Selective attention
        attn_out = self.selective_attention(x)
        
        # Route based on query detection
        route_gate = self.router(x, query_mask)  # [batch, seq, 1]
        
        # At non-query positions: use SSM
        # At query positions: use memory heavily
        memory_weight = route_gate
        ssm_weight = 1 - route_gate
        
        # Combine outputs
        combined = ssm_weight * track_out + memory_weight * memory_out
        
        # Add attention contribution
        combined = combined + 0.1 * attn_out
        
        # Residual
        x = x + combined
        
        x = self.norm(x)
        logits = self.output_head(x)
        
        info = {'memory_weight': route_gate.mean().item()}
        return logits, [info]
