"""
ANA v5: Clean Implementation with Working Memory

Simpler, cleaner approach:
1. Differentiable memory that works with parallel scan
2. Clear separation: SSM for position, Memory for content
3. Mode signals based on token type (hard-coded for now, learned later)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass  
class ANAv5Config:
    vocab_size: int = 60
    d_model: int = 64
    state_dim: int = 64
    num_layers: int = 1
    max_position: int = 8192
    use_parallel_scan: bool = True


class DifferentiableMemory(nn.Module):
    """
    Differentiable key-value memory using linear attention.
    This is essentially the working HoloLink but with explicit modes.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Projections
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable binding strength
        self.binding_strength = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, mode: torch.Tensor):
        """
        Args:
            x: [batch, seq, d_model] - query input
            h: [batch, seq, state_dim] - hidden state for key/value
            mode: [batch, seq] - 0=store, 1=retrieve, 2=ignore
        
        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Project keys and values from hidden state
        k = F.normalize(self.k_proj(h), dim=-1)  # [batch, seq, d]
        v = self.v_proj(h)  # [batch, seq, d]
        
        # Binding strength
        strength = F.softplus(self.binding_strength)
        
        # Build memory cumulatively (only store when mode==0)
        store_mask = (mode == 0).float().unsqueeze(-1).unsqueeze(-1)  # [batch, seq, 1, 1]
        updates = strength * store_mask * torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))  # [batch, seq, d, d]
        
        # Cumulative memory
        memory = torch.cumsum(updates, dim=1)  # [batch, seq, d, d]
        
        # Query (only when mode==1)
        query_mask = (mode == 1).float().unsqueeze(-1)  # [batch, seq, 1]
        q = F.normalize(self.q_proj(x), dim=-1)  # [batch, seq, d]
        
        # Retrieve from memory
        retrieved = torch.matmul(q.unsqueeze(-2), memory).squeeze(-2)  # [batch, seq, d]
        
        # Only use retrieved when in retrieve mode
        output = query_mask * retrieved
        
        return output


class AdaptiveSSM(nn.Module):
    """
    SSM with controllable dynamics.
    The "adaptation" comes from learned per-position alpha/beta.
    """
    
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        self.input_proj = nn.Linear(d_model, state_dim)
        self.output_proj = nn.Linear(state_dim, d_model)
        
        # Base parameters
        self.A_log = nn.Parameter(torch.randn(state_dim))
        self.B = nn.Parameter(torch.randn(state_dim) * 0.1)
        
        # Adaptive parameters (learned adjustments)
        self.delta_proj = nn.Linear(d_model, state_dim)  # Learned per-position delta
        
    def forward(self, x: torch.Tensor):
        """
        Parallel scan SSM with per-position adaptation.
        """
        batch, seq_len, _ = x.shape
        
        u = self.input_proj(x)
        
        # Base A and B
        A = -torch.exp(self.A_log)
        B = self.B
        
        # Per-position delta (adaptation)
        delta = F.softplus(self.delta_proj(x))  # [batch, seq, state_dim]
        
        # Simplified parallel scan
        # h_t = A * h_{t-1} + B * u_t * delta_t
        h = torch.zeros(batch, self.state_dim, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            h = A * h + B * u[:, t, :] * delta[:, t, :]
            outputs.append(self.output_proj(h))
        
        return torch.stack(outputs, dim=1)


class ANAv5(nn.Module):
    """
    ANA v5: Clean separation of concerns
    
    1. SSM: Handles position and local patterns
    2. Memory: Handles key-value associations
    3. Mode: Determines when to store/retrieve (starts with hard rules, learns to generalize)
    
    The "metaprogramming" aspect:
    - The mode signal programs behavior
    - Starts with explicit token-based rules
    - Can learn to generalize to new contexts
    """
    
    def __init__(self, config: ANAv5Config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = nn.Embedding(config.max_position, config.d_model)
        
        # Core components
        self.ssm = AdaptiveSSM(config.d_model, config.state_dim)
        self.memory = DifferentiableMemory(config.d_model)
        
        # Mode network: learns to predict store/retrieve
        self.mode_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 3)  # 3 modes
        )
        
        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Special token IDs
        self.TOK_KEY = 1
        self.TOK_VAL = 2
        self.TOK_QUERY = 3
        
    def forward(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.position_encoding(pos_ids)
        
        # Determine mode
        # Use hard rules during training to bootstrap, then learned during eval
        mode_logits = self.mode_net(x)  # [batch, seq, 3]
        
        if self.training:
            # Hard mode supervision based on token patterns
            # After TOK_KEY: next token is KEY content → STORE mode
            # After TOK_VAL: next token is VALUE content → still STORE (complete the pair)
            # After TOK_QUERY: next token is query key → RETRIEVE mode
            
            mode = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
            
            # Find special tokens
            is_after_key = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
            is_after_val = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
            is_after_query = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
            
            for t in range(1, seq_len):
                is_after_key[:, t] = (input_ids[:, t-1] == self.TOK_KEY)
                is_after_val[:, t] = (input_ids[:, t-1] == self.TOK_VAL)
                is_after_query[:, t] = (input_ids[:, t-1] == self.TOK_QUERY)
            
            # Set modes
            mode[is_after_key] = 0  # STORE (for key)
            mode[is_after_val] = 0  # STORE (for value)
            mode[is_after_query] = 1  # RETRIEVE
            # Default is 2 (ignore/process)
            
        else:
            # Use learned modes
            mode = mode_logits.argmax(dim=-1)
        
        # Process through SSM
        ssm_out = self.ssm(x)  # [batch, seq, d]
        
        # Process through memory
        memory_out = self.memory(x, ssm_out, mode)  # [batch, seq, d]
        
        # Combine: SSM always runs, memory adds retrieved content
        combined = x + ssm_out + memory_out
        
        # Output
        combined = self.norm(combined)
        logits = self.output_head(combined)
        
        return logits


if __name__ == "__main__":
    config = ANAv5Config()
    model = ANAv5(config)
    
    x = torch.randint(0, 60, (2, 32))
    logits = model(x)
    print(f"Input: {x.shape}, Output: {logits.shape}")
