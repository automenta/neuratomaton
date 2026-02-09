import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class LinearRecurrentUnit(nn.Module):
    """
    Diagonal SSM / RNN Unit.
    h_t = alpha * h_{t-1} + beta * x_t
    y_t = h_t
    """
    def __init__(self, d_input: int, d_state: int):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        
        # Projections
        self.in_proj = nn.Linear(d_input, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_input, bias=False)
        
        # Parametric Initialization placeholders (will be set by Controller logic)
        # We store base log-alphas/betas here if we wanted static, 
        # but in ANA they are dynamic. 
        # However, for stability, we learn a specific BASE parameter 
        # that the controller modulates.
        
        self.base_alpha_logit = nn.Parameter(torch.zeros(d_state)) # Sigmoid(base + delta)
        self.base_beta_logit = nn.Parameter(torch.zeros(d_state))
        
    def forward(self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor], 
                gates: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t: [Batch, D_input]
        gates: (g_alpha, g_beta) from Controller. [Batch, 1] or [Batch, D_state]
        """
        u_t = self.in_proj(x_t)
        
        g_alpha, g_beta = gates
        
        # Modulation: Base + Delta
        # Reshape gates to broadcast if they are scalar [B, 1]
        if g_alpha.dim() == 2 and g_alpha.size(1) == 1:
            g_alpha = g_alpha.expand(-1, self.d_state)
            g_beta = g_beta.expand(-1, self.d_state)
            
        alpha = torch.sigmoid(self.base_alpha_logit + g_alpha)
        beta = torch.sigmoid(self.base_beta_logit + g_beta)
        
        if h_prev is None:
            h_prev = torch.zeros_like(u_t)
            
        h_t = alpha * h_prev + beta * u_t
        y_t = self.out_proj(h_t)
        
        return y_t, h_t

class HyperController(nn.Module):
    """
    Low-Rank Meta-Controller.
    Predicts delta-gates for tracks and retrieval gate.
    """
    def __init__(self, d_model: int, d_ctrl: int = 64):
        super().__init__()
        # Input: x_t [D] (We could also input state summary, but let's stick to X for speed for now)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ctrl),
            nn.ReLU(),
            nn.Linear(d_ctrl, 6) # 4 for Tracks (A/B a/b), 1 Ret, 1 Copy
        )
        
    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.net(x_t) # [B, 6]
        
        # Return chunks
        # Track A: 0, 1
        # Track B: 2, 3
        # Ret: 4
        # Copy: 5
        return (out[:, 0:1], out[:, 1:2], 
                out[:, 2:3], out[:, 3:4], 
                out[:, 4:5], out[:, 5:6])

class HoloLink(nn.Module):
    """
    Associative Memory + Induction Head.
    """
    def __init__(self, d_model: int, state_dim: int, key_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.key_dim = key_dim
        self.d_model = d_model
        
        # 1. Associative Core (The "Holo")
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.k_proj = nn.Linear(state_dim, key_dim, bias=False) # Learned!
        self.v_proj = nn.Linear(state_dim, d_model, bias=False)
        
        # 2. Induction / Copy Head (Standard Attention)
        # A small attention window over recent inputs? 
        # Or just standard Self-Attention layer added?
        # Let's make it a standard MHA that attends to the "Memory State" 
        # (Wait, MHA needs sequence. Holo is state. 
        # We can simulate Induction by just having a separate standard Attention over X).
        # But we want O(1) state. 
        # Okay, for O(1) Copying, we need Linear Attention / Fast Weights.
        # The HoloLink IS that mechanism if K/V are set correctly.
        # We will keep HoloLink as the primary, but maybe add a classic "KV Cache" of size 32?
        # That breaks "Infinite Context" O(1).
        # Let's stick to just pure HoloLink for now, but maximize its capacity.
        
        self.norm = nn.LayerNorm(d_model) # Output norm

    def forward(self, x_t, h_t, M_prev):
        # M_prev: [B, K, D]
        if M_prev is None:
            bat = x_t.size(0)
            d_val = self.v_proj.out_features
            M_prev = torch.zeros(bat, self.key_dim, d_val, device=x_t.device)
            
        # Write
        k = F.normalize(self.k_proj(h_t), p=2, dim=-1) # [B, K]
        v = self.v_proj(h_t) # [B, D]
        
        update = torch.bmm(k.unsqueeze(2), v.unsqueeze(1)) # [B, K, 1] * [B, 1, D] -> [B, K, D]
        M_t = M_prev + update
        
        # Read
        q = F.normalize(self.q_proj(x_t), p=2, dim=-1) # [B, K]
        # [B, 1, K] * [B, K, D] -> [B, 1, D]
        read = torch.bmm(q.unsqueeze(1), M_t).squeeze(1)
        
        return read, M_t

class DualTrackBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        
        # Controller
        self.ctrl = HyperController(config.d_model, config.ctrl_dim)
        
        # Tracks
        self.track_A = LinearRecurrentUnit(config.d_model, config.d_state_A)
        self.track_B = LinearRecurrentUnit(config.d_model, config.d_state_B)
        
        # Holo
        self.holo = HoloLink(config.d_model, config.d_state_A + config.d_state_B, config.holo_key_dim)
        
        # Feed Forward
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )
        
    def forward(self, x, h_prev_A, h_prev_B, m_prev, return_info=False):
        # x: [B, D]
        res = x
        x = self.ln1(x)
        
        # 1. Control
        ga_A, gb_A, ga_B, gb_B, g_ret, g_copy = self.ctrl(x)
        
        # 2. Tracks
        y_A, h_next_A = self.track_A(x, h_prev_A, (ga_A, gb_A))
        y_B, h_next_B = self.track_B(x, h_prev_B, (ga_B, gb_B))
        
        # 3. Memory
        h_comb = torch.cat([h_next_A, h_next_B], dim=-1)
        mem_read, m_next = self.holo(x, h_comb, m_prev)
        
        # 4. Gate Memory
        gate = torch.sigmoid(g_ret)
        
        # Combine
        # Standard: A + B + Gate * Mem
        combined = y_A + y_B + gate * mem_read
        
        # Output
        out = res + combined
        
        # MLP
        res2 = out
        out = self.ln2(out)
        out = self.mlp(out)
        out = res2 + out
        
        return out, h_next_A, h_next_B, m_next, (gate if return_info else None)
