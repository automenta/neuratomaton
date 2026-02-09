
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearRecurrentUnit(nn.Module):
    """
    A simplified Linear Recurrent Unit (LRU) / SSM layer.
    h_t = alpha * h_{t-1} + beta * x_t
    y_t = h_t
    
    Alpha and Beta can be static (learned parameters) or dynamic (provided per step).
    """
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Projections for input x_t -> state_dim
        self.input_proj = nn.Linear(d_model, state_dim)
        
        # Output projection state_dim -> d_model
        self.output_proj = nn.Linear(state_dim, d_model)
        
        # Static parameters (used as baseline or base for dynamic)
        # alpha should be in [0, 1]. initialized to preserve history (high alpha)
        # beta should be in [0, 1]. initialized to accept some input
        self.static_alpha_logit = nn.Parameter(torch.Tensor(state_dim).uniform_(2, 4)) # sigmoid(2) ~= 0.88, sigmoid(4) ~= 0.98
        self.static_beta_logit = nn.Parameter(torch.Tensor(state_dim).uniform_(-2, 0)) # sigmoid(-2) ~= 0.12, sigmoid(0) = 0.5

    def forward(self, x, h_prev=None, dynamic_gates=None):
        """
        x: [batch, d_model]
        h_prev: [batch, state_dim]
        dynamic_gates: tuple (gate_alpha, gate_beta) from Controller
                       gate_alpha: [batch, state_dim]
                       gate_beta: [batch, state_dim]
        """
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.state_dim, device=x.device)
            
        u_t = self.input_proj(x) # [batch, state_dim]
        
        # Calculate alpha and beta
        if dynamic_gates is not None:
            gate_alpha, gate_beta = dynamic_gates
            # Modulation: sigmoid(static + dynamic)
            alpha = torch.sigmoid(self.static_alpha_logit + gate_alpha)
            beta = torch.sigmoid(self.static_beta_logit + gate_beta)
        else:
            alpha = torch.sigmoid(self.static_alpha_logit)
            beta = torch.sigmoid(self.static_beta_logit)
            
        # Evolution: h_t = alpha * h_{t-1} + beta * u_t
        h_t = alpha * h_prev + beta * u_t
        
        # Projection to output
        y_t = self.output_proj(h_t)
        
        return y_t, h_t, (alpha, beta)

class HyperController(nn.Module):
    """
    Scalar HyperController for Dual-Track ANA + HoloLink.
    Outputs:
    - alpha_A, beta_A (Track A - Reflex)
    - alpha_B, beta_B (Track B - Reasoning)
    - gamma_ret (HoloLink Retrieval Gate)
    """
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        # 5 scalars: 2 for Track A, 2 for Track B, 1 for Retrieval
        self.head = nn.Linear(hidden_dim, 5)
        
        # Init to small values to start with defaults
        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)

    def forward(self, x, force_prob=0.0):
        # x: [batch, d_model]
        features = self.net(x)
        out = self.head(features) # [batch, 5]
        
        # Split
        g_alpha_A = out[:, 0:1]
        g_beta_A  = out[:, 1:2]
        g_alpha_B = out[:, 2:3]
        g_beta_B  = out[:, 3:4]
        g_ret     = out[:, 4:5]
        
        # Curriculum Gating (Forcing)
        # If force_prob > 0, we overwrite g_ret with a high value for some samples
        if self.training and force_prob > 0.0:
            # Mask: 1 with prob `force_prob`
            mask = (torch.rand_like(g_ret) < force_prob).float()
            # Force g_ret to be large positive (sigmoid -> 1)
            # Mixed: mask * 5.0 + (1-mask) * g_ret
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret
        
        return g_alpha_A, g_beta_A, g_alpha_B, g_beta_B, g_ret

class HoloLink(nn.Module):
    """
    Associative Memory Module using Matrix Accumulation / Linear Attention.
    M_t = M_{t-1} + K(h_t) * V(h_t)^T
    Read = M_t * Q(x_t)
    """
    def __init__(self, d_model, state_dim, key_dim=64):
        super().__init__()
        self.key_dim = key_dim
        self.d_model = d_model
        
        # Q projection from input
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        
        # K projection from state (Learned)
        self.k_proj = nn.Linear(state_dim, key_dim, bias=False)
        # Unfrozen for Phase 3 Fix
        # nn.init.orthogonal_(self.k_proj.weight)
        
        # V projection from state (Learned)
        self.v_proj = nn.Linear(state_dim, d_model, bias=False) 
        
    def forward(self, x_t, h_t, M_prev):
        """
        x_t: [batch, d_model] (Query source)
        h_t: [batch, state_dim] (Key/Value source)
        M_prev: [batch, key_dim, d_model] (Accumulator)
        """
        batch_size = x_t.size(0)
        
        if M_prev is None:
            # v_proj out features
            d_val = self.v_proj.out_features
            M_prev = torch.zeros(batch_size, self.key_dim, d_val, device=x_t.device)
            
        # 1. Write: M_t = M_{t-1} + k_t * v_t^T
        k_t = self.k_proj(h_t) # [batch, key_dim]
        # Normalize Key (L2) - Crucial for stability
        k_t = torch.nn.functional.normalize(k_t, p=2, dim=-1)
        
        v_t = self.v_proj(h_t) # [batch, val_dim]
        
        # Outer product: [batch, key, 1] * [batch, 1, val] -> [batch, key, val]
        update = torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))
        
        # Decay? Standard Linear Attention often has decay.
        # For strict retrieval, maybe no decay (infinite memory).
        # Let's keep it simple: No decay.
        M_t = M_prev + update
        
        # 2. Read: r_t = M_t^T * q_t  (Wait, M maps K->V. Query is in K space. So M * q ?)
        # Dim check: M is [key, val]. q is [key]. Result [val].
        # M_t: [batch, key, val]
        # q_t: [batch, key]
        q_t = self.q_proj(x_t)
        # Normalize Query
        q_t = torch.nn.functional.normalize(q_t, p=2, dim=-1)
        
        # [batch, 1, key] * [batch, key, val] -> [batch, 1, val]
        retrieved = torch.bmm(q_t.unsqueeze(1), M_t).squeeze(1)
        
        return retrieved, M_t

class ANAModel(nn.Module):
    """
    Phase 2: Dual-Track ANA + HoloLink
    """
    def __init__(self, vocab_size, d_model, state_dim, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_dict = nn.ModuleDict({
                'controller': HyperController(d_model),
                'lru_A': LinearRecurrentUnit(d_model, state_dim), # Reflex
                'lru_B': LinearRecurrentUnit(d_model, state_dim), # Reasoning
                'holo': HoloLink(d_model, state_dim * 2) # Inputs concatenated state
            })
            
            # Initialize LRU biases
            # Track A (Reflex):
            with torch.no_grad():
                layer_dict['lru_A'].static_alpha_logit.fill_(-3.0) 
                layer_dict['lru_A'].static_beta_logit.fill_(2.0)   
                
            # Track B (Reasoning): Slow Decay
            with torch.no_grad():
                layer_dict['lru_B'].static_alpha_logit.fill_(3.0)  
                layer_dict['lru_B'].static_beta_logit.fill_(0.0)   
            
            self.layers.append(layer_dict)
        
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        
        # State init
        h_states_A = [None] * len(self.layers)
        h_states_B = [None] * len(self.layers)
        m_states = [None] * len(self.layers)
        
        final_layer_outputs = []
        info_log = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                ctl = layer['controller']
                lru_A = layer['lru_A']
                lru_B = layer['lru_B']
                holo = layer['holo']
                
                # 1. Controller
                # alpha, beta correspond to scalar modulation
                ga_A, gb_A, ga_B, gb_B, g_ret = ctl(xt, force_prob=force_prob)
                
                # Broadcast scalars to state_dim [batch, 1] -> [batch, state_dim]
                # Actually, LRU expects [batch, state_dim] or broadcastable?
                # LRU code: alpha = sigmoid(static + gate).
                # if gate is [batch, 1], it broadcasts to [batch, state_dim]. Perfectly fine.
                
                # 2. Update Tracks
                yt_A, ht_A, _ = lru_A(xt, h_states_A[i], dynamic_gates=(ga_A, gb_A))
                yt_B, ht_B, _ = lru_B(xt, h_states_B[i], dynamic_gates=(ga_B, gb_B))
                
                h_states_A[i] = ht_A
                h_states_B[i] = ht_B
                
                # 3. HoloLink
                # Concatenate states for key/value generation
                ht_combined = torch.cat([ht_A, ht_B], dim=-1)
                qt, mt_next = holo(xt, ht_combined, m_states[i])
                m_states[i] = mt_next
                
                # 4. Merge
                # y_t = y_A + y_B + (gate_ret * retrieved)
                # gate_ret is raw linear out, apply sigmoid
                
                ret_gate = torch.sigmoid(g_ret)
                
                layer_out = (yt_A + yt_B) / 2.0 + ret_gate * qt
                
                xt = xt + layer_out # Residual
                
                if return_info and i == 0 and t < 10:
                   info_log.append({
                       'ga_A': ga_A.mean().item(),
                       'ga_B': ga_B.mean().item(),
                       'ret_gate': ret_gate.mean().item()
                   })

            final_layer_outputs.append(xt)
            
        output_seq = torch.stack(final_layer_outputs, dim=1)
        
        # Add LayerNorm
        output_seq = self.norm(output_seq)
        
        logits = self.output_head(output_seq)
        
        return logits, info_log

class BaselineSSM(nn.Module):
    """
    Standard Static SSM (Mamba-like but purely recurrent for fair comparison)
    """
    def __init__(self, vocab_size, d_model, state_dim, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            LinearRecurrentUnit(d_model, state_dim)
             for _ in range(num_layers)
        ])
        
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        
        h_states = [None] * len(self.layers)
        final_layer_outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            for i, lru in enumerate(self.layers):
                yt, ht_next, _ = lru(xt, h_states[i], dynamic_gates=None)
                h_states[i] = ht_next
                xt = yt + xt # Residual
            final_layer_outputs.append(xt)
            
        output_seq = torch.stack(final_layer_outputs, dim=1)
        logits = self.output_head(output_seq)
        return logits, []
