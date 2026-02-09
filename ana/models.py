
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import ANAConfig

def parallel_scan_lru(u, alpha, beta):
    """
    Parallel implementation of Linear Recurrence:
    h_t = alpha_t * h_{t-1} + beta_t * u_t

    Using log-space cumulative sum trick:
    h_t = exp(cumsum(log(alpha))) * cumsum(beta * u * exp(-cumsum(log(alpha))))

    Args:
        u: [batch, seq, dim]
        alpha: [batch, seq, dim]
        beta: [batch, seq, dim]
    Returns:
        h: [batch, seq, dim]
    """
    # 1. Compute log alpha (Use float64 for stability)
    log_alpha = torch.log(alpha.to(torch.float64) + 1e-8) # Avoid log(0)

    # 2. Cumulative sum of log alpha
    # C_t = sum_{i=0}^t log(alpha_i)
    # Note: Sequence dimension is 1
    C = torch.cumsum(log_alpha, dim=1)

    # 3. Compute the term to be accumulated
    # term = beta * u * exp(-C)

    term = beta.to(torch.float64) * u.to(torch.float64) * torch.exp(-C)

    # 4. Cumulative sum of term
    S = torch.cumsum(term, dim=1)

    # 5. Final result
    h = torch.exp(C) * S

    return h.to(alpha.dtype)

class LinearRecurrentUnit(nn.Module):
    """
    A simplified Linear Recurrent Unit (LRU) / SSM layer.
    h_t = alpha * h_{t-1} + beta * x_t
    y_t = h_t
    
    Alpha and Beta can be static (learned parameters) or dynamic (provided per step).
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        self.use_parallel_scan = config.use_parallel_scan
        
        # Projections for input x_t -> state_dim
        self.input_proj = nn.Linear(self.d_model, self.state_dim)
        
        # Output projection state_dim -> d_model
        self.output_proj = nn.Linear(self.state_dim, self.d_model)
        
        # Static parameters (used as baseline or base for dynamic)
        # alpha should be in [0, 1]. initialized to preserve history (high alpha)
        # beta should be in [0, 1]. initialized to accept some input
        self.static_alpha_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(2, 4)) # sigmoid(2) ~= 0.88, sigmoid(4) ~= 0.98
        self.static_beta_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(-2, 0)) # sigmoid(-2) ~= 0.12, sigmoid(0) = 0.5

    def forward(self, x, h_prev=None, dynamic_gates=None):
        """
        x: [batch, d_model] (or [batch, seq, d_model] if parallel)
        h_prev: [batch, state_dim]
        dynamic_gates: tuple (gate_alpha, gate_beta)
        """
        if x.dim() == 2: # [batch, d_model]
             return self.forward_step(x, h_prev, dynamic_gates)
        elif x.dim() == 3: # [batch, seq, d_model]
             return self.forward_parallel(x, dynamic_gates)
        else:
             raise ValueError(f"Invalid input shape: {x.shape}")

    def forward_step(self, x, h_prev=None, dynamic_gates=None):
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

    def forward_parallel(self, x, dynamic_gates=None):
        """
        x: [batch, seq, d_model]
        dynamic_gates: tuple(gate_alpha, gate_beta), each [batch, seq, state_dim]
        """
        batch, seq, _ = x.shape
        u = self.input_proj(x) # [batch, seq, state_dim]

        if dynamic_gates is not None:
            gate_alpha, gate_beta = dynamic_gates
            # gate_alpha: [batch, seq, state_dim]

            # Broadcast static logits: [state_dim] -> [1, 1, state_dim]
            static_alpha = self.static_alpha_logit.view(1, 1, -1)
            static_beta = self.static_beta_logit.view(1, 1, -1)

            alpha = torch.sigmoid(static_alpha + gate_alpha)
            beta = torch.sigmoid(static_beta + gate_beta)
        else:
            static_alpha = self.static_alpha_logit.view(1, 1, -1)
            static_beta = self.static_beta_logit.view(1, 1, -1)
            alpha = torch.sigmoid(static_alpha).expand(batch, seq, self.state_dim)
            beta = torch.sigmoid(static_beta).expand(batch, seq, self.state_dim)

        # Parallel Scan
        h = parallel_scan_lru(u, alpha, beta)

        # Output proj
        y = self.output_proj(h)

        return y, h, (alpha, beta) # Return FULL state and gates

class HyperController(nn.Module):
    """
    Scalar HyperController for Dual-Track ANA + HoloLink.
    Outputs:
    - alpha_k, beta_k for each k in num_tracks
    - gamma_ret (HoloLink Retrieval Gate) if use_hololink
    """
    def __init__(self, config: ANAConfig, hidden_dim=64):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Outputs: 2 per track + 1 for HoloLink (if enabled)
        self.output_dim = config.num_tracks * 2
        if config.use_hololink:
            self.output_dim += 1

        self.head = nn.Linear(hidden_dim, self.output_dim)
        
        # Init to small values to start with defaults
        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)

    def forward(self, x, force_prob=0.0):
        # x: [batch, d_model] OR [batch, seq, d_model]
        features = self.net(x)
        out = self.head(features) # [batch, ..., output_dim]
        
        gates = {}
        idx = 0
        for k in range(self.config.num_tracks):
            gates[f'alpha_{k}'] = out[..., idx:idx+1]
            idx += 1
            gates[f'beta_{k}'] = out[..., idx:idx+1]
            idx += 1

        if self.config.use_hololink:
            g_ret = out[..., idx:idx+1]

            # Curriculum Gating (Forcing)
            if self.training and force_prob > 0.0:
                mask = (torch.rand_like(g_ret) < force_prob).float()
                g_ret = mask * 5.0 + (1.0 - mask) * g_ret

            gates['ret'] = g_ret
        
        return gates

class HoloLink(nn.Module):
    """
    Associative Memory Module using Matrix Accumulation / Linear Attention.
    M_t = M_{t-1} + K(h_t) * V(h_t)^T
    Read = M_t * Q(x_t)
    """
    def __init__(self, config: ANAConfig, input_state_dim, key_dim=64):
        super().__init__()
        self.key_dim = key_dim
        self.d_model = config.d_model
        
        # Q projection from input
        self.q_proj = nn.Linear(config.d_model, key_dim, bias=False)
        
        # K projection from state (Learned)
        self.k_proj = nn.Linear(input_state_dim, key_dim, bias=False)
        
        # V projection from state (Learned)
        self.v_proj = nn.Linear(input_state_dim, config.d_model, bias=False)
        
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
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            modules = {}
            if config.use_controller:
                modules['controller'] = HyperController(config)
            
            for k in range(config.num_tracks):
                modules[f'lru_{k}'] = LinearRecurrentUnit(config)
                # Initialize specific tracks if needed (e.g. 0=Reflex, 1=Reasoning)
                with torch.no_grad():
                    if k == 0:
                        modules[f'lru_{k}'].static_alpha_logit.fill_(-3.0)
                        modules[f'lru_{k}'].static_beta_logit.fill_(2.0)
                    elif k == 1:
                        modules[f'lru_{k}'].static_alpha_logit.fill_(3.0)
                        modules[f'lru_{k}'].static_beta_logit.fill_(0.0)

            if config.use_hololink:
                modules['holo'] = HoloLink(config, config.state_dim * config.num_tracks)
            
            layer_dict = nn.ModuleDict(modules)
            self.layers.append(layer_dict)
        
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        
        if self.config.use_parallel_scan:
             return self.forward_parallel(x, return_info, force_prob)

        # State init
        h_states = [[None] * len(self.layers) for _ in range(self.config.num_tracks)]
        m_states = [None] * len(self.layers)
        
        final_layer_outputs = []
        info_log = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                # 1. Controller
                gates = None
                if self.config.use_controller:
                    gates = layer['controller'](xt, force_prob=force_prob)
                
                # 2. Update Tracks
                track_outputs = []
                current_h_states = []
                
                for k in range(self.config.num_tracks):
                    lru = layer[f'lru_{k}']

                    dynamic_gates = None
                    if gates:
                        dynamic_gates = (gates[f'alpha_{k}'], gates[f'beta_{k}'])

                    yt, ht, _ = lru(xt, h_states[k][i], dynamic_gates=dynamic_gates)
                    h_states[k][i] = ht

                    track_outputs.append(yt)
                    current_h_states.append(ht)
                
                # 3. HoloLink
                holo_out = 0.0
                if self.config.use_hololink:
                    holo = layer['holo']
                    ht_combined = torch.cat(current_h_states, dim=-1)
                    qt, mt_next = holo(xt, ht_combined, m_states[i])
                    m_states[i] = mt_next

                    if gates:
                        ret_gate = torch.sigmoid(gates['ret'])
                        holo_out = ret_gate * qt
                    else:
                         # Default if no controller? maybe 0.5?
                         # Or simpler: just qt.
                         holo_out = qt

                # 4. Merge
                # Average track outputs
                avg_track = sum(track_outputs) / self.config.num_tracks
                
                layer_out = avg_track + holo_out
                
                xt = xt + layer_out # Residual
                
                if return_info and i == 0 and t < 10 and gates:
                   info_log.append({
                       'ga_A': gates.get('alpha_0', torch.tensor(0.0)).mean().item(),
                       'ret_gate': gates.get('ret', torch.tensor(0.0)).mean().item() if 'ret' in gates else 0.0
                   })

            final_layer_outputs.append(xt)
            
        output_seq = torch.stack(final_layer_outputs, dim=1)
        output_seq = self.norm(output_seq)
        logits = self.output_head(output_seq)

        return logits, info_log

    def forward_parallel(self, x, return_info=False, force_prob=0.0):
        # x is already [batch, seq, dim] embeddings
        batch, seq_len, _ = x.shape
        info_log = []
        
        for i, layer in enumerate(self.layers):
            # 1. Controller (Parallel)
            gates = None
            if self.config.use_controller:
                gates = layer['controller'](x, force_prob=force_prob)

            # 2. Update Tracks (Parallel)
            track_outputs = []
            current_h_states = []

            for k in range(self.config.num_tracks):
                lru = layer[f'lru_{k}']

                dynamic_gates = None
                if gates:
                    dynamic_gates = (gates[f'alpha_{k}'], gates[f'beta_{k}'])

                yt, ht, _ = lru(x, dynamic_gates=dynamic_gates)
                track_outputs.append(yt)
                current_h_states.append(ht)

            # 3. HoloLink
            holo_out = 0.0
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(current_h_states, dim=-1) # [batch, seq, state_dim*num_tracks]

                # Run HoloLink loop
                holo_outs = []
                m_state = None
                for t in range(seq_len):
                    xt_curr = x[:, t, :]
                    ht_curr = ht_combined[:, t, :]
                    qt, m_next = holo(xt_curr, ht_curr, m_state)
                    m_state = m_next
                    holo_outs.append(qt)

                holo_seq = torch.stack(holo_outs, dim=1) # [batch, seq, dim]

                if gates:
                    ret_gate = torch.sigmoid(gates['ret']) # [batch, seq, 1]
                    holo_out = ret_gate * holo_seq
                else:
                    holo_out = holo_seq

            # 4. Merge
            avg_track = sum(track_outputs) / self.config.num_tracks

            layer_out = avg_track + holo_out

            x = x + layer_out # Residual

            if return_info and i == 0 and gates:
                 info_log.append({
                       'ga_A': gates.get('alpha_0', torch.zeros_like(x))[:, 0, :].mean().item(),
                       'ret_gate': gates.get('ret', torch.zeros_like(x))[:, 0, :].mean().item() if 'ret' in gates else 0.0
                 })

        output_seq = self.norm(x)
        logits = self.output_head(output_seq)
        
        return logits, info_log

class BaselineSSM(nn.Module):
    """
    Standard Static SSM (Mamba-like but purely recurrent for fair comparison)
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            LinearRecurrentUnit(config)
             for _ in range(config.num_layers)
        ])
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        
        # Parallel scan if enabled
        if self.config.use_parallel_scan:
            for i, lru in enumerate(self.layers):
                 yt, _, _ = lru(x, dynamic_gates=None)
                 x = yt + x
            output_seq = x
        else:
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
