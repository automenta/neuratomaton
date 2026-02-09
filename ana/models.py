
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ANAConfig

# JIT compiled scan function for efficiency
@torch.jit.script
def lru_scan(u, alpha, beta, h_init):
    # u: [batch, seq, dim]
    # alpha: [batch, seq, dim]
    # beta: [batch, seq, dim]
    # h_init: [batch, dim]

    h = h_init
    # Pre-allocate output
    # We cannot easily pre-allocate with jit script dynamic shapes in some versions,
    # but torch.zeros_like works.

    # We use a list to collect outputs because direct tensor assignment
    # in a loop can be problematic for autograd if not done carefully,
    # though with JIT it is usually fine.
    # List append is safer for JIT.
    h_out_list = []

    # Iterate over sequence length
    seq_len = u.size(1)
    for t in range(seq_len):
        h = alpha[:, t] * h + beta[:, t] * u[:, t]
        h_out_list.append(h)

    return torch.stack(h_out_list, dim=1)

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
        Step-wise forward
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

    def forward_sequence(self, x, dynamic_gates=None):
        # x: [batch, seq, d_model]
        batch_size, seq_len, _ = x.shape

        u = self.input_proj(x) # [batch, seq, state_dim]

        if dynamic_gates is not None:
            g_alpha, g_beta = dynamic_gates
            # g_alpha: [batch, seq, state_dim]
            alpha = torch.sigmoid(self.static_alpha_logit + g_alpha)
            beta = torch.sigmoid(self.static_beta_logit + g_beta)
        else:
            # Broadcast static
            alpha = torch.sigmoid(self.static_alpha_logit).view(1, 1, -1).expand(batch_size, seq_len, -1)
            beta = torch.sigmoid(self.static_beta_logit).view(1, 1, -1).expand(batch_size, seq_len, -1)

        h_init = torch.zeros(batch_size, self.state_dim, device=x.device)

        # Run scan
        h_seq = lru_scan(u, alpha, beta, h_init)

        # Output proj
        y_seq = self.output_proj(h_seq)

        return y_seq, h_seq

class HyperController(nn.Module):
    """
    Scalar HyperController for Dual-Track ANA + HoloLink.
    Outputs:
    - alpha_A, beta_A (Track A - Reflex)
    - alpha_B, beta_B (Track B - Reasoning)
    - gamma_ret (HoloLink Retrieval Gate)
    """
    def __init__(self, config: ANAConfig, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        # 5 scalars: 2 for Track A, 2 for Track B, 1 for Retrieval
        self.head = nn.Linear(hidden_dim, 5)
        
        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)

    def forward(self, x, force_prob=0.0):
        # x: [batch, d_model]
        features = self.net(x)
        out = self.head(features) # [batch, 5]
        
        g_alpha_A = out[:, 0:1]
        g_beta_A  = out[:, 1:2]
        g_alpha_B = out[:, 2:3]
        g_beta_B  = out[:, 3:4]
        g_ret     = out[:, 4:5]
        
        if self.training and force_prob > 0.0:
            mask = (torch.rand_like(g_ret) < force_prob).float()
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret
        
        return g_alpha_A, g_beta_A, g_alpha_B, g_beta_B, g_ret

    def forward_sequence(self, x, force_prob=0.0):
        # x: [batch, seq, d_model]
        features = self.net(x)
        out = self.head(features) # [batch, seq, 5]

        g_alpha_A = out[..., 0:1]
        g_beta_A  = out[..., 1:2]
        g_alpha_B = out[..., 2:3]
        g_beta_B  = out[..., 3:4]
        g_ret     = out[..., 4:5]

        if self.training and force_prob > 0.0:
            mask = (torch.rand_like(g_ret) < force_prob).float()
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret

        return g_alpha_A, g_beta_A, g_alpha_B, g_beta_B, g_ret

class HoloLink(nn.Module):
    """
    Associative Memory Module using Matrix Accumulation / Linear Attention.
    """
    def __init__(self, config: ANAConfig, input_dim: int):
        super().__init__()
        self.key_dim = config.key_dim
        self.d_model = config.d_model
        
        # Q projection from input
        self.q_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        
        # K projection from state (Learned)
        self.k_proj = nn.Linear(input_dim, self.key_dim, bias=False)
        
        # V projection from state (Learned)
        self.v_proj = nn.Linear(input_dim, self.d_model, bias=False)
        
    def forward(self, x_t, h_t, M_prev):
        """
        Step-wise forward
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
        
        M_t = M_prev + update
        
        # 2. Read: r_t = M_t^T * q_t
        q_t = self.q_proj(x_t)
        q_t = torch.nn.functional.normalize(q_t, p=2, dim=-1)
        
        # [batch, 1, key] * [batch, key, val] -> [batch, 1, val]
        retrieved = torch.bmm(q_t.unsqueeze(1), M_t).squeeze(1)
        
        return retrieved, M_t

    def forward_sequence(self, x, h):
        # x: [batch, seq, d_model]
        # h: [batch, seq, input_dim]

        k = self.k_proj(h) # [batch, seq, key_dim]
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        v = self.v_proj(h) # [batch, seq, val_dim]

        # Update: k * v^T
        # [batch, seq, key, 1] * [batch, seq, 1, val] -> [batch, seq, key, val]
        update = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))

        # Cumulative Sum (Parallel Scan for additive update)
        # M_t = sum(updates[:t+1])
        M_seq = torch.cumsum(update, dim=1)

        # Read: M_t * q_t
        q = self.q_proj(x) # [batch, seq, key_dim]
        q = torch.nn.functional.normalize(q, p=2, dim=-1)

        # [batch, seq, 1, key] @ [batch, seq, key, val] -> [batch, seq, 1, val]
        retrieved = torch.matmul(q.unsqueeze(-2), M_seq).squeeze(-2)

        return retrieved, M_seq

class ANAModel(nn.Module):
    """
    Phase 2: Dual-Track ANA + HoloLink
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer_dict = nn.ModuleDict()

            if config.use_controller:
                layer_dict['controller'] = HyperController(config)

            # TODO: Support arbitrary number of tracks
            layer_dict['lru_A'] = LinearRecurrentUnit(config) # Reflex
            layer_dict['lru_B'] = LinearRecurrentUnit(config) # Reasoning

            if config.use_hololink:
                # Inputs concatenated state of track A and B
                layer_dict['holo'] = HoloLink(config, input_dim=config.state_dim * 2)
            
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
        
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward_parallel(self, input_ids, return_info=False, force_prob=0.0):
        x = self.embedding(input_ids)
        # x: [batch, seq, d_model]

        info_log = []

        for i, layer in enumerate(self.layers):
            lru_A = layer['lru_A']
            lru_B = layer['lru_B']

            # 1. Controller (Parallel over time)
            ga_A, gb_A, ga_B, gb_B, g_ret = None, None, None, None, None

            if self.config.use_controller:
                ctl = layer['controller']
                ga_A, gb_A, ga_B, gb_B, g_ret = ctl.forward_sequence(x, force_prob=force_prob)

            # 2. Update Tracks (Parallel over time using scan)
            yt_A, ht_A = lru_A.forward_sequence(x, dynamic_gates=(ga_A, gb_A) if ga_A is not None else None)
            yt_B, ht_B = lru_B.forward_sequence(x, dynamic_gates=(ga_B, gb_B) if ga_B is not None else None)

            # 3. HoloLink (Parallel over time using cumsum)
            qt = 0
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat([ht_A, ht_B], dim=-1)
                qt, _ = holo.forward_sequence(x, ht_combined)

            # 4. Merge
            if self.config.use_controller and self.config.use_hololink:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = (yt_A + yt_B) / 2.0 + ret_gate * qt
            else:
                layer_out = (yt_A + yt_B) / 2.0
                if self.config.use_hololink:
                    layer_out = layer_out + qt

            x = x + layer_out # Residual

            if return_info and i == 0:
               stats = {}
               if ga_A is not None:
                   stats['ga_A'] = ga_A.mean().item()
                   stats['ga_B'] = ga_B.mean().item()
               if g_ret is not None:
                   stats['ret_gate'] = torch.sigmoid(g_ret).mean().item()
               info_log.append(stats)

        # Norm and Head
        x = self.norm(x)
        logits = self.output_head(x)

        return logits, info_log

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        if self.config.use_parallel_scan:
             return self.forward_parallel(input_ids, return_info, force_prob)

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
                lru_A = layer['lru_A']
                lru_B = layer['lru_B']
                
                # 1. Controller
                ga_A, gb_A, ga_B, gb_B, g_ret = None, None, None, None, None
                
                if self.config.use_controller:
                    ctl = layer['controller']
                    # alpha, beta correspond to scalar modulation
                    ga_A, gb_A, ga_B, gb_B, g_ret = ctl(xt, force_prob=force_prob)
                
                # 2. Update Tracks
                yt_A, ht_A, _ = lru_A(xt, h_states_A[i], dynamic_gates=(ga_A, gb_A) if ga_A is not None else None)
                yt_B, ht_B, _ = lru_B(xt, h_states_B[i], dynamic_gates=(ga_B, gb_B) if ga_B is not None else None)
                
                h_states_A[i] = ht_A
                h_states_B[i] = ht_B
                
                # 3. HoloLink
                qt = 0
                if self.config.use_hololink:
                    holo = layer['holo']
                    # Concatenate states for key/value generation
                    ht_combined = torch.cat([ht_A, ht_B], dim=-1)
                    qt, mt_next = holo(xt, ht_combined, m_states[i])
                    m_states[i] = mt_next
                
                # 4. Merge
                # y_t = y_A + y_B + (gate_ret * retrieved)
                
                if self.config.use_controller and self.config.use_hololink:
                    ret_gate = torch.sigmoid(g_ret)
                    layer_out = (yt_A + yt_B) / 2.0 + ret_gate * qt
                else:
                    layer_out = (yt_A + yt_B) / 2.0
                    if self.config.use_hololink:
                        # Simple addition if no controller
                        layer_out = layer_out + qt
                
                xt = xt + layer_out # Residual
                
                if return_info and i == 0 and t < 10:
                   stats = {}
                   if ga_A is not None:
                       stats['ga_A'] = ga_A.mean().item()
                       stats['ga_B'] = ga_B.mean().item()
                   if g_ret is not None:
                       stats['ret_gate'] = torch.sigmoid(g_ret).mean().item()
                   info_log.append(stats)

            final_layer_outputs.append(xt)
            
        output_seq = torch.stack(final_layer_outputs, dim=1)
        
        # Add LayerNorm
        output_seq = self.norm(output_seq)
        
        logits = self.output_head(output_seq)
        
        return logits, info_log
