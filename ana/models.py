
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import ANAConfig

# JIT compiled scan function for efficiency (Sequential fallback)
@torch.jit.script
def lru_scan(u, alpha, beta, h_init):
    h = h_init
    h_out_list = []

    seq_len = u.size(1)
    for t in range(seq_len):
        h = alpha[:, t] * h + beta[:, t] * u[:, t]
        h_out_list.append(h)

    return torch.stack(h_out_list, dim=1)

def parallel_scan_log(u, alpha, beta, h_init):
    # Hillis-Steele Parallel Scan
    # (a_r, b_r) o (a_l, b_l) = (a_r * a_l, a_r * b_l + b_r)

    b = beta * u
    a = alpha

    batch, seq, dim = a.shape

    # Pad to power of 2
    n = 1
    while n < seq:
        n *= 2

    # Pad
    if n > seq:
        pad = n - seq
        a_pad = torch.ones(batch, pad, dim, device=a.device)
        b_pad = torch.zeros(batch, pad, dim, device=b.device)
        a = torch.cat([a, a_pad], dim=1)
        b = torch.cat([b, b_pad], dim=1)
    else:
        # Clone to avoid in-place modification issues if any (though we assign to new vars)
        a = a.clone()
        b = b.clone()

    # Log iterations
    log_n = int(math.log2(n))

    curr_a = a
    curr_b = b

    for i in range(log_n):
        d = 2**i

        # Shift
        # Pad with identity (1, 0)
        a_shifted = torch.cat([torch.ones(batch, d, dim, device=a.device), curr_a[:, :-d]], dim=1)
        b_shifted = torch.cat([torch.zeros(batch, d, dim, device=b.device), curr_b[:, :-d]], dim=1)

        # Update
        # new_a = curr_a * a_shifted
        # new_b = curr_a * b_shifted + curr_b
        curr_a = curr_a * a_shifted
        curr_b = curr_a * b_shifted + curr_b # Error: curr_a is already updated? No, new assignment.
        # Wait, Python evaluates RHS first.
        # But `curr_a` in the second line uses the OLD `curr_a`?
        # No, if I do `curr_a = ...`, then `curr_b = ...` uses the NEW `curr_a`.
        # I need temporaries.

    # Correct Loop:
    curr_a = a
    curr_b = b

    for i in range(log_n):
        d = 2**i

        a_shifted = torch.cat([torch.ones(batch, d, dim, device=a.device), curr_a[:, :-d]], dim=1)
        b_shifted = torch.cat([torch.zeros(batch, d, dim, device=b.device), curr_b[:, :-d]], dim=1)

        next_a = curr_a * a_shifted
        next_b = curr_a * b_shifted + curr_b

        curr_a = next_a
        curr_b = next_b

    # Slice back
    final_a = curr_a[:, :seq]
    final_b = curr_b[:, :seq]

    # Apply to h_init
    h = final_a * h_init.unsqueeze(1) + final_b

    return h

class LinearRecurrentUnit(nn.Module):
    """
    A simplified Linear Recurrent Unit (LRU) / SSM layer.
    h_t = alpha * h_{t-1} + beta * x_t
    y_t = h_t
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        
        self.input_proj = nn.Linear(self.d_model, self.state_dim)
        self.output_proj = nn.Linear(self.state_dim, self.d_model)
        
        # Static parameters
        self.static_alpha_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(2, 4))
        self.static_beta_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(-2, 0))

    def forward(self, x, h_prev=None, dynamic_gates=None):
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.state_dim, device=x.device)
            
        u_t = self.input_proj(x)
        
        if dynamic_gates is not None:
            gate_alpha, gate_beta = dynamic_gates
            alpha = torch.sigmoid(self.static_alpha_logit + gate_alpha)
            beta = torch.sigmoid(self.static_beta_logit + gate_beta)
        else:
            alpha = torch.sigmoid(self.static_alpha_logit)
            beta = torch.sigmoid(self.static_beta_logit)
            
        h_t = alpha * h_prev + beta * u_t
        y_t = self.output_proj(h_t)
        
        return y_t, h_t

    def forward_sequence(self, x, dynamic_gates=None):
        batch_size, seq_len, _ = x.shape
        u = self.input_proj(x)

        if dynamic_gates is not None:
            g_alpha, g_beta = dynamic_gates
            alpha = torch.sigmoid(self.static_alpha_logit + g_alpha)
            beta = torch.sigmoid(self.static_beta_logit + g_beta)
        else:
            alpha = torch.sigmoid(self.static_alpha_logit).view(1, 1, -1).expand(batch_size, seq_len, -1)
            beta = torch.sigmoid(self.static_beta_logit).view(1, 1, -1).expand(batch_size, seq_len, -1)

        h_init = torch.zeros(batch_size, self.state_dim, device=x.device)

        if self.config.use_parallel_scan:
            h_seq = parallel_scan_log(u, alpha, beta, h_init)
        else:
            h_seq = lru_scan(u, alpha, beta, h_init)

        y_seq = self.output_proj(h_seq)
        return y_seq, h_seq

class HyperController(nn.Module):
    """
    HyperController for Multi-Track ANA + HoloLink.
    Outputs per track: alpha_gate, beta_gate, mix_logit
    Plus: retrieval_gate
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
        
        # 2 scalars (alpha, beta) + 1 scalar (mix) per track
        # + 1 scalar (retrieval)
        self.output_dim = config.track_count * 3 + 1
        self.head = nn.Linear(hidden_dim, self.output_dim)

        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)

    def split_outputs(self, out):
        # out: [..., output_dim]
        # Structure: [Track1_Alpha, Track1_Beta, Track1_Mix, Track2..., Ret]

        track_outputs = []
        idx = 0
        for _ in range(self.config.track_count):
            alpha = out[..., idx:idx+1]
            beta = out[..., idx+1:idx+2]
            mix = out[..., idx+2:idx+3]
            track_outputs.append((alpha, beta, mix))
            idx += 3

        ret_gate = out[..., idx:idx+1]
        return track_outputs, ret_gate

    def forward(self, x, force_prob=0.0):
        features = self.net(x)
        out = self.head(features)
        
        track_outputs, g_ret = self.split_outputs(out)
        
        if self.training and force_prob > 0.0:
            mask = (torch.rand_like(g_ret) < force_prob).float()
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret
        
        return track_outputs, g_ret

    def forward_sequence(self, x, force_prob=0.0):
        features = self.net(x)
        out = self.head(features)

        track_outputs, g_ret = self.split_outputs(out)

        if self.training and force_prob > 0.0:
            mask = (torch.rand_like(g_ret) < force_prob).float()
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret

        return track_outputs, g_ret

class HoloLink(nn.Module):
    """
    Associative Memory Module using Matrix Accumulation / Linear Attention.
    """
    def __init__(self, config: ANAConfig, input_dim: int):
        super().__init__()
        self.key_dim = config.key_dim
        self.d_model = config.d_model
        
        self.q_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.d_model, bias=False)
        
    def forward(self, x_t, h_t, M_prev):
        batch_size = x_t.size(0)
        
        if M_prev is None:
            d_val = self.v_proj.out_features
            M_prev = torch.zeros(batch_size, self.key_dim, d_val, device=x_t.device)
            
        k_t = self.k_proj(h_t)
        k_t = torch.nn.functional.normalize(k_t, p=2, dim=-1)
        v_t = self.v_proj(h_t)
        
        update = torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))
        M_t = M_prev + update
        
        q_t = self.q_proj(x_t)
        q_t = torch.nn.functional.normalize(q_t, p=2, dim=-1)
        
        retrieved = torch.bmm(q_t.unsqueeze(1), M_t).squeeze(1)
        return retrieved, M_t

    def forward_sequence(self, x, h):
        k = self.k_proj(h)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
        v = self.v_proj(h)

        update = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        M_seq = torch.cumsum(update, dim=1)

        q = self.q_proj(x)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)

        retrieved = torch.matmul(q.unsqueeze(-2), M_seq).squeeze(-2)
        return retrieved, M_seq

class ANAModel(nn.Module):
    """
    Phase 2: Multi-Track ANA + HoloLink
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

            layer_dict['tracks'] = nn.ModuleList([
                LinearRecurrentUnit(config) for _ in range(config.track_count)
            ])

            if config.use_hololink:
                # Inputs concatenated state of all tracks
                layer_dict['holo'] = HoloLink(config, input_dim=config.state_dim * config.track_count)

            # Initialize LRU biases to diversify tracks
            if config.track_count == 2:
                with torch.no_grad():
                    # Track A (Reflex)
                    layer_dict['tracks'][0].static_alpha_logit.fill_(-3.0)
                    layer_dict['tracks'][0].static_beta_logit.fill_(2.0)
                    # Track B (Reasoning)
                    layer_dict['tracks'][1].static_alpha_logit.fill_(3.0)
                    layer_dict['tracks'][1].static_beta_logit.fill_(0.0)
            
            self.layers.append(layer_dict)
        
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward_parallel(self, input_ids, return_info=False, force_prob=0.0):
        x = self.embedding(input_ids)
        info_log = []

        for i, layer in enumerate(self.layers):
            # 1. Controller
            track_outputs = None
            g_ret = None

            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret = ctl.forward_sequence(x, force_prob=force_prob)

            # 2. Update Tracks
            track_states = []
            track_results = []
            track_mix_logits = []

            tracks = layer['tracks']
            for t_idx, track in enumerate(tracks):
                gates = None
                mix = None
                if track_outputs is not None:
                    g_alpha, g_beta, g_mix = track_outputs[t_idx]
                    gates = (g_alpha, g_beta)
                    mix = g_mix

                yt, ht = track.forward_sequence(x, dynamic_gates=gates)
                track_states.append(ht)
                track_results.append(yt)
                if mix is not None:
                    track_mix_logits.append(mix)
                else:
                    track_mix_logits.append(torch.zeros_like(yt[..., :1]))

            # Mixing
            stacked_results = torch.stack(track_results, dim=2)
            stacked_mix = torch.stack(track_mix_logits, dim=2)
            mix_weights = torch.softmax(stacked_mix, dim=2)

            layer_out = (stacked_results * mix_weights).sum(dim=2)

            # 3. HoloLink
            qt = 0
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(track_states, dim=-1)
                qt, _ = holo.forward_sequence(x, ht_combined)

            # 4. Merge
            if self.config.use_controller and self.config.use_hololink:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = layer_out + ret_gate * qt
            elif self.config.use_hololink:
                layer_out = layer_out + qt

            x = x + layer_out # Residual

            if return_info and i == 0:
               stats = {}
               if track_outputs is not None:
                   stats['ga_0'] = track_outputs[0][0].mean().item()
               if g_ret is not None:
                   stats['ret_gate'] = torch.sigmoid(g_ret).mean().item()
               info_log.append(stats)

        x = self.norm(x)
        logits = self.output_head(x)
        return logits, info_log

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        if self.config.use_parallel_scan:
             return self.forward_parallel(input_ids, return_info, force_prob)

        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        
        h_states = [[None] * self.config.track_count for _ in range(self.config.num_layers)]
        m_states = [None] * len(self.layers)
        
        final_layer_outputs = []
        info_log = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                tracks = layer['tracks']
                
                # 1. Controller
                track_outputs = None
                g_ret = None
                
                if self.config.use_controller:
                    ctl = layer['controller']
                    track_outputs, g_ret = ctl(xt, force_prob=force_prob)
                
                # 2. Update Tracks
                track_results = []
                track_mix_logits = []

                for t_idx, track in enumerate(tracks):
                    gates = None
                    mix = None
                    if track_outputs is not None:
                        g_alpha, g_beta, g_mix = track_outputs[t_idx]
                        gates = (g_alpha, g_beta)
                        mix = g_mix

                    yt, ht = track(xt, h_states[i][t_idx], dynamic_gates=gates)
                    h_states[i][t_idx] = ht
                    track_results.append(yt)

                    if mix is not None:
                        track_mix_logits.append(mix)
                    else:
                        track_mix_logits.append(torch.zeros(batch, 1, device=x.device))
                
                # Mixing
                stacked_results = torch.stack(track_results, dim=1)
                stacked_mix = torch.stack(track_mix_logits, dim=1)
                mix_weights = torch.softmax(stacked_mix, dim=1)

                layer_out = (stacked_results * mix_weights).sum(dim=1)
                
                # 3. HoloLink
                qt = 0
                if self.config.use_hololink:
                    holo = layer['holo']
                    ht_combined = torch.cat(h_states[i], dim=-1)
                    qt, mt_next = holo(xt, ht_combined, m_states[i])
                    m_states[i] = mt_next
                
                # 4. Merge
                if self.config.use_controller and self.config.use_hololink:
                    ret_gate = torch.sigmoid(g_ret)
                    layer_out = layer_out + ret_gate * qt
                elif self.config.use_hololink:
                    layer_out = layer_out + qt
                
                xt = xt + layer_out
                
                if return_info and i == 0 and t < 10:
                   stats = {}
                   if track_outputs is not None:
                       stats['ga_0'] = track_outputs[0][0].mean().item()
                   if g_ret is not None:
                       stats['ret_gate'] = torch.sigmoid(g_ret).mean().item()
                   info_log.append(stats)

            final_layer_outputs.append(xt)
            
        output_seq = torch.stack(final_layer_outputs, dim=1)
        output_seq = self.norm(output_seq)
        logits = self.output_head(output_seq)
        return logits, info_log
