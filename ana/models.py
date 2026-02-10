import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import ANAConfig

@torch.jit.script
def lru_scan_jit(u: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, h_init: torch.Tensor) -> torch.Tensor:
    h = h_init
    h_out_list = []
    seq_len = u.size(1)
    for t in range(seq_len):
        h = alpha[:, t] * h + beta[:, t] * u[:, t]
        h_out_list.append(h)
    return torch.stack(h_out_list, dim=1)

def parallel_scan_cumsum(u, log_alpha, beta):
    C = torch.cumsum(log_alpha.to(torch.float64), dim=1)
    term = beta.to(torch.float64) * u.to(torch.float64) * torch.exp(-C)
    S = torch.cumsum(term, dim=1)
    h = torch.exp(C) * S
    return h.to(u.dtype)

def parallel_scan_hillis_steele(u, alpha, beta, h_init):
    b = beta * u
    a = alpha
    batch, seq, dim = a.shape
    
    n = 1
    while n < seq:
        n *= 2
    
    if n > seq:
        pad = n - seq
        a_pad = torch.ones(batch, pad, dim, device=a.device)
        b_pad = torch.zeros(batch, pad, dim, device=b.device)
        a = torch.cat([a, a_pad], dim=1)
        b = torch.cat([b, b_pad], dim=1)
    else:
        a = a.clone()
        b = b.clone()
    
    log_n = int(math.log2(n))
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
    
    final_a = curr_a[:, :seq]
    final_b = curr_b[:, :seq]
    h = final_a * h_init.unsqueeze(1) + final_b
    return h

class LinearRecurrentUnit(nn.Module):
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        
        self.input_proj = nn.Linear(self.d_model, self.state_dim)
        self.output_proj = nn.Linear(self.state_dim, self.d_model)
        
        self.static_alpha_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(2, 4))
        self.static_beta_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(-2, 0))

    def forward(self, x, h_prev=None, dynamic_gates=None):
        if x.dim() == 2:
            return self._forward_step(x, h_prev, dynamic_gates)
        elif x.dim() == 3:
            return self._forward_sequence(x, dynamic_gates)
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

    def _forward_step(self, x, h_prev, dynamic_gates):
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

    def _forward_sequence(self, x, dynamic_gates=None):
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
            log_alpha = F.logsigmoid(torch.log(alpha / (1 - alpha + 1e-8) + 1e-8))
            h_seq = parallel_scan_cumsum(u, torch.log(alpha + 1e-8), beta)
        else:
            h_seq = lru_scan_jit(u, alpha, beta, h_init)
        
        y_seq = self.output_proj(h_seq)
        return y_seq, h_seq

class HyperController(nn.Module):
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.controller_hidden_dim
        
        layers = []
        layers.append(nn.Linear(config.d_model, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(config.controller_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        self.net = nn.Sequential(*layers)
        
        self.output_dim = config.track_count * 3 + 2
        self.head = nn.Linear(hidden_dim, self.output_dim)
        
        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)

    def _split_outputs(self, out):
        track_outputs = []
        idx = 0
        for _ in range(self.config.track_count):
            alpha = out[..., idx:idx+1]
            beta = out[..., idx+1:idx+2]
            mix = out[..., idx+2:idx+3]
            track_outputs.append((alpha, beta, mix))
            idx += 3
        
        ret_gate = out[..., idx:idx+1]
        halt_logit = out[..., idx+1:idx+2]
        return track_outputs, ret_gate, halt_logit

    def forward(self, x, force_prob=0.0):
        features = self.net(x)
        out = self.head(features)
        
        track_outputs, g_ret, g_halt = self._split_outputs(out)
        
        if self.training and force_prob > 0.0:
            mask = (torch.rand_like(g_ret) < force_prob).float()
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret
        
        return track_outputs, g_ret, g_halt

class HoloLink(nn.Module):
    def __init__(self, config: ANAConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.key_dim = config.key_dim
        self.d_model = config.d_model
        self.decay = config.hololink_decay
        
        self.q_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.d_model, bias=False)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        
        if config.orthogonal_init:
            nn.init.orthogonal_(self.k_proj.weight)
        
        if config.use_learned_binding:
            self.binding_strength = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('binding_strength', torch.tensor(1.0))

    def forward(self, x_t, h_t, M_prev):
        batch_size = x_t.size(0)
        
        if M_prev is None:
            d_val = self.v_proj.out_features
            M_prev = torch.zeros(batch_size, self.key_dim, d_val, device=x_t.device)
        
        k_t = self.k_proj(h_t)
        k_t = F.normalize(k_t, p=2, dim=-1)
        v_t = self.v_proj(h_t)
        
        strength = F.softplus(self.binding_strength)
        update = strength * torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))
        
        M_t = self.decay * M_prev + update
        
        q_t = self.q_proj(x_t)
        q_t = F.normalize(q_t, p=2, dim=-1)
        
        retrieved = torch.bmm(q_t.unsqueeze(1), M_t).squeeze(1)
        retrieved = self.out_proj(retrieved)
        retrieved = self.norm(retrieved)
        return retrieved, M_t

    def forward_sequence(self, x, h):
        k = self.k_proj(h)
        k = F.normalize(k, p=2, dim=-1)
        v = self.v_proj(h)
        
        strength = F.softplus(self.binding_strength)
        update = strength * torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        
        M_seq = torch.cumsum(update, dim=1)
        
        q = self.q_proj(x)
        q = F.normalize(q, p=2, dim=-1)
        
        retrieved = torch.matmul(q.unsqueeze(-2), M_seq).squeeze(-2)
        retrieved = self.out_proj(retrieved)
        retrieved = self.norm(retrieved)
        return retrieved, M_seq

class ANAModel(nn.Module):
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if config.use_position_encoding:
            self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(config.max_seq_len, config.d_model))
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer_dict = nn.ModuleDict()
            
            if config.use_controller:
                layer_dict['controller'] = HyperController(config)
            
            layer_dict['tracks'] = nn.ModuleList([
                LinearRecurrentUnit(config) for _ in range(config.track_count)
            ])
            
            if config.use_hololink:
                layer_dict['holo'] = HoloLink(config, input_dim=config.state_dim * config.track_count)
            
            if config.track_count >= 2:
                with torch.no_grad():
                    layer_dict['tracks'][0].static_alpha_logit.fill_(-3.0)
                    layer_dict['tracks'][0].static_beta_logit.fill_(2.0)
                    layer_dict['tracks'][1].static_alpha_logit.fill_(3.0)
                    layer_dict['tracks'][1].static_beta_logit.fill_(0.0)
            
            self.layers.append(layer_dict)
        
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

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        if self.config.use_parallel_scan and self.config.max_thinking_steps == 0:
            return self._forward_parallel(input_ids, return_info, force_prob)
        else:
            return self._forward_sequential(input_ids, return_info, force_prob)

    def _forward_parallel(self, input_ids, return_info=False, force_prob=0.0):
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        info_log = []
        
        for i, layer in enumerate(self.layers):
            track_outputs = None
            g_ret = None
            
            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, _ = ctl(x, force_prob=force_prob)
            
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
                
                yt, ht = track._forward_sequence(x, dynamic_gates=gates)
                track_states.append(ht)
                track_results.append(yt)
                if mix is not None:
                    track_mix_logits.append(mix)
                else:
                    track_mix_logits.append(torch.zeros_like(yt[..., :1]))
            
            stacked_results = torch.stack(track_results, dim=2)
            stacked_mix = torch.stack(track_mix_logits, dim=2)
            mix_weights = torch.softmax(stacked_mix, dim=2)
            layer_out = (stacked_results * mix_weights).sum(dim=2)
            
            qt = 0
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(track_states, dim=-1)
                qt, _ = holo.forward_sequence(x, ht_combined)
            
            if self.config.use_controller and self.config.use_hololink and g_ret is not None:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = layer_out + ret_gate * qt
            elif self.config.use_hololink:
                layer_out = layer_out + qt
            
            x = x + layer_out
            
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

    def _forward_sequential(self, input_ids, return_info=False, force_prob=0.0):
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        batch, seq_len, _ = x.shape
        
        h_states = [[None] * self.config.track_count for _ in range(self.config.num_layers)]
        m_states = [None] * len(self.layers)
        
        final_layer_outputs = []
        info_log = []
        total_thinking_steps = 0
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                tracks = layer['tracks']
                
                steps_taken = 0
                max_steps = self.config.max_thinking_steps + 1
                
                while steps_taken < max_steps:
                    track_outputs = None
                    g_ret = None
                    g_halt = None
                    
                    if self.config.use_controller:
                        ctl = layer['controller']
                        track_outputs, g_ret, g_halt = ctl(xt, force_prob=force_prob)
                    
                    should_halt = False
                    if self.config.use_act and g_halt is not None and steps_taken > 0:
                        halt_prob = torch.sigmoid(g_halt)
                        if halt_prob.mean().item() > 0.5:
                            should_halt = True
                    
                    if should_halt:
                        break
                    
                    new_h_states_layer = []
                    track_results = []
                    track_mix_logits = []
                    
                    for t_idx, track in enumerate(tracks):
                        gates = None
                        mix = None
                        if track_outputs is not None:
                            g_alpha, g_beta, g_mix = track_outputs[t_idx]
                            gates = (g_alpha, g_beta)
                            mix = g_mix
                        
                        h_prev = h_states[i][t_idx]
                        yt, ht = track(xt, h_prev, dynamic_gates=gates)
                        
                        new_h_states_layer.append(ht)
                        track_results.append(yt)
                        
                        if mix is not None:
                            track_mix_logits.append(mix)
                        else:
                            track_mix_logits.append(torch.zeros(batch, 1, device=x.device))
                    
                    h_states[i] = new_h_states_layer
                    
                    stacked_results = torch.stack(track_results, dim=1)
                    stacked_mix = torch.stack(track_mix_logits, dim=1)
                    mix_weights = torch.softmax(stacked_mix, dim=1)
                    layer_out = (stacked_results * mix_weights).sum(dim=1)
                    
                    qt = 0
                    if self.config.use_hololink:
                        holo = layer['holo']
                        ht_combined = torch.cat(h_states[i], dim=-1)
                        qt, mt_next = holo(xt, ht_combined, m_states[i])
                        m_states[i] = mt_next
                    
                    if self.config.use_controller and self.config.use_hololink and g_ret is not None:
                        ret_gate = torch.sigmoid(g_ret)
                        layer_out = layer_out + ret_gate * qt
                    elif self.config.use_hololink:
                        layer_out = layer_out + qt
                    
                    xt = xt + layer_out
                    steps_taken += 1
                    total_thinking_steps += 1
                
                if return_info and i == 0 and t < 10:
                    stats = {}
                    if track_outputs is not None:
                        stats['ga_0'] = track_outputs[0][0].mean().item()
                    if g_ret is not None:
                        stats['ret_gate'] = torch.sigmoid(g_ret).mean().item()
                    if self.config.max_thinking_steps > 0:
                        stats['thinking_steps'] = steps_taken
                    info_log.append(stats)
            
            final_layer_outputs.append(xt)
        
        output_seq = torch.stack(final_layer_outputs, dim=1)
        output_seq = self.norm(output_seq)
        logits = self.output_head(output_seq)
        return logits, info_log

class BaselineSSM(nn.Module):
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            LinearRecurrentUnit(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        
        if self.config.use_parallel_scan:
            for i, lru in enumerate(self.layers):
                yt, _ = lru._forward_sequence(x, dynamic_gates=None)
                x = yt + x
            output_seq = self.norm(x)
        else:
            h_states = [None] * len(self.layers)
            final_layer_outputs = []
            
            for t in range(seq_len):
                xt = x[:, t, :]
                for i, lru in enumerate(self.layers):
                    yt, ht_next = lru(xt, h_states[i], dynamic_gates=None)
                    h_states[i] = ht_next
                    xt = yt + xt
                final_layer_outputs.append(xt)
            
            output_seq = torch.stack(final_layer_outputs, dim=1)
            output_seq = self.norm(output_seq)
        
        logits = self.output_head(output_seq)
        return logits, []
