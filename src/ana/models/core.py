"""
ANA Core Models

Implementation of Adaptive Neural Automaton with HoloLink associative memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import ANAConfig


def parallel_scan_cumsum(u, log_alpha, beta):
    """
    O(N) parallel scan using cumsum trick.

    h_t = alpha_t * h_{t-1} + beta_t * u_t

    Using log-space cumsum:
    h_t = exp(C_t) * S_t
    where C_t = cumsum(log(alpha_t))
          S_t = cumsum(beta_t * u_t * exp(-C_t))

    Numerically stable version using float32 for internal computation.
    """
    log_alpha_f32 = log_alpha.float().clamp(max=0, min=-20)  # Clamp to avoid underflow
    beta_f32 = beta.float().clamp(min=-10, max=10)  # Clamp beta
    u_f32 = u.float().clamp(min=-10, max=10)  # Clamp input

    C = torch.cumsum(log_alpha_f32, dim=1)
    term = beta_f32 * u_f32 * torch.exp(-C)
    S = torch.cumsum(term, dim=1)
    h = torch.exp(C) * S

    # Final NaN check and replacement
    h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)

    return h.to(u.dtype)


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
        a = a.clone()
        b = b.clone()

    # Log iterations
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

        # Initialize projections with small values for stability
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)

        # Static parameters - conservative initialization
        # alpha close to 0.9 (decay), beta close to 0.1 (input weight)
        self.static_alpha_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(1.5, 2.5))
        self.static_beta_logit = nn.Parameter(torch.Tensor(self.state_dim).uniform_(-1.5, -0.5))

    def forward(self, x, h_prev=None, dynamic_gates=None):
        batch_size = x.size(0)

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.state_dim, device=x.device)

        u_t = self.input_proj(x)

        if dynamic_gates is not None:
            gate_alpha, gate_beta = dynamic_gates
            # Clamp gates to prevent extreme values
            gate_alpha = gate_alpha.clamp(-5, 5)
            gate_beta = gate_beta.clamp(-5, 5)
            alpha = torch.sigmoid(self.static_alpha_logit + gate_alpha)
            beta = torch.sigmoid(self.static_beta_logit + gate_beta)
        else:
            alpha = torch.sigmoid(self.static_alpha_logit)
            beta = torch.sigmoid(self.static_beta_logit)

        h_t = alpha * h_prev + beta * u_t

        # NaN protection
        h_t = torch.where(torch.isnan(h_t) | torch.isinf(h_t), torch.zeros_like(h_t), h_t)

        y_t = self.output_proj(h_t)

        return y_t, h_t

    def forward_sequence(self, x, dynamic_gates=None):
        batch_size, seq_len, _ = x.shape
        u = self.input_proj(x)

        if dynamic_gates is not None:
            g_alpha, g_beta = dynamic_gates
            # Clamp gate values to prevent extreme values
            g_alpha = g_alpha.clamp(-5, 5)
            g_beta = g_beta.clamp(-5, 5)
            alpha = torch.sigmoid(self.static_alpha_logit + g_alpha)
            beta = torch.sigmoid(self.static_beta_logit + g_beta)
        else:
            alpha = torch.sigmoid(self.static_alpha_logit).view(1, 1, -1).expand(batch_size, seq_len, -1)
            beta = torch.sigmoid(self.static_beta_logit).view(1, 1, -1).expand(batch_size, seq_len, -1)

        h_init = torch.zeros(batch_size, self.state_dim, device=x.device)

        if self.config.use_parallel_scan:
            log_alpha = torch.log(alpha.clamp(min=1e-7, max=1-1e-7))
            h_seq = parallel_scan_cumsum(u, log_alpha, beta)
        else:
            h_seq = lru_scan(u, alpha, beta, h_init)

        y_seq = self.output_proj(h_seq)
        return y_seq, h_seq


class HyperController(nn.Module):
    """
    HyperController for Multi-Track ANA + HoloLink.
    Outputs per track: alpha_gate, beta_gate, mix_logit
    Plus: retrieval_gate, halt_logit
    """
    def __init__(self, config: ANAConfig, hidden_dim=None):
        super().__init__()
        self.config = config
        if hidden_dim is None:
            hidden_dim = max(64, config.d_model // 4)  # Scale with model

        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # 2 scalars (alpha, beta) + 1 scalar (mix) per track
        # + 1 scalar (retrieval) + 1 scalar (halt)
        self.output_dim = config.track_count * 3 + 2
        self.head = nn.Linear(hidden_dim, self.output_dim)

        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)

    def split_outputs(self, out):
        # out: [..., output_dim]
        # Structure: [Track1_Alpha, Track1_Beta, Track1_Mix, Track2..., Ret, Halt]

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

        track_outputs, g_ret, g_halt = self.split_outputs(out)

        if self.training and force_prob > 0.0:
            mask = (torch.rand_like(g_ret) < force_prob).float()
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret

        return track_outputs, g_ret, g_halt

    def forward_sequence(self, x, force_prob=0.0):
        features = self.net(x)
        out = self.head(features)

        track_outputs, g_ret, g_halt = self.split_outputs(out)

        if self.training and force_prob > 0.0:
            mask = (torch.rand_like(g_ret) < force_prob).float()
            g_ret = mask * 5.0 + (1.0 - mask) * g_ret

        return track_outputs, g_ret, g_halt


class HoloLink(nn.Module):
    """
    Associative Memory Module using Matrix Accumulation / Linear Attention.
    Includes learned binding strength.
    """
    def __init__(self, config: ANAConfig, input_dim: int):
        super().__init__()
        self.key_dim = config.key_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.d_model, bias=False)

        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

        self.binding_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_t, h_t, M_prev):
        batch_size = x_t.size(0)

        if M_prev is None:
            d_val = self.v_proj.out_features
            M_prev = torch.zeros(batch_size, self.key_dim, d_val, device=x_t.device)

        k_t = self.k_proj(h_t)
        k_t = torch.nn.functional.normalize(k_t + 1e-8, p=2, dim=-1)  # Epsilon for stability
        v_t = self.v_proj(h_t)

        strength = F.softplus(self.binding_strength).clamp(max=10)  # Limit strength
        update = strength * torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))

        M_t = M_prev + update

        q_t = self.q_proj(x_t)
        q_t = torch.nn.functional.normalize(q_t + 1e-8, p=2, dim=-1)  # Epsilon for stability

        retrieved = torch.bmm(q_t.unsqueeze(1), M_t).squeeze(1)

        # NaN protection
        retrieved = torch.where(torch.isnan(retrieved) | torch.isinf(retrieved),
                                torch.zeros_like(retrieved), retrieved)

        retrieved = self.out_proj(retrieved)
        retrieved = self.norm(retrieved)
        return retrieved, M_t

    def forward_sequence(self, x, h):
        k = self.k_proj(h)
        k = torch.nn.functional.normalize(k + 1e-8, p=2, dim=-1)  # Epsilon for stability
        v = self.v_proj(h)

        strength = F.softplus(self.binding_strength).clamp(max=10)  # Limit strength
        update = strength * torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))

        M_seq = torch.cumsum(update, dim=1)

        q = self.q_proj(x)
        q = torch.nn.functional.normalize(q + 1e-8, p=2, dim=-1)  # Epsilon for stability

        retrieved = torch.matmul(q.unsqueeze(-2), M_seq).squeeze(-2)

        # NaN protection
        retrieved = torch.where(torch.isnan(retrieved) | torch.isinf(retrieved),
                                torch.zeros_like(retrieved), retrieved)

        retrieved = self.out_proj(retrieved)
        retrieved = self.norm(retrieved)
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

        # Position encoding for algorithmic reasoning
        self.position_encoding = nn.Embedding(config.max_position, config.d_model)

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

            if config.track_count == 2:
                with torch.no_grad():
                    # Track A (Fast/Reflex) - alpha ~0.7, beta ~0.3
                    layer_dict['tracks'][0].static_alpha_logit.fill_(1.0)
                    layer_dict['tracks'][0].static_beta_logit.fill_(-1.0)
                    # Track B (Slow/Reasoning) - alpha ~0.95, beta ~0.05
                    layer_dict['tracks'][1].static_alpha_logit.fill_(3.0)
                    layer_dict['tracks'][1].static_beta_logit.fill_(-3.0)

            self.layers.append(layer_dict)

        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward_features_parallel(self, input_ids=None, inputs_embeds=None, return_info=False, force_prob=0.0):
        if self.config.max_thinking_steps > 0:
            return self.forward_features_sequential(input_ids, inputs_embeds, return_info, force_prob)

        if inputs_embeds is not None:
            x = inputs_embeds
            input_shape = inputs_embeds.shape[:-1]
        else:
            x = self.embedding(input_ids)
            input_shape = input_ids.shape

        # Add position encoding
        batch, seq_len = input_shape
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        pos_encoding = self.position_encoding(pos_ids)
        x = x + pos_encoding
        info_log = {'layers': [{} for _ in range(self.config.num_layers)]}

        for i, layer in enumerate(self.layers):
            # 1. Controller
            track_outputs = None
            g_ret = None
            g_halt = None

            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, g_halt = ctl.forward_sequence(x, force_prob=force_prob)

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

            if return_info:
                # Capture detailed stats for visualization
                layer_stats = info_log['layers'][i]

                # mix_weights: [Batch, Seq, Tracks, 1] -> [Batch, Seq, Tracks]
                layer_stats['mix_weights'] = mix_weights.detach().cpu().squeeze(-1)

                # Retrieval gate
                if g_ret is not None:
                    layer_stats['ret_gate'] = torch.sigmoid(g_ret).detach().cpu() # [Batch, Seq, 1]

                # Halt logits (if available)
                if g_halt is not None:
                    layer_stats['halt_gate'] = torch.sigmoid(g_halt).detach().cpu() # [Batch, Seq, 1]

        x = self.norm(x)
        return x, info_log

    def forward_sequential(self, input_ids=None, inputs_embeds=None, return_info=False, force_prob=0.0):
        return self.forward_features_sequential(input_ids, inputs_embeds, return_info, force_prob)

    def forward_features_sequential(self, input_ids=None, inputs_embeds=None, return_info=False, force_prob=0.0):
        if inputs_embeds is not None:
            x = inputs_embeds
            input_shape = inputs_embeds.shape[:-1]
        else:
            x = self.embedding(input_ids)
            input_shape = input_ids.shape

        # Add position encoding
        batch, seq_len = input_shape
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        pos_encoding = self.position_encoding(pos_ids)
        x = x + pos_encoding
        batch, seq_len, _ = x.shape

        h_states = [[None] * self.config.track_count for _ in range(self.config.num_layers)]
        m_states = [None] * len(self.layers)

        final_layer_outputs = []
        info_log = {'layers': [{'mix_weights': [], 'ret_gate': [], 'halt_gate': []} for _ in range(self.config.num_layers)]}

        for t in range(seq_len):
            xt = x[:, t, :]

            for i, layer in enumerate(self.layers):
                tracks = layer['tracks']

                # Thinking Steps Loop
                # We loop up to max_thinking_steps + 1 (the 1 is the actual processing)
                # But actually, thinking steps implies we refine the state without consuming new input.
                # Simplified ACT (Adaptive Computation Time):
                # We reuse the same xt? No, we update xt.
                # Let's say we have an internal recurrence.

                steps_taken = 0
                while steps_taken <= self.config.max_thinking_steps:
                    # 1. Controller
                    track_outputs = None
                    g_ret = None
                    g_halt = None

                    if self.config.use_controller:
                        ctl = layer['controller']
                        track_outputs, g_ret, g_halt = ctl(xt, force_prob=force_prob)

                    # Check halt
                    should_halt = False
                    if self.config.max_thinking_steps > 0:
                         if g_halt is not None:
                             halt_prob = torch.sigmoid(g_halt)
                             # If halt prob > 0.5, we stop thinking (batch-wise?)
                             # For simplicity in batch training, we just run fixed steps or average?
                             # Standard ACT is complex.
                             # Simplified: We just run max_thinking_steps fixed for now if > 0.
                             # Or we define it as: we run at least 1 step.
                             pass

                    # 2. Update Tracks
                    track_results = []
                    track_mix_logits = []

                    # Temporary state update for thinking?
                    # If we "think", do we update the permanent state h_t?
                    # Usually ACT updates the hidden state in place.

                    new_h_states_layer = []

                    for t_idx, track in enumerate(tracks):
                        gates = None
                        mix = None
                        if track_outputs is not None:
                            g_alpha, g_beta, g_mix = track_outputs[t_idx]
                            gates = (g_alpha, g_beta)
                            mix = g_mix

                        # Use current h_state
                        h_prev = h_states[i][t_idx]
                        yt, ht = track(xt, h_prev, dynamic_gates=gates)

                        new_h_states_layer.append(ht)
                        track_results.append(yt)

                        if mix is not None:
                            track_mix_logits.append(mix)
                        else:
                            track_mix_logits.append(torch.zeros(batch, 1, device=x.device))

                    # Update states (in-place for the next micro-step)
                    h_states[i] = new_h_states_layer

                    # Mixing
                    stacked_results = torch.stack(track_results, dim=1)
                    stacked_mix = torch.stack(track_mix_logits, dim=1)
                    mix_weights = torch.softmax(stacked_mix, dim=1)

                    layer_out = (stacked_results * mix_weights).sum(dim=1)

                    # 3. HoloLink (Only update memory once? Or every micro-step?)
                    # If we update memory every micro-step, we write multiple times per token.
                    # Maybe only read?
                    # Let's say we update everything.

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

                    # Residual update of xt for next layer OR next thinking step
                    xt = xt + layer_out

                    steps_taken += 1

                    # Logic to break loop
                    if self.config.max_thinking_steps == 0:
                        break

                    # If using halt logic, we would check here.
                    # For this implementation, we treat max_thinking_steps as "Extra steps".
                    # So loop runs max_thinking_steps + 1 times?
                    # Plan said: "run multiple internal updates... until max_thinking_steps is reached"
                    # Let's interpret max_thinking_steps as *additional* steps.

                    if steps_taken > self.config.max_thinking_steps:
                        break

                if return_info:
                    layer_stats = info_log['layers'][i]

                    # Accumulate for sequence. mix_weights is [Batch, Tracks, 1]
                    layer_stats['mix_weights'].append(mix_weights.detach().cpu())

                    if g_ret is not None:
                        layer_stats['ret_gate'].append(torch.sigmoid(g_ret).detach().cpu())

                    if g_halt is not None:
                         layer_stats['halt_gate'].append(torch.sigmoid(g_halt).detach().cpu())

            final_layer_outputs.append(xt)

        # Post-process info_log to stack lists into tensors
        if return_info:
            for l_idx in range(self.config.num_layers):
                l_stats = info_log['layers'][l_idx]
                if l_stats['mix_weights']:
                    # Stack -> [Batch, Seq, Tracks, 1] -> Squeeze -> [Batch, Seq, Tracks]
                    l_stats['mix_weights'] = torch.stack(l_stats['mix_weights'], dim=1).squeeze(-1)

                if l_stats['ret_gate']:
                    # Stack -> [Batch, Seq, 1]
                    l_stats['ret_gate'] = torch.stack(l_stats['ret_gate'], dim=1)

                if l_stats['halt_gate']:
                    # Stack -> [Batch, Seq, 1]
                    l_stats['halt_gate'] = torch.stack(l_stats['halt_gate'], dim=1)

        output_seq = torch.stack(final_layer_outputs, dim=1)
        output_seq = self.norm(output_seq)
        return output_seq, info_log

    def forward_features_step(self, inputs_embeds, state=None):
        """
        Single step forward for RL/Series generation.
        Args:
            inputs_embeds: [Batch, D_Model] (No time dim)
            state: Tuple (h_states, m_states)
                h_states: List[List[Tensor]] - [Layer][Track] -> [Batch, State_Dim]
                m_states: List[Tensor] - [Layer] -> [Batch, Key_Dim, Val_Dim]
        """
        x = inputs_embeds
        batch_size = x.size(0)

        if state is None:
            h_states = [[None] * self.config.track_count for _ in range(self.config.num_layers)]
            m_states = [None] * self.config.num_layers
        else:
            h_states, m_states = state

        new_h_states = []
        new_m_states = []

        for i, layer in enumerate(self.layers):
            tracks = layer['tracks']

            # 1. Controller
            track_outputs = None
            g_ret = None

            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, _ = ctl(x)

            # 2. Tracks
            track_results = []
            track_mix_logits = []
            layer_h_states = []

            for t_idx, track in enumerate(tracks):
                gates = None
                mix = None
                if track_outputs is not None:
                    g_alpha, g_beta, g_mix = track_outputs[t_idx]
                    gates = (g_alpha, g_beta)
                    mix = g_mix

                # State handling
                h_prev = h_states[i][t_idx] if h_states[i][t_idx] is not None else None
                yt, ht = track(x, h_prev, dynamic_gates=gates)

                layer_h_states.append(ht)
                track_results.append(yt)

                if mix is not None:
                    track_mix_logits.append(mix)
                else:
                    track_mix_logits.append(torch.zeros(batch_size, 1, device=x.device))

            new_h_states.append(layer_h_states)

            # Mixing
            stacked_results = torch.stack(track_results, dim=1)
            stacked_mix = torch.stack(track_mix_logits, dim=1)
            mix_weights = torch.softmax(stacked_mix, dim=1)
            layer_out = (stacked_results * mix_weights).sum(dim=1)

            # 3. HoloLink
            qt = 0
            m_next = None
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(layer_h_states, dim=-1)
                m_prev = m_states[i]
                qt, m_next = holo(x, ht_combined, m_prev)

            new_m_states.append(m_next)

            # 4. Merge
            if self.config.use_controller and self.config.use_hololink:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = layer_out + ret_gate * qt
            elif self.config.use_hololink:
                layer_out = layer_out + qt

            x = x + layer_out # Residual

        x = self.norm(x)
        return x, (new_h_states, new_m_states)

    def forward_parallel(self, input_ids=None, inputs_embeds=None, return_info=False, force_prob=0.0):
        x, info_log = self.forward_features_parallel(input_ids, inputs_embeds, return_info, force_prob)
        logits = self.output_head(x)
        return logits, info_log

    def forward_sequential(self, input_ids=None, inputs_embeds=None, return_info=False, force_prob=0.0):
        x, info_log = self.forward_features_sequential(input_ids, inputs_embeds, return_info, force_prob)
        logits = self.output_head(x)
        return logits, info_log

    def forward_step(self, inputs_embeds, state=None):
        x, new_state = self.forward_features_step(inputs_embeds, state)
        logits = self.output_head(x)
        return logits, new_state

    def forward(self, input_ids=None, inputs_embeds=None, return_info=False, force_prob=0.0):
        if self.config.use_parallel_scan and self.config.max_thinking_steps == 0:
             return self.forward_parallel(input_ids, inputs_embeds, return_info, force_prob)
        else:
             return self.forward_sequential(input_ids, inputs_embeds, return_info, force_prob)


class BaselineSSM(nn.Module):
    """
    Single-track SSM baseline for fair comparison.
    Same parameter count as ANA but without multi-track or HoloLink.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            LinearRecurrentUnit(config)
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        x = self.embedding(input_ids)

        for lru in self.layers:
            yt, _ = lru.forward_sequence(x)
            x = x + yt

        x = self.norm(x)
        logits = self.output_head(x)
        return logits, []
class ANARLAgent(nn.Module):
    """
    Phase 4: RL Agent wrapping ANA.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.ana = ANAModel(config)

        self.input_proj = nn.Linear(config.observation_space, config.d_model)
        self.policy_head = nn.Linear(config.d_model, config.action_space)
        self.value_head = nn.Linear(config.d_model, 1)

    def forward(self, obs, state=None):
        """
        Single step forward for RL.
        obs: [Batch, Obs_Dim]
        """
        x = self.input_proj(obs)
        features, next_state = self.ana.forward_features_step(x, state)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value, next_state

class ANASeriesModel(nn.Module):
    """
    Phase 5: Time Series Model wrapping ANA.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.ana = ANAModel(config)

        self.input_proj = nn.Linear(config.series_dim, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.series_dim)

    def forward(self, x, state=None):
        """
        Single step forward for series prediction.
        x: [Batch, Series_Dim]
        """
        embeds = self.input_proj(x)
        features, next_state = self.ana.forward_features_step(embeds, state)
        pred = self.output_proj(features)

        return pred, next_state

    def forward_sequence(self, x_seq):
        """
        Process a sequence for training.
        x_seq: [Batch, Seq, Series_Dim]
        """
        embeds = self.input_proj(x_seq)

        # Use parallel or sequential features
        if self.config.use_parallel_scan and self.config.max_thinking_steps == 0:
             features, _ = self.ana.forward_features_parallel(inputs_embeds=embeds)
        else:
             features, _ = self.ana.forward_features_sequential(inputs_embeds=embeds)

        pred_seq = self.output_proj(features)
        return pred_seq
