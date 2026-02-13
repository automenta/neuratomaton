import torch
import torch.nn as nn
from ana.models import LinearRecurrentUnit, HoloLink, HyperController
from ana.config import ANAConfig

class ANARLAgent(nn.Module):
    """
    ANA Agent for RL tasks.
    Takes observation vector, outputs action logits and value estimate.
    Supports both sequence processing (training) and step-wise inference (rollout).
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        self.obs_dim = config.observation_space
        self.act_dim = config.action_space

        self.input_proj = nn.Linear(self.obs_dim, self.d_model)

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer_dict = nn.ModuleDict()
            if config.use_controller:
                layer_dict['controller'] = HyperController(config)

            layer_dict['tracks'] = nn.ModuleList([
                LinearRecurrentUnit(config) for _ in range(config.track_count)
            ])

            # HoloLink for episodic memory
            if config.use_hololink:
                layer_dict['holo'] = HoloLink(config, input_dim=config.state_dim * config.track_count)

            self.layers.append(layer_dict)

        self.norm = nn.LayerNorm(self.d_model)

        # Actor-Critic heads
        self.actor = nn.Linear(self.d_model, self.act_dim)
        self.critic = nn.Linear(self.d_model, 1)

    def forward(self, obs, prev_hidden=None):
        """
        Dispatches to sequence or step processing based on input dimensions.
        obs: (B, T, ObsDim) or (B, ObsDim)
        """
        if obs.dim() == 2:
            return self.forward_step(obs, prev_hidden)
        else:
            return self.forward_sequence(obs)

    def forward_step(self, obs, prev_hidden=None):
        """
        Step-wise forward pass for RL rollout.
        obs: (B, ObsDim)
        prev_hidden: tuple(h_tracks_list, m_holo_list)
        """
        x = self.input_proj(obs) # (B, D)
        batch = x.size(0)

        if prev_hidden is None:
            # Init empty states
            # h_tracks: list of list of tensors (layer -> track -> state)
            h_tracks = [[None] * self.config.track_count for _ in range(self.config.num_layers)]
            # m_holo: list of tensors (layer -> state)
            m_holo = [None] * self.config.num_layers
        else:
            h_tracks, m_holo = prev_hidden

        new_h_tracks = [] # structure: layer -> track -> state
        new_m_holo = []   # structure: layer -> state

        for i, layer in enumerate(self.layers):
            # 1. Controller (Single Step)
            # Controller forward expects (B, D) -> returns (B, ...).
            # HyperController has `forward` (step) and `forward_sequence` methods.
            track_outputs = None
            g_ret = None
            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, _ = ctl(x)

            # 2. Tracks (Single Step)
            track_results = []
            track_mix_logits = []
            current_layer_h_tracks_new = []

            current_layer_h_tracks_prev = h_tracks[i]

            for t_idx, track in enumerate(layer['tracks']):
                gates = None
                mix = None
                if track_outputs is not None:
                    g_alpha, g_beta, g_mix = track_outputs[t_idx]
                    gates = (g_alpha, g_beta)
                    mix = g_mix

                # Retrieve previous state
                h_prev_t = current_layer_h_tracks_prev[t_idx]

                # Use single step forward of LRU
                yt, ht = track(x, h_prev=h_prev_t, dynamic_gates=gates)

                current_layer_h_tracks_new.append(ht)
                track_results.append(yt)

                if mix is not None:
                    track_mix_logits.append(mix)
                else:
                    track_mix_logits.append(torch.zeros_like(yt[..., :1]))

            new_h_tracks.append(current_layer_h_tracks_new)

            # Mixing
            stacked_results = torch.stack(track_results, dim=1) # (B, Tracks, D)
            stacked_mix = torch.stack(track_mix_logits, dim=1)
            mix_weights = torch.softmax(stacked_mix, dim=1)
            layer_out = (stacked_results * mix_weights).sum(dim=1)

            # 3. HoloLink (Single Step)
            qt = 0
            m_prev = m_holo[i]
            m_next = m_prev

            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(current_layer_h_tracks_new, dim=-1)
                # HoloLink.forward is single step
                qt, m_next = holo(x, ht_combined, m_prev)

            new_m_holo.append(m_next)

            # 4. Merge
            if self.config.use_controller and self.config.use_hololink:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = layer_out + ret_gate * qt
            elif self.config.use_hololink:
                layer_out = layer_out + qt

            x = x + layer_out

        x = self.norm(x)
        policy_logits = self.actor(x)
        value = self.critic(x)

        return policy_logits, value, (new_h_tracks, new_m_holo)

    def forward_sequence(self, obs_seq):
        """
        Sequence processing for training (uses parallel scan if config enabled).
        obs_seq: (B, T, ObsDim)
        """
        x = self.input_proj(obs_seq) # (B, T, D)

        for i, layer in enumerate(self.layers):
            # 1. Controller (Sequence)
            track_outputs = None
            g_ret = None
            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, _ = ctl.forward_sequence(x)

            # 2. Tracks (Sequence)
            track_states = [] # To store ht sequence for HoloLink input
            track_results = []
            track_mix_logits = []

            for t_idx, track in enumerate(layer['tracks']):
                gates = None
                mix = None
                if track_outputs is not None:
                    g_alpha, g_beta, g_mix = track_outputs[t_idx]
                    gates = (g_alpha, g_beta)
                    mix = g_mix

                yt, ht_seq = track.forward_sequence(x, dynamic_gates=gates)
                track_states.append(ht_seq)
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

            # 3. HoloLink (Sequence)
            qt = 0
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(track_states, dim=-1) # (B, T, Dim)
                qt, _ = holo.forward_sequence(x, ht_combined)

            # 4. Merge
            if self.config.use_controller and self.config.use_hololink:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = layer_out + ret_gate * qt
            elif self.config.use_hololink:
                layer_out = layer_out + qt

            x = x + layer_out

        x = self.norm(x)
        policy_logits = self.actor(x)
        value = self.critic(x)

        return policy_logits, value
