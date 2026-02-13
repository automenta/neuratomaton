import torch
import torch.nn as nn
from ana.models import LinearRecurrentUnit, HoloLink, HyperController
from ana.config import ANAConfig

class ANASeriesModel(nn.Module):
    """
    ANA adapted for 1D Time Series / Scientific Data.
    Features:
    - Continuous input projection
    - Long-range dependency handling via LRU
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.series_dim
        self.d_model = config.d_model

        # Projection: (B, Seq, In) -> (B, Seq, D)
        self.input_proj = nn.Linear(self.input_dim, self.d_model)

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
            self.layers.append(layer_dict)

        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, self.input_dim) # Predict next value(s)

    def forward(self, x):
        # x: (B, Seq, In)
        h = self.input_proj(x)

        for i, layer in enumerate(self.layers):
            # Controller
            track_outputs = None
            g_ret = None
            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, _ = ctl.forward_sequence(h)

            # Tracks
            track_states = []
            track_results = []
            track_mix_logits = []

            for t_idx, track in enumerate(layer['tracks']):
                gates = None
                mix = None
                if track_outputs is not None:
                    g_alpha, g_beta, g_mix = track_outputs[t_idx]
                    gates = (g_alpha, g_beta)
                    mix = g_mix

                yt, ht = track.forward_sequence(h, dynamic_gates=gates)
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

            # HoloLink
            qt = 0
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(track_states, dim=-1)
                qt, _ = holo.forward_sequence(h, ht_combined)

            # Merge
            if self.config.use_controller and self.config.use_hololink:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = layer_out + ret_gate * qt
            elif self.config.use_hololink:
                layer_out = layer_out + qt

            h = h + layer_out

        h = self.norm(h)
        pred = self.head(h)
        return pred

if __name__ == "__main__":
    config = ANAConfig(series_dim=1, d_model=32)
    model = ANASeriesModel(config)
    # Batch 2, Seq 100, Dim 1
    x = torch.randn(2, 100, 1)
    out = model(x)
    print(f"Series Output: {out.shape}")
