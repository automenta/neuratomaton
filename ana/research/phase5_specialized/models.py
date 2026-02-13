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

        for layer in self.layers:
            track_outs = []
            for track in layer['tracks']:
                yt, _ = track.forward_sequence(h)
                track_outs.append(yt)
            h = h + torch.stack(track_outs).mean(dim=0)

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
