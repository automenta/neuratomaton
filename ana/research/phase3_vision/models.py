import torch
import torch.nn as nn
from ana.models import LinearRecurrentUnit, HoloLink, HyperController
from ana.config import ANAConfig

class ANAVisionModel(nn.Module):
    """
    ANA adapted for Vision tasks.
    Treats image patches as a sequence of tokens.
    """
    def __init__(self, config: ANAConfig, num_classes=1000):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.d_model = config.d_model

        # Patch Embedding: (B, C, H, W) -> (B, Seq, D)
        # Assuming 3 channels
        patch_dim = 3 * self.patch_size * self.patch_size
        self.patch_embed = nn.Linear(patch_dim, self.d_model)

        # Reuse ANA core components
        # We can't use ANAModel directly because it has an Embedding layer.
        # We will reconstruct the layers similarly.

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
        self.head = nn.Linear(config.d_model, num_classes) # Classification head

    def forward(self, images):
        # images: (B, 3, H, W)
        b, c, h, w = images.shape
        p = self.patch_size

        # Patchify
        # (B, 3, H, W) -> (B, 3, H/P, P, W/P, P) -> (B, H/P * W/P, 3*P*P)
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, c, -1, p, p).permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, -1, c * p * p)

        x = self.patch_embed(patches)

        # Add position encoding (simplified)
        seq_len = x.size(1)
        # ... could add learned pos embed here ...

        # Process layers (simplified duplicate of ANAModel logic)
        # For brevity, I'll just run tracks.
        # In a real implementation, we should refactor ANAModel to separate the "Block" logic.

        for layer in self.layers:
            # Simple aggregation for now
            track_outs = []
            for track in layer['tracks']:
                yt, _ = track.forward_sequence(x)
                track_outs.append(yt)

            # Mean of tracks
            x = x + torch.stack(track_outs).mean(dim=0)

        # Global Average Pooling
        x = x.mean(dim=1)
        x = self.norm(x)
        logits = self.head(x)

        return logits

if __name__ == "__main__":
    config = ANAConfig(d_model=64, patch_size=16, image_size=224, num_layers=2)
    model = ANAVisionModel(config, num_classes=10)
    img = torch.randn(1, 3, 224, 224)
    out = model(img)
    print(f"Vision Output: {out.shape}")
