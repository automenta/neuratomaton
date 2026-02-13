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
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, c, -1, p, p).permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, -1, c * p * p)

        x = self.patch_embed(patches)

        for layer in self.layers:
            track_outs = []
            for track in layer['tracks']:
                yt, _ = track.forward_sequence(x)
                track_outs.append(yt)
            x = x + torch.stack(track_outs).mean(dim=0)

        x = x.mean(dim=1)
        x = self.norm(x)
        logits = self.head(x)

        return logits

class ANAVisionCaptioner(nn.Module):
    """
    Image Captioning using ANA.
    Prefixes image patches to text sequence.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.d_model = config.d_model

        # Patch Embedding
        patch_dim = 3 * self.patch_size * self.patch_size
        self.patch_embed = nn.Linear(patch_dim, self.d_model)

        # Text Embedding
        self.text_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Core layers (shared)
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
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, images, text_ids):
        # 1. Process Images -> Patches
        b, c, h, w = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, c, -1, p, p).permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, -1, c * p * p)
        img_embeds = self.patch_embed(patches) # (B, P_Seq, D)

        # 2. Process Text -> Embeddings
        txt_embeds = self.text_embed(text_ids) # (B, T_Seq, D)

        # 3. Concatenate (Image first, then Text)
        x = torch.cat([img_embeds, txt_embeds], dim=1)

        # 4. Run ANA Core
        for layer in self.layers:
            track_outs = []
            for track in layer['tracks']:
                yt, _ = track.forward_sequence(x)
                track_outs.append(yt)
            x = x + torch.stack(track_outs).mean(dim=0)

        x = self.norm(x)
        logits = self.head(x)

        # Return only text part logits (shifted for training usually, but here full sequence or just text part)
        # Typically we predict text given image + prev_text
        # So we return logits for the text portion
        t_len = txt_embeds.size(1)
        return logits[:, -t_len:, :]
