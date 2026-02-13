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
        self.head = nn.Linear(config.d_model, num_classes)

    def forward(self, images):
        b, c, h, w = images.shape
        p = self.patch_size

        # Patchify
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, c, -1, p, p).permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, -1, c * p * p)

        x = self.patch_embed(patches)

        # Run layers
        for i, layer in enumerate(self.layers):
            # 1. Controller
            track_outputs = None
            g_ret = None
            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, _ = ctl.forward_sequence(x)

            # 2. Tracks
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

            x = x + layer_out

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

        # Core layers
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
        # 1. Process Images
        b, c, h, w = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, c, -1, p, p).permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, -1, c * p * p)
        img_embeds = self.patch_embed(patches)

        # 2. Process Text
        txt_embeds = self.text_embed(text_ids)

        # 3. Concatenate
        x = torch.cat([img_embeds, txt_embeds], dim=1)

        # 4. Run ANA Core (Accurate Implementation)
        for i, layer in enumerate(self.layers):
            # Controller
            track_outputs = None
            g_ret = None
            if self.config.use_controller:
                ctl = layer['controller']
                track_outputs, g_ret, _ = ctl.forward_sequence(x)

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

            # HoloLink
            qt = 0
            if self.config.use_hololink:
                holo = layer['holo']
                ht_combined = torch.cat(track_states, dim=-1)
                qt, _ = holo.forward_sequence(x, ht_combined)

            # Merge
            if self.config.use_controller and self.config.use_hololink:
                ret_gate = torch.sigmoid(g_ret)
                layer_out = layer_out + ret_gate * qt
            elif self.config.use_hololink:
                layer_out = layer_out + qt

            x = x + layer_out

        x = self.norm(x)
        logits = self.head(x)

        t_len = txt_embeds.size(1)
        return logits[:, -t_len:, :]
