import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eqprop"))

from .config import BioANAConfig, get_bio_config
from .tracks import BioSpecializedTracks
from .hololink import BioHoloLink


class BioANAModel(nn.Module):
    def __init__(self, config: BioANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if config.use_position_encoding:
            self.register_buffer(
                'pos_encoding',
                self._create_sinusoidal_encoding(config.max_seq_len, config.d_model)
            )
        
        self.tracks = BioSpecializedTracks(
            d_model=config.d_model,
            syntax_dim=config.syntax_dim,
            semantic_dim=config.semantic_dim,
            logic_dim=config.logic_dim,
            use_spectral_norm=True,
        )
        
        if config.use_hebbian_memory:
            self.hololink = BioHoloLink(
                input_dim=config.total_track_dim,
                key_dim=config.hololink_key_dim,
                capacity=config.hololink_capacity,
                hebbian_lr=config.hebbian_lr,
            )
        else:
            self.hololink = None
        
        self.mixer = nn.Linear(config.total_track_dim, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.mixer.weight, gain=0.5)
        nn.init.zeros_(self.mixer.bias)
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.5)
        nn.init.zeros_(self.output_head.bias)
    
    def _add_position_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.use_position_encoding:
            return x
        seq_len = x.size(1)
        return x + self.pos_encoding[:, :seq_len, :]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_info: bool = False,
        return_energy: bool = False,
        relaxation_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        
        outputs = []
        track_states = {
            'syntax': None,
            'semantic': None,
            'logic': None
        }
        energy_history = []
        all_info = []
        
        steps = relaxation_steps or self.config.relaxation_iterations
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            track_out, track_states = self.tracks(
                xt,
                h_syntax=track_states['syntax'],
                h_semantic=track_states['semantic'],
                h_logic=track_states['logic'],
                steps=steps,
            )
            
            if self.hololink is not None:
                track_out, hl_info = self.hololink(track_out, write_mode=self.training)
            else:
                hl_info = {}
            
            if return_energy:
                energy = self.tracks.compute_energy(
                    track_states['syntax'],
                    track_states['semantic'],
                    track_states['logic'],
                    xt,
                )
                energy_history.append(energy)
            
            mixed = self.mixer(track_out)
            out = self.norm(xt + mixed)
            outputs.append(out)
            
            if return_info:
                all_info.append({
                    'track_states': {k: v.detach().clone() for k, v in track_states.items()},
                    'hololink_info': hl_info,
                })
        
        output_seq = torch.stack(outputs, dim=1)
        logits = self.output_head(output_seq)
        
        if return_energy:
            return logits, energy_history
        
        if return_info:
            return logits, all_info
        
        return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        energy: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, seq, vocab = logits.shape
        
        ce_loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1), ignore_index=0)
        
        loss = ce_loss
        
        result = {'total': loss, 'ce': ce_loss}
        
        if energy is not None and len(energy) > 0:
            energy_reg = sum(e['total'].mean() for e in energy) / len(energy)
            loss = loss + 0.001 * energy_reg
            result['energy'] = energy_reg
            result['total'] = loss
        
        return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        if self.hololink is None:
            return {'hololink': None}
        return {'hololink': self.hololink.get_memory_stats()}


def create_bio_ana(variant: str = 'nano', **kwargs) -> BioANAModel:
    config = get_bio_config(variant, **kwargs)
    return BioANAModel(config)
