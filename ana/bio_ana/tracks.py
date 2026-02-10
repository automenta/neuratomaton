import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from typing import Optional, Dict, Tuple, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eqprop"))

from bioplausible.models.eqprop_base import EqPropModel


class BioTrackEnergy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        track_type: str = 'semantic',
        tau: float = 1.0,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.track_type = track_type
        self.tau = tau
        
        tau_defaults = {'syntax': 0.5, 'semantic': 2.0, 'logic': 1.0}
        self.tau = tau_defaults.get(track_type, tau)
        
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.W_in, self.W_rec]:
            actual = layer
            if hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, 'weight'):
                actual = layer.parametrizations.weight.original
            if hasattr(actual, 'weight'):
                nn.init.xavier_uniform_(actual.weight, gain=0.5)
                if actual.bias is not None:
                    nn.init.zeros_(actual.bias)
    
    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.track_type == 'syntax':
            t = torch.tanh(x)
            s = torch.sigmoid(2 * x)
            return t * s
        elif self.track_type == 'logic':
            return torch.tanh(x) ** 3
        else:
            return torch.tanh(x)
    
    def energy(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        quadratic = (h ** 2).sum(dim=-1) / (2 * self.tau)
        net = self.W_in(x) + self.W_rec(h)
        f_net = self._activation(net)
        interaction = -torch.sum(h * f_net, dim=-1)
        return quadratic + interaction
    
    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        net = self.W_in(x) + self.W_rec(h)
        return self._activation(net)
    
    def forward(self, x: torch.Tensor, h_init: Optional[torch.Tensor] = None, 
                steps: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        if h_init is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h = h_init
        
        for _ in range(steps):
            h = self.forward_step(h, x)
        
        return h, h


class BioSyntaxTrack(BioTrackEnergy):
    def __init__(self, input_dim: int, hidden_dim: int, use_spectral_norm: bool = True):
        super().__init__(input_dim, hidden_dim, track_type='syntax', tau=0.5, 
                        use_spectral_norm=use_spectral_norm)


class BioSemanticTrack(BioTrackEnergy):
    def __init__(self, input_dim: int, hidden_dim: int, use_spectral_norm: bool = True):
        super().__init__(input_dim, hidden_dim, track_type='semantic', tau=2.0,
                        use_spectral_norm=use_spectral_norm)


class BioLogicTrack(BioTrackEnergy):
    def __init__(self, input_dim: int, hidden_dim: int, use_spectral_norm: bool = True):
        super().__init__(input_dim, hidden_dim, track_type='logic', tau=1.0,
                        use_spectral_norm=use_spectral_norm)


class BioSpecializedTracks(nn.Module):
    def __init__(
        self,
        d_model: int,
        syntax_dim: int,
        semantic_dim: int,
        logic_dim: int,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.syntax_dim = syntax_dim
        self.semantic_dim = semantic_dim
        self.logic_dim = logic_dim
        
        self.syntax_track = BioSyntaxTrack(d_model, syntax_dim, use_spectral_norm)
        self.semantic_track = BioSemanticTrack(d_model, semantic_dim, use_spectral_norm)
        self.logic_track = BioLogicTrack(d_model, logic_dim, use_spectral_norm)
        
        self.output_dim = syntax_dim + semantic_dim + logic_dim
    
    def forward(
        self,
        x: torch.Tensor,
        h_syntax: Optional[torch.Tensor] = None,
        h_semantic: Optional[torch.Tensor] = None,
        h_logic: Optional[torch.Tensor] = None,
        steps: int = 20,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_syntax, _ = self.syntax_track(x, h_syntax, steps)
        h_semantic, _ = self.semantic_track(x, h_semantic, steps)
        h_logic, _ = self.logic_track(x, h_logic, steps)
        
        output = torch.cat([h_syntax, h_semantic, h_logic], dim=-1)
        
        states = {
            'syntax': h_syntax,
            'semantic': h_semantic,
            'logic': h_logic
        }
        
        return output, states
    
    def compute_energy(
        self,
        h_syntax: torch.Tensor,
        h_semantic: torch.Tensor,
        h_logic: torch.Tensor,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            'syntax': self.syntax_track.energy(h_syntax, x),
            'semantic': self.semantic_track.energy(h_semantic, x),
            'logic': self.logic_track.energy(h_logic, x),
            'total': (
                self.syntax_track.energy(h_syntax, x) +
                self.semantic_track.energy(h_semantic, x) +
                self.logic_track.energy(h_logic, x)
            )
        }
