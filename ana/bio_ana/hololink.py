import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class HoloLinkHebbian(nn.Module):
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        capacity: int = 1000,
        hebbian_lr: float = 0.01,
        decay: float = 0.001,
        normalize: bool = True,
    ):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        self.hebbian_lr = hebbian_lr
        self.decay = decay
        self.normalize = normalize
        
        self.key_proj = nn.Linear(value_dim, key_dim)
        self.query_proj = nn.Linear(value_dim, key_dim)
        self.value_proj = nn.Linear(value_dim, value_dim)
        self.output_proj = nn.Linear(key_dim, value_dim)
        
        memory_init = torch.zeros(capacity, key_dim)
        self.register_buffer('memory', memory_init)
        self.register_buffer('usage', torch.zeros(capacity))
        self.register_buffer('write_ptr', torch.tensor(0))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        for p in [self.key_proj, self.query_proj, self.value_proj, self.output_proj]:
            if p.bias is not None:
                nn.init.zeros_(p.bias)
    
    def hebbian_update(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        
        similarity = torch.matmul(query, self.memory.T)
        weights = F.softmax(similarity, dim=-1)
        
        write_delta = torch.matmul(weights.T, key)
        
        delta = self.hebbian_lr * (
            write_delta / batch_size -
            self.decay * self.memory
        )
        
        new_memory = self.memory + delta
        
        if self.normalize:
            norms = torch.norm(new_memory, dim=-1, keepdim=True)
            new_memory = new_memory / (norms + 1e-8)
        
        self.memory.data.copy_(new_memory)
        
        return delta.norm()
    
    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        batch_size = key.shape[0]
        
        for b in range(batch_size):
            ptr = self.write_ptr.item() % self.capacity
            self.memory[ptr] = key[b].detach()
            self.usage[ptr] = 1.0
            self.write_ptr = (self.write_ptr + 1) % self.capacity
    
    def read(self, query: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        similarity = torch.matmul(query, self.memory.T)
        
        weights = F.softmax(similarity, dim=-1)
        retrieved = torch.matmul(weights, self.memory)
        
        return retrieved, weights
    
    def forward(
        self,
        h: torch.Tensor,
        write_mode: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        key = self.key_proj(h)
        query = self.query_proj(h)
        value = self.value_proj(h)
        
        retrieved, weights = self.read(query)
        
        info = {'weights': weights, 'retrieved': retrieved}
        
        if write_mode and self.training:
            delta_norm = self.hebbian_update(query, key, value)
            info['delta_norm'] = delta_norm
        
        output = self.output_proj(retrieved)
        
        return output, info
    
    def get_memory_stats(self) -> Dict[str, float]:
        used = (self.usage > 0).sum().item()
        avg_usage = self.usage.mean().item()
        memory_norm = torch.norm(self.memory, dim=-1).mean().item()
        
        return {
            'capacity': self.capacity,
            'used': used,
            'utilization': used / self.capacity,
            'avg_usage': avg_usage,
            'avg_norm': memory_norm,
        }


class BioHoloLink(nn.Module):
    def __init__(
        self,
        input_dim: int,
        key_dim: int = 128,
        capacity: int = 1000,
        hebbian_lr: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        
        self.memory = HoloLinkHebbian(
            key_dim=key_dim,
            value_dim=input_dim,
            capacity=capacity,
            hebbian_lr=hebbian_lr,
        )
        
        self.gate = nn.Linear(input_dim + key_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        
        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.zeros_(self.gate.bias)
    
    def forward(
        self,
        h: torch.Tensor,
        write_mode: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mem_output, mem_info = self.memory(h, write_mode=write_mode)
        
        retrieved_key = mem_info['retrieved']
        combined = torch.cat([h, retrieved_key], dim=-1)
        gate_weight = torch.sigmoid(self.gate(combined))
        
        output = self.norm(h + gate_weight * mem_output)
        
        return output, mem_info
    
    def get_memory_stats(self) -> Dict[str, float]:
        return self.memory.get_memory_stats()
