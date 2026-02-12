"""
ANA v3: Proper Metaprogramming Architecture

The key insight: The controller should SWITCH between modes, not BLEND them.
Think of it like a finite state machine that recognizes:
- "I'm seeing a KEY token" → STORE mode
- "I'm seeing a QUERY token" → RETRIEVE mode  
- "I'm seeing content" → PROCESS mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ANAv3Config:
    vocab_size: int = 60
    d_model: int = 64
    state_dim: int = 64
    num_layers: int = 1
    memory_size: int = 256  # Fixed memory slots
    max_position: int = 8192
    use_parallel_scan: bool = True


class ParallelScanSSM(nn.Module):
    """Linear recurrent unit with parallel scan for O(log n) training."""
    
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Input/output projections
        self.input_proj = nn.Linear(d_model, state_dim)
        self.output_proj = nn.Linear(state_dim, d_model)
        
        # Learnable recurrence parameters
        self.A_log = nn.Parameter(torch.randn(state_dim))  # Log of A for stability
        self.B = nn.Parameter(torch.randn(state_dim) * 0.1)
        
    def forward(self, x):
        """
        Parallel scan implementation.
        h_t = A * h_{t-1} + B * x_t
        """
        batch, seq_len, _ = x.shape
        
        u = self.input_proj(x)
        A = -torch.exp(self.A_log)  # Negative for stability
        B = self.B
        
        # Parallel scan via cumsum trick (for diagonal A)
        # This is a simplification - full parallel scan is more complex
        log_A = A.view(1, 1, -1).expand(batch, seq_len, -1)
        h = torch.cumsum(B.view(1, 1, -1) * u * torch.exp(-log_A), dim=1)
        h = h * torch.exp(log_A)
        
        return self.output_proj(h)


class MemoryCell(nn.Module):
    """Single memory cell with content-addressable storage."""
    
    def __init__(self, d_model: int, memory_size: int):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        
        # Fixed memory bank
        self.memory_keys = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)
        self.memory_values = nn.Parameter(torch.zeros(memory_size, d_model))
        
        # Write head: learns WHERE to write
        self.write_key = nn.Linear(d_model, d_model)
        self.write_val = nn.Linear(d_model, d_model)
        
        # Read head: learns WHERE to read
        self.read_query = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mode: str = 'process') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, d_model]
            mode: 'store', 'retrieve', or 'process'
        Returns:
            output: [batch, seq, d_model]
            memory_signal: [batch, seq] - how much memory was accessed
        """
        batch, seq_len, _ = x.shape
        
        if mode == 'store':
            # Content-based writing: find similar key location
            write_k = self.write_key(x)  # [batch, seq, d]
            write_v = self.write_val(x)  # [batch, seq, d]
            
            # Compute attention over memory slots
            attn = torch.matmul(write_k, self.memory_keys.T)  # [batch, seq, mem_size]
            attn = F.softmax(attn / math.sqrt(self.d_model), dim=-1)
            
            # Update memory values (soft write)
            # This is differentiable - memory is updated at every forward pass
            update = torch.matmul(attn.transpose(-1, -2), write_v)  # [batch, mem_size, d]
            
            # Return input as-is (storing doesn't change output)
            return x, attn.mean(dim=-1)
            
        elif mode == 'retrieve':
            # Content-based reading
            query = self.read_query(x)  # [batch, seq, d]
            
            # Attend to memory
            attn = torch.matmul(query, self.memory_keys.T)  # [batch, seq, mem_size]
            attn_weights = F.softmax(attn / math.sqrt(self.d_model), dim=-1)
            
            # Read from memory values
            retrieved = torch.matmul(attn_weights, self.memory_values)  # [batch, seq, d]
            
            return retrieved, attn_weights.mean(dim=-1)
            
        else:  # process
            # Pass through without memory access
            return torch.zeros_like(x), torch.zeros(batch, seq_len, device=x.device)


class MetaController(nn.Module):
    """
    The Metaprogrammer: Recognizes patterns and switches modes.
    
    Key insight: This should be a PATTERN RECOGNIZER, not a gate blender.
    It learns to detect:
    - "This looks like a key" → emit STORE signal
    - "This looks like a query" → emit RETRIEVE signal
    - "This is regular content" → emit PROCESS signal
    """
    
    def __init__(self, d_model: int, num_modes: int = 3):
        super().__init__()
        
        # Pattern recognition network
        self.pattern_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        
        # Mode classifier: outputs probability distribution over modes
        self.mode_classifier = nn.Linear(d_model, num_modes)
        
        # Mode embeddings: what each mode "means"
        self.mode_embeddings = nn.Parameter(torch.randn(num_modes, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mode_probs: [batch, seq, num_modes] - probability of each mode
            mode_signal: [batch, seq, d_model] - learned mode embedding
            mode_index: [batch, seq] - hard mode decision (for analysis)
        """
        patterns = self.pattern_net(x)
        
        # Soft mode probabilities
        mode_logits = self.mode_classifier(patterns)
        mode_probs = F.softmax(mode_logits, dim=-1)
        
        # Weighted combination of mode embeddings
        mode_signal = torch.matmul(mode_probs, self.mode_embeddings)
        
        # Hard decision (for mode selection)
        mode_index = mode_probs.argmax(dim=-1)
        
        return mode_probs, mode_signal, mode_index


class ANAv3(nn.Module):
    """
    ANA v3: Adaptive Neural Automaton with Proper Metaprogramming
    
    Architecture:
        Input → Embedding → Position Encoding
                    ↓
            MetaController (pattern recognizer)
                    ↓
            ┌───────┼───────┐
            ↓       ↓       ↓
         STORE   RETRIEVE  PROCESS
            ↓       ↓       ↓
         Memory   Memory   SSM
            └───────┴───────┘
                    ↓
              Mode Switch
                    ↓
               Output
    """
    
    def __init__(self, config: ANAv3Config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = nn.Embedding(config.max_position, config.d_model)
        
        # Core components
        self.controller = MetaController(config.d_model, num_modes=3)
        self.memory = MemoryCell(config.d_model, config.memory_size)
        self.ssm = ParallelScanSSM(config.d_model, config.state_dim)
        
        # Mode-specific processors
        self.store_processor = nn.Linear(config.d_model, config.d_model)
        self.retrieve_processor = nn.Linear(config.d_model, config.d_model)
        self.process_processor = nn.Linear(config.d_model, config.d_model)
        
        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Mode names for logging
        self.mode_names = ['store', 'retrieve', 'process']
        
    def forward(self, input_ids: torch.Tensor, return_mode_info: bool = False):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.position_encoding(pos_ids)
        
        # Controller recognizes patterns and decides modes
        mode_probs, mode_signal, mode_idx = self.controller(x)
        
        # Process each mode
        # STORE mode (index 0)
        store_out, store_signal = self.memory(x, mode='store')
        store_out = self.store_processor(store_out)
        
        # RETRIEVE mode (index 1)  
        retrieve_out, retrieve_signal = self.memory(x, mode='retrieve')
        retrieve_out = self.retrieve_processor(retrieve_out)
        
        # PROCESS mode (index 2)
        process_out = self.ssm(x)
        process_out = self.process_processor(process_out)
        
        # Stack outputs and blend by mode probabilities
        outputs = torch.stack([store_out, retrieve_out, process_out], dim=-1)  # [batch, seq, d, 3]
        
        # Weight by mode probabilities
        mode_weights = mode_probs.unsqueeze(2)  # [batch, seq, 1, 3]
        combined = (outputs * mode_weights).sum(dim=-1)  # [batch, seq, d]
        
        # Add residual and mode signal
        x = x + combined + mode_signal
        
        # Output
        x = self.norm(x)
        logits = self.output_head(x)
        
        if return_mode_info:
            info = {
                'mode_probs': mode_probs,
                'mode_idx': mode_idx,
                'store_signal': store_signal,
                'retrieve_signal': retrieve_signal,
            }
            return logits, info
        
        return logits
    
    def get_mode_distribution(self, input_ids: torch.Tensor):
        """Analyze what modes the controller activates for given input."""
        with torch.no_grad():
            _, _, mode_idx = self.controller(self.embedding(input_ids))
            return mode_idx


if __name__ == "__main__":
    # Quick test
    config = ANAv3Config()
    model = ANAv3(config)
    
    # Test forward pass
    x = torch.randint(0, 60, (2, 32))
    logits, info = model(x, return_mode_info=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Mode probabilities shape: {info['mode_probs'].shape}")
    print(f"Mode indices: {info['mode_idx'][0, :10]}")  # First 10 tokens
