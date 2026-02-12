"""
ANA v4: Forced Memory Usage

Key insight: The model should CANNOT solve the task without using memory correctly.
The SSM alone cannot do key-value recall because it requires:
1. Storing arbitrary associations
2. Retrieving by exact key match

Design:
- The SSM is designed to NOT memorize specific content
- The memory MUST be used for storage
- The controller MUST emit correct modes or the model fails
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass  
class ANAv4Config:
    vocab_size: int = 60
    d_model: int = 64
    state_dim: int = 64
    memory_slots: int = 32  # Number of key-value slots
    max_position: int = 8192


class ContentMemory(nn.Module):
    """
    Memory that stores EXACT key-value pairs.
    Uses differentiable addressing but with sharp (near-hard) attention.
    """
    
    def __init__(self, d_model: int, memory_slots: int):
        super().__init__()
        self.d_model = d_model
        self.memory_slots = memory_slots
        
        # Memory state: keys and values
        # Keys are normalized, values are arbitrary vectors
        self.register_buffer('memory_keys', torch.zeros(1, memory_slots, d_model))
        self.register_buffer('memory_values', torch.zeros(1, memory_slots, d_model))
        self.register_buffer('memory_occupied', torch.zeros(1, memory_slots))
        
        # Projections
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Sharpness for attention (higher = harder attention)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
    def reset_memory(self, batch_size: int, device):
        """Reset memory for new sequence."""
        self.memory_keys = torch.zeros(batch_size, self.memory_slots, self.d_model, device=device)
        self.memory_values = torch.zeros(batch_size, self.memory_slots, self.d_model, device=device)
        self.memory_occupied = torch.zeros(batch_size, self.memory_slots, device=device)
        
    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Write key-value pair to memory.
        Finds least occupied slot or most similar slot.
        
        Args:
            key: [batch, d_model]
            value: [batch, d_model]
        """
        batch_size = key.shape[0]
        device = key.device
        
        # Project keys and values
        k = F.normalize(self.key_proj(key), dim=-1)  # [batch, d]
        v = self.value_proj(value)  # [batch, d]
        
        # Ensure memory is the right batch size
        if self.memory_keys.shape[0] != batch_size:
            self.memory_keys = torch.zeros(batch_size, self.memory_slots, self.d_model, device=device)
            self.memory_values = torch.zeros(batch_size, self.memory_slots, self.d_model, device=device)
            self.memory_occupied = torch.zeros(batch_size, self.memory_slots, device=device)
        
        # Find slot: use least occupied slot (or overwrite most similar)
        occupancy = self.memory_occupied  # [batch, slots]
        
        # Find empty slot (lowest occupancy)
        _, slot_idx = occupancy.min(dim=-1)  # [batch]
        
        # Write to selected slot
        batch_idx = torch.arange(batch_size, device=device)
        self.memory_keys[batch_idx, slot_idx] = k
        self.memory_values[batch_idx, slot_idx] = v
        self.memory_occupied[batch_idx, slot_idx] = 1.0
        
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from memory by key matching.
        
        Args:
            query: [batch, d_model]
        Returns:
            retrieved: [batch, d_model]
        """
        # Project query
        q = F.normalize(self.query_proj(query), dim=-1)  # [batch, d]
        
        # Compute attention over memory slots
        # [batch, slots] = [batch, 1, d] @ [batch, d, slots]
        scores = torch.bmm(q.unsqueeze(1), self.memory_keys.transpose(-1, -2)).squeeze(1)
        
        # Sharpen attention with temperature
        attn = F.softmax(scores * self.temperature, dim=-1)  # [batch, slots]
        
        # Read from memory
        retrieved = torch.bmm(attn.unsqueeze(1), self.memory_values).squeeze(1)  # [batch, d]
        
        return retrieved


class ModeDetector(nn.Module):
    """
    Detects the current operation mode based on input.
    Uses hard token detection + learned context.
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        
        # Special token IDs (learned embeddings)
        self.key_token_id = 1
        self.val_token_id = 2
        self.query_token_id = 3
        
        # Context network: looks at surrounding tokens
        self.context_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # 3 modes: STORE_KEY, STORE_VAL, RETRIEVE
        )
        
    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns mode logits.
        
        Modes:
            0: STORE_KEY - we're seeing a key, prepare to store
            1: STORE_VAL - we're seeing a value, complete the store  
            2: RETRIEVE - we're seeing a query, retrieve from memory
        """
        # Context-based mode prediction
        mode_logits = self.context_net(x)  # [batch, seq, 3]
        
        # Hard token-based supervision hint (for training)
        # This helps the model learn the correct modes faster
        with torch.no_grad():
            # Token after TOK_KEY should be STORE_KEY mode
            # Token after TOK_VAL should be STORE_VAL mode
            # Token after TOK_QUERY should be RETRIEVE mode
            hint = torch.zeros_like(mode_logits)
            
            # Find special tokens
            key_positions = (token_ids == self.key_token_id)
            val_positions = (token_ids == self.val_token_id)
            query_positions = (token_ids == self.query_token_id)
            
            # The token AFTER a special token determines the mode
            # Shift right to get "next token" positions
            hint[key_positions.roll(1, dims=1), 0] = 10.0  # STORE_KEY after TOK_KEY
            hint[val_positions.roll(1, dims=1), 1] = 10.0  # STORE_VAL after TOK_VAL
            hint[query_positions.roll(1, dims=1), 2] = 10.0  # RETRIEVE after TOK_QUERY
            
            # Zero out first position (no previous token)
            hint[:, 0, :] = 0
            
        # Combine learned logits with hints (during training, hints dominate)
        if self.training:
            mode_logits = mode_logits + hint
            
        return mode_logits


class MinimalSSM(nn.Module):
    """
    Minimal SSM that handles position encoding and local patterns.
    Intentionally NOT capable of key-value memorization.
    """
    
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(d_model, state_dim)
        self.output_proj = nn.Linear(state_dim, d_model)
        
        # Simple diagonal SSM
        self.A = nn.Parameter(torch.randn(state_dim))
        self.B = nn.Parameter(torch.randn(state_dim) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        u = self.input_proj(x)
        h = torch.zeros(batch, self.A.shape[0], device=x.device)
        
        outputs = []
        for t in range(seq_len):
            h = self.A * h + self.B * u[:, t, :]
            outputs.append(self.output_proj(h))
            
        return torch.stack(outputs, dim=1)


class ANAv4(nn.Module):
    """
    ANA v4: Forced Memory Architecture
    
    The model MUST use memory to solve the task:
    1. SSM handles local patterns and position
    2. Memory handles arbitrary key-value associations
    3. ModeDetector switches between store/retrieve operations
    """
    
    def __init__(self, config: ANAv4Config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = nn.Embedding(config.max_position, config.d_model)
        
        # Components
        self.mode_detector = ModeDetector(config.d_model, config.vocab_size)
        self.memory = ContentMemory(config.d_model, config.memory_slots)
        self.ssm = MinimalSSM(config.d_model, config.state_dim)
        
        # Output processing for each mode
        self.store_processor = nn.Linear(config.d_model, config.d_model)
        self.retrieve_processor = nn.Linear(config.d_model, config.d_model)
        
        # Final output
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Reset memory for each sequence
        self.memory.reset_memory(batch, device)
        
        # Embed
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.position_encoding(pos_ids)
        
        # Get mode predictions
        mode_logits = self.mode_detector(x, input_ids)  # [batch, seq, 3]
        
        # SSM processes the sequence (for position/context)
        ssm_out = self.ssm(x)
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            token_x = x[:, t, :]  # [batch, d]
            token_mode = mode_logits[:, t, :]  # [batch, 3]
            
            # Soft mode selection
            mode_probs = F.softmax(token_mode, dim=-1)  # [batch, 3]
            
            # STORE_KEY mode: remember the key for next token
            store_key_out = self.store_processor(token_x)
            
            # STORE_VAL mode: store (key_prev, val_current)
            store_val_out = self.store_processor(token_x)
            
            # RETRIEVE mode: query memory
            retrieve_out = self.retrieve_processor(self.memory.read(token_x))
            
            # Combine based on mode
            mode_outs = torch.stack([store_key_out, store_val_out, retrieve_out], dim=-1)  # [batch, d, 3]
            combined = (mode_outs * mode_probs.unsqueeze(1)).sum(dim=-1)  # [batch, d]
            
            # Residual: SSM + mode-specific processing
            out_t = ssm_out[:, t, :] + combined
            outputs.append(out_t)
            
            # Execute memory operations based on HARD mode decision
            mode_idx = token_mode.argmax(dim=-1)  # [batch]
            
            # STORE_KEY: remember key for next position
            # STORE_VAL: write (key, val) to memory
            # RETRIEVE: already handled above
            
            # For simplicity: write at STORE_VAL position
            # (key was seen 2 positions ago)
            store_val_mask = (mode_idx == 1)
            if store_val_mask.any() and t >= 2:
                # Get indices where we should store
                store_indices = torch.where(store_val_mask)[0]
                # Get the key from 1 position ago (after TOK_KEY)
                key_x = x[store_indices, t-1, :]
                val_x = token_x[store_indices]
                # Write each key-value pair
                for i, idx in enumerate(store_indices):
                    self.memory.write(key_x[i:i+1], val_x[i:i+1])
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq, d]
        output = self.norm(output)
        logits = self.output_head(output)
        
        return logits


if __name__ == "__main__":
    config = ANAv4Config()
    model = ANAv4(config)
    
    x = torch.randint(0, 60, (2, 32))
    logits = model(x)
    print(f"Input: {x.shape}, Output: {logits.shape}")
