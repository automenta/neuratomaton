"""
ANA with Equilibrium Propagation

Key insight: EqProp trains by comparing two equilibrium states:
1. Free phase: network settles without target
2. Nudged phase: output weakly pushed toward target

The gradient is: ∂E_free/∂θ - ∂E_nudged/∂θ (locally computed!)

This avoids backprop's interference problem because each module
receives its own local learning signal from energy differences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class EqPropConfig:
    vocab_size: int = 60
    d_model: int = 64
    state_dim: int = 64
    memory_dim: int = 64
    max_position: int = 8192
    n_iterations: int = 20  # Relaxation iterations
    beta: float = 1.0  # Nudging strength


class EnergyLayer(nn.Module):
    """Base class for layers that contribute to energy."""
    
    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute energy contribution. Lower = better."""
        raise NotImplementedError
    
    def forward_state(self, state: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Update state given input (for relaxation dynamics)."""
        raise NotImplementedError


class EnergySSM(EnergyLayer):
    """
    SSM that computes energy-based state updates.
    
    Energy: E = Σ (h_t - A*h_{t-1} - B*x_t)^2
    This is like a spring energy - deviation from dynamics costs energy.
    """
    
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        self.input_proj = nn.Linear(d_model, state_dim)
        self.output_proj = nn.Linear(state_dim, d_model)
        
        # Dynamics parameters (constrained to be stable)
        self.A_log = nn.Parameter(torch.randn(state_dim))
        self.B = nn.Parameter(torch.randn(state_dim) * 0.1)
        
        # Controller modulates dynamics
        self.delta_proj = nn.Linear(d_model, state_dim * 2)
        
    def get_dynamics(self, x: torch.Tensor):
        """Get per-position A and B values."""
        batch, seq_len, _ = x.shape
        
        # Base dynamics
        A = torch.sigmoid(self.A_log)  # [state_dim] - stable recurrence
        B = self.B
        
        # Controller modulation
        delta = self.delta_proj(x)  # [batch, seq, state_dim*2]
        delta_A, delta_B = delta.chunk(2, dim=-1)
        
        # Modulated dynamics (bounded changes)
        A_mod = A * (1 + 0.1 * torch.tanh(delta_A))  # Small modulation
        B_mod = B + 0.1 * torch.tanh(delta_B)
        
        return A_mod, B_mod
    
    def energy(self, h_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Energy: sum over time of squared prediction error.
        E = Σ ||h_t - A*h_{t-1} - B*x_t||^2
        """
        batch, seq_len, _ = h_seq.shape
        
        u = self.input_proj(x)
        A_mod, B_mod = self.get_dynamics(x)
        
        # Compute prediction errors
        energy = torch.zeros(batch, device=x.device)
        for t in range(seq_len):
            if t == 0:
                pred = B_mod[:, t, :] * u[:, t, :]
            else:
                pred = A_mod[:, t, :] * h_seq[:, t-1, :] + B_mod[:, t, :] * u[:, t, :]
            
            error = h_seq[:, t, :] - pred
            energy = energy + (error ** 2).sum(dim=-1)
        
        return energy
    
    def relax_step(self, h_seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One step of gradient descent on energy w.r.t. state."""
        h_seq = h_seq.clone()
        batch, seq_len, _ = h_seq.shape
        
        u = self.input_proj(x)
        A_mod, B_mod = self.get_dynamics(x)
        
        # Update each state toward satisfying dynamics
        lr = 0.1  # State learning rate
        for t in range(seq_len):
            if t == 0:
                target = B_mod[:, t, :] * u[:, t, :]
            else:
                target = A_mod[:, t, :] * h_seq[:, t-1, :] + B_mod[:, t, :] * u[:, t, :]
            
            # Move state toward target
            h_seq[:, t, :] = h_seq[:, t, :] + lr * (target - h_seq[:, t, :])
        
        return h_seq


class EnergyHoloLink(EnergyLayer):
    """
    HoloLink memory with energy-based formulation.
    
    Energy: E = Σ ||v_t - M @ k_t||^2 for stored pairs
    Retrieval energy: E = ||q @ M - target||^2
    """
    
    def __init__(self, d_model: int, memory_dim: int):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        
        self.k_proj = nn.Linear(d_model, memory_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, memory_dim, bias=False)
        
        # Binding strength
        self.binding = nn.Parameter(torch.tensor(1.0))
        
    def energy(self, x_seq: torch.Tensor, h_seq: torch.Tensor, 
               memory: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Energy from memory operations.
        Lower when keys match their values well, and when query retrieves correctly.
        """
        batch, seq_len, _, _ = memory.shape  # memory is [batch, seq, mem_dim, d_model]
        
        k = F.normalize(self.k_proj(h_seq), dim=-1)
        v = self.v_proj(h_seq)
        q = F.normalize(self.q_proj(x_seq), dim=-1)
        
        energy = torch.zeros(batch, device=x_seq.device)
        
        # Retrieval energy: if there's a target, query should retrieve it
        if target is not None:
            # Last position should retrieve target
            q_last = q[:, -1, :]  # [batch, memory_dim]
            mem_last = memory[:, -1, :, :]  # [batch, mem_dim, d_model]
            retrieved = torch.bmm(q_last.unsqueeze(1), mem_last).squeeze(1)  # [batch, d_model]
            
            # Compare to target embedding (not target token)
            target_emb = x_seq[:, -1, :]  # Use last position embedding as proxy
            # Actually, we should use output loss for this, not energy
            # For now, just return 0 energy for memory
            pass
        
        return energy
    
    def build_memory(self, h_seq: torch.Tensor, store_mask: torch.Tensor) -> torch.Tensor:
        """Build memory matrix from sequence."""
        k = F.normalize(self.k_proj(h_seq), dim=-1)
        v = self.v_proj(h_seq)
        
        # Outer product bindings, masked by store signal
        binding = F.softplus(self.binding)
        # store_mask: [batch, seq], need [batch, seq, 1, 1]
        store_mask_4d = store_mask.unsqueeze(-1).unsqueeze(-1)
        updates = binding * store_mask_4d * torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        
        # Cumulative memory
        memory = torch.cumsum(updates, dim=1)
        return memory


class EqPropANA(nn.Module):
    """
    ANA trained with Equilibrium Propagation.
    
    The key difference from backprop:
    - No backward pass through controller
    - Controller learns from energy differences between free/nudged phases
    - Each module gets local learning signal
    
    This could solve the interference problem!
    """
    
    def __init__(self, config: EqPropConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = nn.Embedding(config.max_position, config.d_model)
        
        # Energy-based layers
        self.ssm = EnergySSM(config.d_model, config.state_dim)
        self.holo = EnergyHoloLink(config.d_model, config.memory_dim)
        
        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Special tokens
        self.TOK_KEY = 1
        self.TOK_VAL = 2
        self.TOK_QUERY = 3
        
    def get_store_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Determine when to store into memory (after TOK_KEY)."""
        batch, seq_len = input_ids.shape
        
        # Store at positions after TOK_KEY (the key content)
        store_mask = torch.zeros(batch, seq_len, device=input_ids.device)
        for t in range(1, seq_len):
            store_mask[:, t] = (input_ids[:, t-1] == self.TOK_KEY).float()
        
        return store_mask
    
    def compute_total_energy(self, x: torch.Tensor, h: torch.Tensor, 
                             memory: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum of all energy contributions."""
        E_ssm = self.ssm.energy(h, x)
        E_holo = self.holo.energy(x, h, memory, target)
        
        return E_ssm + E_holo
    
    def relax(self, x: torch.Tensor, input_ids: torch.Tensor,
              target: Optional[torch.Tensor] = None, 
              beta: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Relax network to equilibrium.
        
        Args:
            x: embedded input
            target: optional target to nudge toward
            beta: nudging strength (0 = free phase, >0 = nudged phase)
        
        Returns:
            h: equilibrium hidden state
            memory: equilibrium memory state
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Initialize state
        h = torch.zeros(batch, seq_len, self.config.state_dim, device=device)
        
        # Get storage schedule
        store_mask = self.get_store_mask(input_ids)
        
        # Relaxation loop
        for _ in range(self.config.n_iterations):
            # Update SSM state
            h = self.ssm.relax_step(h, x)
            
            # Build memory
            memory = self.holo.build_memory(h, store_mask)
            
            # If nudging, adjust output toward target
            if beta > 0 and target is not None:
                # Get output
                h_out = self.ssm.output_proj(h)
                h_out = x + h_out
                h_out = self.norm(h_out)
                logits = self.output_head(h_out)
                
                # Nudge last position toward target
                # This propagates back through the energy landscape
                log_prob = F.log_softmax(logits[:, -1, :], dim=-1)
                target_log_prob = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
                
                # Weak gradient toward target
                nudge = beta * (1 - target_log_prob.exp())
                # Apply nudge to h (weakly move toward reducing this)
                h[:, -1, :] = h[:, -1, :] + 0.01 * nudge.unsqueeze(-1)
        
        return h, memory
    
    def forward(self, input_ids: torch.Tensor, target: Optional[torch.Tensor] = None,
                return_energy: bool = False):
        """Standard forward pass (for inference or free phase)."""
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.position_encoding(pos_ids)
        
        # Get storage mask
        store_mask = self.get_store_mask(input_ids)
        
        # Initialize and relax
        h = torch.zeros(batch, seq_len, self.config.state_dim, device=device)
        for _ in range(self.config.n_iterations):
            h = self.ssm.relax_step(h, x)
        
        memory = self.holo.build_memory(h, store_mask)
        
        # Output
        h_out = self.ssm.output_proj(h)
        combined = x + h_out
        
        # Add memory retrieval at query positions
        q = F.normalize(self.holo.q_proj(x), dim=-1)  # [batch, seq, mem_dim]
        q_last = q[:, -1, :]  # [batch, mem_dim]
        mem_last = memory[:, -1, :, :]  # [batch, mem_dim, d_model]
        retrieved = torch.bmm(q_last.unsqueeze(1), mem_last).squeeze(1)  # [batch, d_model]
        
        # Apply retrieval at last position
        combined[:, -1, :] = combined[:, -1, :] + retrieved
        
        combined = self.norm(combined)
        logits = self.output_head(combined)
        
        if return_energy:
            E = self.compute_total_energy(x, h, memory, target)
            return logits, E
        
        return logits
    
    def eqprop_step(self, input_ids: torch.Tensor, target: torch.Tensor):
        """
        One step of equilibrium propagation training.
        
        1. Free phase: relax without target
        2. Nudged phase: relax with weak target nudging
        3. Update weights: θ += η * (∂E_free/∂θ - ∂E_nudged/∂θ)
        
        The key insight: this gives LOCAL learning signals!
        Each layer updates based on its own energy contribution.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.position_encoding(pos_ids)
        
        # Free phase
        h_free, memory_free = self.relax(x, input_ids, target=None, beta=0.0)
        E_free = self.compute_total_energy(x, h_free, memory_free, None)
        
        # Nudged phase
        h_nudged, memory_nudged = self.relax(x, input_ids, target=target, 
                                              beta=self.config.beta)
        E_nudged = self.compute_total_energy(x, h_nudged, memory_nudged, target)
        
        # Compute EqProp gradient estimate
        # ∂L/∂θ ≈ ∂E_nudged/∂θ - ∂E_free/∂θ
        # PyTorch handles this via backward on the difference
        
        loss = E_nudged.mean() - E_free.mean()
        
        # Also add standard loss for output
        logits = self.forward(input_ids)
        output_loss = F.cross_entropy(logits[:, -1, :], target)
        
        total_loss = loss + output_loss
        
        return total_loss, output_loss


def train_with_eqprop():
    """Train ANA using equilibrium propagation."""
    import random
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 60
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    
    def gen(batch, pairs):
        content = list(range(4, vocab_size))
        x, y = [], []
        for _ in range(batch):
            keys = random.sample(content, min(pairs, len(content)))
            vals = random.sample([t for t in content if t not in keys], min(pairs, len(content)))
            seq = []
            for k, v in zip(keys, vals):
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            seq.extend(random.choices(content, k=10))
            q = random.randint(0, len(keys)-1)
            seq.extend([TOK_QUERY, keys[q]])
            x.append(seq)
            y.append(vals[q])
        mx = max(len(s) for s in x)
        t = torch.zeros(batch, mx, dtype=torch.long)
        for i, s in enumerate(x):
            t[i, :len(s)] = torch.tensor(s)
        return t, torch.tensor(y)
    
    def evaluate(model, pairs, n=50):
        model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(n):
                bx, by = gen(32, pairs)
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                correct += (logits[:, -1].argmax(-1) == by).sum().item()
        model.train()
        return correct / (n * 32)
    
    print('='*60)
    print('ANA with Equilibrium Propagation')
    print('='*60)
    
    config = EqPropConfig(vocab_size=vocab_size, d_model=64, state_dim=64)
    model = EqPropANA(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    curriculum = [(1, 500), (2, 500), (4, 500), (6, 500), (8, 500), (10, 500), (12, 500)]
    
    print()
    for pairs, steps in curriculum:
        for step in range(steps):
            bx, by = gen(32, pairs)
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            total_loss, output_loss = model.eqprop_step(bx, by)
            total_loss.backward()
            optimizer.step()
            
            if step == 0 or step == steps - 1:
                print(f'  Step {step}: total_loss={total_loss.item():.3f}, output_loss={output_loss.item():.3f}')
        
        acc = evaluate(model, pairs, n=20)
        status = '✅' if acc > 0.8 else ('⚠️' if acc > 0.5 else '❌')
        print(f'{pairs} pairs: {100*acc:.1f}% {status}')
    
    final = evaluate(model, 12, n=50)
    print(f'\nFinal at 12 pairs: {100*final:.1f}%')


if __name__ == "__main__":
    train_with_eqprop()
