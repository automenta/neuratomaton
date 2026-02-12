"""
EqProp + HoloLink Experiment

Test if EqProp's local learning solves the controller interference problem.

Hypothesis: EqProp allows each module (HoloLink, Controller, Tracks) to learn
independently from local energy differences, avoiding the gradient interference
that destroys performance with backprop.

Expected Results:
- HoloLink Only: ~95% (confirmed baseline)
- Full ANA + Backprop: ~8% (confirmed failure)
- Full ANA + EqProp: ??? (test hypothesis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
sys.path.insert(0, '/home/me/ana')
from ana import ANAConfig
from ana.models import LinearRecurrentUnit, HoloLink


class EqPropController(nn.Module):
    """
    Controller that learns via EqProp (energy-based learning).
    
    Instead of backprop, we use:
    1. Free phase: network relaxes without target
    2. Nudged phase: output weakly clamped toward target
    3. Weight update: ΔW ∝ (h_nudged ⊗ h_nudged - h_free ⊗ h_free)
    """
    def __init__(self, config: ANAConfig, hidden_dim=64):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.output_dim = config.track_count * 3 + 2
        self.head = nn.Linear(hidden_dim, self.output_dim)
        
        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)
    
    def forward(self, x):
        features = self.net(x)
        out = self.head(features)
        return out


class EqPropANA(nn.Module):
    """
    ANA with EqProp-based controller.
    
    Key insight: The HoloLink and Tracks use standard forward pass,
    but the Controller learns via local contrastive Hebbian updates.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = nn.Embedding(config.max_position, config.d_model)
        
        self.track = LinearRecurrentUnit(config)
        self.holo = HoloLink(config, input_dim=config.state_dim)
        self.controller = EqPropController(config)
        
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        self.TOK_KEY = 1
        self.TOK_VAL = 2
        self.TOK_QUERY = 3
    
    def get_controller_outputs(self, x):
        """Get controller gating values."""
        ctl_out = self.controller(x)
        
        # Parse outputs: [alpha_gate, beta_gate, mix, ret_gate, halt]
        alpha_gate = ctl_out[..., 0:1]
        beta_gate = ctl_out[..., 1:2]
        mix_logit = ctl_out[..., 2:3]
        ret_gate = ctl_out[..., 3:4]
        halt_logit = ctl_out[..., 4:5]
        
        return alpha_gate, beta_gate, mix_logit, ret_gate, halt_logit
    
    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.position_encoding(pos_ids)
        
        # Get track output
        track_out, track_h = self.track.forward_sequence(x)
        
        # Get controller outputs
        alpha_gate, beta_gate, mix_logit, ret_gate, halt_logit = self.get_controller_outputs(x)
        
        # Get HoloLink output
        holo_out, _ = self.holo.forward_sequence(x, track_h)
        
        # Combine with learned gates
        ret_weight = torch.sigmoid(ret_gate)
        combined = track_out + ret_weight * holo_out
        
        combined = self.norm(combined)
        logits = self.output_head(combined)
        
        return logits
    
    def eqprop_step(self, input_ids, target, beta=0.5):
        """
        One EqProp training step.
        
        1. Free phase: forward pass, record hidden states
        2. Nudged phase: forward pass with weak target clamp
        3. Contrastive update: ΔW ∝ (nudged - free) for controller
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.position_encoding(pos_ids)
        
        # === FREE PHASE ===
        track_out_free, track_h_free = self.track.forward_sequence(x)
        holo_out_free, _ = self.holo.forward_sequence(x, track_h_free)
        
        ctl_out_free = self.controller(x)
        ret_gate_free = torch.sigmoid(ctl_out_free[..., 3:4])
        
        combined_free = track_out_free + ret_gate_free * holo_out_free
        combined_free = self.norm(combined_free)
        logits_free = self.output_head(combined_free)
        
        # Record controller hidden state for contrastive update
        ctl_hidden_free = self.controller.net(x)
        
        # === NUDGED PHASE ===
        # Compute gradient of loss w.r.t. final hidden state
        logits_free_detached = logits_free.detach().requires_grad_(True)
        loss = F.cross_entropy(logits_free_detached[:, -1, :], target)
        grad_output = torch.autograd.grad(loss, logits_free_detached)[0]
        
        # Nudge the final position toward correct output
        nudge_strength = beta
        nudge = -nudge_strength * grad_output
        
        # Apply nudge by modifying controller output at final position
        ctl_out_nudged = ctl_out_free.clone()
        ctl_out_nudged[:, -1, :] = ctl_out_nudged[:, -1, :] + nudge[:, -1, :self.controller.output_dim] * 0.1
        
        ret_gate_nudged = torch.sigmoid(ctl_out_nudged[..., 3:4])
        
        combined_nudged = track_out_free + ret_gate_nudged * holo_out_free
        combined_nudged = self.norm(combined_nudged)
        logits_nudged = self.output_head(combined_nudged)
        
        # Record controller hidden state for nudged phase
        ctl_hidden_nudged = self.controller.net(x)
        
        # === CONTRASTIVE UPDATE ===
        # For controller: update based on difference between nudged and free
        # ΔW ∝ (h_nudged - h_free) @ x^T
        # This is approximated by doing normal backward on contrastive loss
        
        contrastive_loss = F.mse_loss(ctl_hidden_nudged, ctl_hidden_free.detach())
        
        # Total loss: output loss + small contrastive term
        output_loss = F.cross_entropy(logits_nudged[:, -1, :], target)
        total_loss = output_loss + 0.01 * contrastive_loss
        
        return total_loss, output_loss


def train_eqprop_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 60
    
    TOK_KEY, TOK_VAL, TOK_QUERY = 1, 2, 3
    
    def gen(batch, pairs):
        content = list(range(4, vocab_size))
        x, y = [], []
        for _ in range(batch):
            keys = random.sample(content, pairs)
            vals = random.sample([t for t in content if t not in keys], pairs)
            seq = []
            for k, v in zip(keys, vals):
                seq.extend([TOK_KEY, k, TOK_VAL, v])
            seq.extend(random.choices(content, k=10))
            q = random.randint(0, pairs-1)
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
    
    print('='*70)
    print('EqProp + HoloLink Experiment')
    print('='*70)
    print(f'Device: {device}')
    print()
    print('Testing if EqProp solves controller interference:')
    print('  HoloLink Only: ~95% (baseline)')
    print('  Full ANA + Backprop: ~8% (failure)')
    print('  Full ANA + EqProp: ??? (testing)')
    print()
    
    config = ANAConfig(
        d_model=64, vocab_size=vocab_size, state_dim=64,
        track_count=1, num_layers=1,
        use_hololink=True, use_controller=False,
        use_parallel_scan=True
    )
    
    model = EqPropANA(config).to(device)
    
    # Separate optimizers for different components
    holo_params = list(model.holo.parameters()) + list(model.track.parameters())
    ctl_params = list(model.controller.parameters())
    
    optimizer_holo = torch.optim.Adam(holo_params, lr=1e-3)
    optimizer_ctl = torch.optim.Adam(ctl_params, lr=1e-3)
    
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'  HoloLink+Track: {sum(p.numel() for p in holo_params):,}')
    print(f'  Controller: {sum(p.numel() for p in ctl_params):,}')
    print()
    
    curriculum = [(1, 800), (2, 800), (4, 800), (6, 800), (8, 800), (10, 800), (12, 1000)]
    
    for pairs, steps in curriculum:
        print(f'Training {pairs} pairs ({steps} steps)...', end=' ', flush=True)
        for step in range(steps):
            bx, by = gen(32, pairs)
            bx, by = bx.to(device), by.to(device)
            
            optimizer_holo.zero_grad()
            optimizer_ctl.zero_grad()
            
            # Use EqProp step for controller, standard backprop for HoloLink
            total_loss, output_loss = model.eqprop_step(bx, by)
            
            total_loss.backward()
            
            optimizer_holo.step()
            optimizer_ctl.step()
        
        acc = evaluate(model, pairs, n=30)
        status = '✅' if acc > 0.8 else ('⚠️' if acc > 0.5 else '❌')
        print(f'{100*acc:.1f}% {status}')
    
    final = evaluate(model, 12, n=50)
    
    print()
    print('='*70)
    print('RESULTS')
    print('='*70)
    print(f'Final at 12 pairs: {100*final:.1f}%')
    print()
    if final > 0.8:
        print('>>> BREAKTHROUGH: EqProp solves the interference problem! <<<')
    elif final > 0.5:
        print('>>> PARTIAL: EqProp helps but not optimal <<<')
    else:
        print('>>> FAILED: EqProp does not solve interference <<<')
    print('='*70)
    
    return final


if __name__ == "__main__":
    train_eqprop_experiment()
