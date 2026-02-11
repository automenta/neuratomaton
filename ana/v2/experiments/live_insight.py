#!/usr/bin/env python3
"""
ANA v2: LIVE INSIGHT - Watch the model THINK!

Shows interesting intermediate results as the model processes data.
"""

import sys
sys.path.insert(0, '/home/me/ana')

import torch
import torch.nn.functional as F
import numpy as np

print("="*70)
print("ANA v2: LIVE INSIGHT - Watch the Model THINK!")
print("="*70)

# Import
from ana.v2.core import ANAConfig, ANAModel, GumbelSoftmax

# Tiny data - one example
input_seq = torch.tensor([[1, 2, 3, 4, 5]])
target_seq = torch.tensor([[5, 4, 3, 2, 1]])
vocab_size = 6

print(f"\n📝 TASK: Reverse sequence")
print(f"   Input:  {input_seq[0].tolist()}")
print(f"   Target: {target_seq[0].tolist()}")

# Create model
config = ANAConfig(
    d_model=32, vocab_size=vocab_size,
    track_dims=(8, 16, 8), stack_depth=3,
    stack_dim=16, num_layers=1
)
model = ANAModel(config)
print(f"\n🧠 MODEL: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"   Tracks: {config.track_dims} (fast, slow, logic)")
print(f"   Stack:  depth={config.stack_depth}, dim={config.stack_dim}")
print(f"   Opcodes: PUSH, POP, BIND, CALL")

# Show architecture
print(f"\n🏗️  ARCHITECTURE:")
print(f"   Input → Embedding → [Layer × 1] → Output")
print(f"                        │")
print(f"                        ├── Interpreter (executes opcodes)")
print(f"                        ├── 3 Tracks (SSM with dynamic α,β)")
print(f"                        └── Holographic Memory (FFT binding)")

# Run forward pass with info tracking
print(f"\n🔬 RUNNING MODEL STEP-BY-STEP...")
print(f"{'Step':<6} {'Token':<6} {'Opcode':<12} {'Stack':<6} {'α_mods':<20} {'β_mods':<20}")
print(f"{'-'*70}")

model.eval()
with torch.no_grad():
    x = model.embedding(input_seq)
    
    for layer in model.layers:
        layer.reset_state()
        
        track_states = [None] * config.num_tracks
        all_info = []
        
        for t in range(5):
            xt = x[:, t, :]
            token = input_seq[0, t].item()
            
            opcode_logits = layer.opcode_head(xt)
            h_stack = torch.zeros(1, config.stack_dim)
            
            # Execute interpreter
            alpha_mods, beta_mods, h_stack, info = layer.interpreter.execute(
                xt, opcode_logits, layer.stack, layer.hologram, h_stack
            )
            
            # Process tracks
            for i, track in enumerate(layer.tracks):
                _, track_states[i] = track._step(xt, track_states[i], 
                                                   alpha_mods[:, i:i+1], 
                                                   beta_mods[:, i:i+1])
            
            # Decode opcode
            op_code = info['opcode'][0].argmax().item()
            opcode_name = ['PUSH', 'POP', 'BIND', 'CALL'][op_code]
            
            # Format mods
            alpha_str = '[' + ', '.join([f'{a:.2f}' for a in alpha_mods[0]]) + ']'
            beta_str = '[' + ', '.join([f'{b:.2f}' for b in beta_mods[0]]) + ']'
            
            print(f"{t:<6} {token:<6} {opcode_name:<12} {info['stack_depth']:<6} {alpha_str:<20} {beta_str:<20}")

# Show what this means
print(f"\n📊 INSIGHTS:")
print(f"   Opcode distribution:")
print(f"     PUSH  : Stores info on stack, slows down slow track")
print(f"     POP   : Retrieves from stack, speeds up fast track")  
print(f"     BIND  : Writes to holographic memory")
print(f"     CALL  : Enters subroutine, engages logic track")
print(f"\n   Track modulation (α,β):")
print(f"     α values control memory retention (higher = more memory)")
print(f"     β values control input influence (higher = more new info)")
print(f"     Different opcodes trigger different modulation patterns!")

# Show holographic memory
print(f"\n💾 HOLOGRAPHIC MEMORY:")
print(f"   Uses FFT circular convolution for VSA binding")
print(f"   bind:   M += FFT(key) ⊙ FFT(value)")
print(f"   unbind: v ≈ IFFT(conj(FFT(query)) ⊙ FFT(M))")
print(f"   Benefits: Superposition storage, O(1) retrieval")

# Quick training demo
print(f"\n🎓 QUICK TRAINING DEMO (20 steps):")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []

for step in range(20):
    optimizer.zero_grad()
    logits = model(input_seq)
    loss = F.cross_entropy(logits.view(-1, vocab_size), target_seq.view(-1), ignore_index=0)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if (step + 1) % 5 == 0:
        print(f"   Step {step+1:2}: Loss = {loss.item():.4f}")

print(f"\n📈 LEARNING CURVE:")
print(f"   Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
print(f"   Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

# Final prediction
model.eval()
with torch.no_grad():
    final_logits = model(input_seq)
    final_preds = final_logits.argmax(dim=-1)
    
    print(f"\n🎯 FINAL PREDICTION:")
    print(f"   Input:    {input_seq[0].tolist()}")
    print(f"   Predicted: {final_preds[0].tolist()}")
    print(f"   Target:   {target_seq[0].tolist()}")
    
    # Check what got right
    correct = (final_preds[0] == target_seq[0]).sum().item()
    print(f"   Correct: {correct}/5 = {correct/5:.1%}")

# Summary
print(f"\n" + "="*70)
print(f"🚀 KEY TAKEAWAYS:")
print(f"   1. Model EXECUTES opcodes, doesn't just sample them")
print(f"   2. Opcodes modulate track dynamics (α, β values)")
print(f"   3. Stack enables multi-step reasoning")
print(f"   4. Holographic memory provides O(1) associative recall")
print(f"   5. Architecture is DIFFERENT from Transformer/Mamba")
print(f"\n   The BEAST is ready. Time to train and test generalization!")
print(f"="*70)
