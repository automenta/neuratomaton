#!/usr/bin/env python
"""
Example: Reinforcement Learning Agent with ANA

This script demonstrates how to initialize the ANA RL Agent and run a forward pass
simulating an environment interaction step.
"""

import torch
from ana import ANAConfig, ANARLAgent

def main():
    print("=== ANA RL Agent Example ===")

    # 1. Create Configuration
    # Using small parameters for demonstration
    config = ANAConfig(
        action_space=4,         # Number of discrete actions
        observation_space=10,   # Size of observation vector
        d_model=64,
        state_dim=32,
        num_layers=2,
        use_hololink=True,
        track_count=2
    )
    print(f"Configuration: {config}")

    # 2. Initialize Agent
    agent = ANARLAgent(config)
    print(f"Agent created with {sum(p.numel() for p in agent.parameters())} parameters.")

    # 3. Create Dummy Observation
    # Batch size 1 (single environment step), Obs size 10
    batch_size = 1
    obs = torch.randn(batch_size, config.observation_space)
    print(f"Observation shape: {obs.shape}")

    # 4. Forward Pass (Step)
    # The agent also maintains internal state (hidden states, memory)
    # Initial state is None
    state = None

    # Run a few steps
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        policy_logits, value, next_state, info = agent(obs, state)

        print(f"Policy Logits shape: {policy_logits.shape}") # Should be [1, action_space]
        print(f"Value shape: {value.shape}")                 # Should be [1, 1]

        # Update state for next step
        state = next_state
        # In a real loop, obs would be updated from the environment
        obs = torch.randn(batch_size, config.observation_space)

    print("\nRL Agent simulation successful!")

if __name__ == "__main__":
    main()
