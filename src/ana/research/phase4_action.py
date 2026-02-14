
from ana.research.core import ExperimentBase, ExperimentRegistry
from ana.models.config import ANAConfig
from ana.models.core import ANARLAgent
from ana.utils.rl_env import SimpleGridWorld
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json

@ExperimentRegistry.register(phase=4, name="action")
class ActionExperiment(ExperimentBase):
    @property
    def name(self) -> str: return "action"
    @property
    def phase(self) -> int: return 4

    def execute(self, quick: bool = False, **kwargs):
        self.results.log("Starting Phase 4: Action (RL)")

        # Hyperparameters
        grid_size = 4
        episodes = 50 if quick else 500
        max_steps = 10 if quick else 20
        lr = 1e-3
        gamma = 0.99

        # Config
        config = ANAConfig(
            d_model=32, state_dim=32, num_layers=2, track_count=2,
            observation_space=grid_size*grid_size,
            action_space=4,
            max_thinking_steps=1
        )

        agent = ANARLAgent(config)
        optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

        env = SimpleGridWorld(grid_size=grid_size, max_steps=max_steps)

        stats = {'rewards': [], 'lengths': []}
        best_reward = -float('inf')

        for ep in range(episodes):
            obs = env.reset() # tensor
            done = False
            state = None

            log_probs = []
            values = []
            rewards = []

            while not done:
                # Add batch dim
                obs_b = obs.unsqueeze(0)

                # Unpack 4 return values (including step_info)
                logits, value, state, step_info = agent(obs_b, state)

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                action = dist.sample()
                log_prob = dist.log_prob(action)

                obs, r, done, _ = env.step(action.item())

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(r)

            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Update
            policy_loss = []
            value_loss = []

            for log_prob, val, R in zip(log_probs, values, returns):
                advantage = R - val.item()
                policy_loss.append(-log_prob * advantage)
                value_loss.append(F.mse_loss(val.squeeze(), R.clone().detach().to(val.device))) # Ensure scalar tensor match

            if policy_loss:
                loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ep_reward = sum(rewards)
            stats['rewards'].append(ep_reward)
            stats['lengths'].append(len(rewards))

            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(agent.state_dict(), os.path.join(self.results.output_dir, "best_agent.pt"))

            if (ep+1) % 10 == 0:
                avg_r = np.mean(stats['rewards'][-10:])
                self.results.log(f"Episode {ep+1}/{episodes}: Avg Reward: {avg_r:.2f}")

        # Save stats
        with open(os.path.join(self.results.output_dir, "rl_stats.json"), 'w') as f:
            json.dump(stats, f)

        # Plot Rewards
        self.plot_rewards(stats['rewards'])

        # Visualize Episode
        self.visualize_episode(agent, env)

        self.results.log("Phase 4 Complete.")

    def plot_rewards(self, rewards):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title("RL Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(self.results.output_dir, "rewards.png"))
        plt.close()

    def visualize_episode(self, agent, env):
        obs = env.reset()
        done = False
        state = None
        path = [env.state]

        # Track gating
        ret_gates = []
        mix_weights = []

        while not done:
            obs_b = obs.unsqueeze(0)
            with torch.no_grad():
                logits, value, state, step_info = agent(obs_b, state)
                action = torch.argmax(logits, dim=-1).item()

            obs, r, done, _ = env.step(action)
            path.append(env.state)

            # Extract info from last layer
            layer_info = step_info['layers'][-1]
            if 'ret_gate' in layer_info:
                ret_gates.append(layer_info['ret_gate'].item())
            if 'mix_weights' in layer_info:
                mix_weights.append(layer_info['mix_weights'][0, :, 0].numpy()) # [Tracks]

        # Plot Path
        self.plot_grid_path(path, env.grid_size)

        # Plot Gating
        if ret_gates:
            plt.figure(figsize=(10, 4))
            plt.plot(ret_gates, label="Retrieval Gate", marker='o')
            plt.title("Memory Retrieval During Episode")
            plt.xlabel("Step")
            plt.ylabel("Gate Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.results.output_dir, "episode_gating.png"))
            plt.close()

    def plot_grid_path(self, path, grid_size):
        grid = np.zeros((grid_size, grid_size))

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='Greys', extent=[0, grid_size, grid_size, 0])
        plt.grid(True)

        # Plot path
        path = np.array(path)
        # Add 0.5 to center in cells
        plt.plot(path[:, 1] + 0.5, path[:, 0] + 0.5, marker='o', color='red', linewidth=2, label='Agent Path')

        # Start/Goal
        plt.plot(0.5, 0.5, 'go', markersize=15, label='Start')
        plt.plot(grid_size-0.5, grid_size-0.5, 'bo', markersize=15, label='Goal')

        plt.legend()
        plt.title("Agent Path in Grid World")
        plt.savefig(os.path.join(self.results.output_dir, "agent_path.png"))
        plt.close()
