
from ana.research.core import ExperimentBase, ExperimentRegistry
from ana.models.config import ANAConfig
from ana.models.core import ANARLAgent
from ana.utils.rl_env import SimpleGridWorld
import torch
import torch.nn.functional as F
import numpy as np

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

                logits, value, state = agent(obs_b, state)

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

            stats['rewards'].append(sum(rewards))
            stats['lengths'].append(len(rewards))

            if (ep+1) % 10 == 0:
                avg_r = np.mean(stats['rewards'][-10:])
                self.results.log(f"Episode {ep+1}/{episodes}: Avg Reward: {avg_r:.2f}")

        # Save stats
        import json
        with open(self.results.output_dir + "/rl_stats.json", 'w') as f:
            json.dump(stats, f)

        self.results.log("Phase 4 Complete.")
