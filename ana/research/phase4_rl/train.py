import torch
import torch.optim as optim
from ana.research.phase4_rl.agent import ANARLAgent
from ana.config import ANAConfig
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class RLTrainer:
    def __init__(self, agent: ANARLAgent, device="cpu"):
        self.agent = agent.to(device)
        self.device = device
        self.optimizer = optim.Adam(agent.parameters(), lr=1e-3)
        self.rewards_log = []
        self.results_dir = "results/phase4_rl"
        os.makedirs(self.results_dir, exist_ok=True)

    def train_step(self, obs, action, reward, next_obs, done):
        """
        Dummy training step (like a simplified PPO or A2C update).
        In reality, would take a batch of rollouts.
        """
        obs = obs.to(self.device)

        # Forward
        output = self.agent(obs)
        if len(output) == 3:
            logits, value, _ = output
        else:
            logits, value = output

        logits = logits.squeeze(1) # (B, ActDim) or (B, 1, ActDim) if squeezed already?
        # If logits is (B, ActDim) from step, squeeze(1) might fail or squeeze nothing.
        if logits.dim() == 2:
             pass # Already (B, ActDim)
        elif logits.dim() == 3:
             logits = logits.squeeze(1)

        # Dummy loss
        # Maximize logits for taken action (imitation learning style for dummy)
        target_action = torch.tensor([action], device=self.device)
        policy_loss = F.cross_entropy(logits, target_action)

        value_loss = (value - reward)**2

        loss = policy_loss + value_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards_log.append(reward)
        return loss.item()

    def plot_learning_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_log, label='Reward')
        plt.title("RL Training Progress")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.results_dir, "learning_curve.png")
        plt.savefig(save_path)
        print(f"Learning curve saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    config = ANAConfig(observation_space=10, action_space=4, d_model=32)
    agent = ANARLAgent(config)
    trainer = RLTrainer(agent)

    # Simulate an episode
    print("Simulating RL training steps...")
    for i in range(100):
        # Dummy interaction: reward increases over time
        obs = torch.randn(1, 10)
        reward = i * 0.1 + torch.randn(1).item()
        loss = trainer.train_step(obs, action=1, reward=reward, next_obs=obs, done=False)

    trainer.plot_learning_curve()
