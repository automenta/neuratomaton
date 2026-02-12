import torch
import torch.optim as optim
from ana.research.phase4_rl.agent import ANARLAgent
from ana.config import ANAConfig
import torch.nn.functional as F

class RLTrainer:
    def __init__(self, agent: ANARLAgent, device="cpu"):
        self.agent = agent.to(device)
        self.device = device
        self.optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    def train_step(self, obs, action, reward, next_obs, done):
        """
        Dummy training step (like a simplified PPO or A2C update).
        In reality, would take a batch of rollouts.
        """
        obs = obs.to(self.device)

        # Forward
        logits, value = self.agent(obs)
        logits = logits.squeeze(1) # (B, ActDim)

        # Dummy loss
        # Maximize logits for taken action (imitation learning style for dummy)
        target_action = torch.tensor([action], device=self.device)
        policy_loss = F.cross_entropy(logits, target_action)

        value_loss = (value - reward)**2

        loss = policy_loss + value_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    config = ANAConfig(observation_space=10, action_space=4, d_model=32)
    agent = ANARLAgent(config)
    trainer = RLTrainer(agent)

    # Dummy interaction
    obs = torch.randn(1, 10)
    loss = trainer.train_step(obs, action=1, reward=1.0, next_obs=obs, done=False)
    print(f"RL Training Step Loss: {loss:.4f}")
