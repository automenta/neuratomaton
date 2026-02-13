import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ana.research.phase4_rl.agent import ANARLAgent
from ana.config import ANAConfig
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=4, name="train_rl")
class RLTrainerExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "train_rl"

    @property
    def phase(self) -> int:
        return 4

    def setup(self):
        self.agent = ANARLAgent(self.config).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-3)
        self.rewards_log = []

    def train_step(self, obs, action, reward, next_obs, done):
        """
        Dummy training step.
        """
        obs = obs.to(self.device)

        # Forward
        output = self.agent(obs)
        if len(output) == 3:
            logits, value, _ = output
        else:
            logits, value = output

        # Assuming logits shape (1, 1, ActDim) or similar.
        if logits.dim() == 3:
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
        self.results.save_plot("learning_curve.png")

    def execute(self):
        self.results.log("Simulating RL training steps...")
        final_loss = 0
        for i in range(50):
            # Dummy interaction: reward increases over time
            obs = torch.randn(1, 10)
            reward = i * 0.1 + torch.randn(1).item()
            loss = self.train_step(obs, action=1, reward=reward, next_obs=obs, done=False)
            final_loss = loss

        self.results.log(f"Final Loss: {final_loss:.4f}")
        self.results.save_json("rl_results.json", {"final_loss": final_loss, "steps": 50})
        self.plot_learning_curve()

if __name__ == "__main__":
    config = ANAConfig(observation_space=10, action_space=4, d_model=32)
    exp = RLTrainerExperiment(config)
    exp.run()
