import torch
import torch.nn as nn
from ana.models import LinearRecurrentUnit, HoloLink, HyperController
from ana.config import ANAConfig

class ANARLAgent(nn.Module):
    """
    ANA Agent for RL tasks.
    Takes observation vector, outputs action logits and value estimate.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        self.obs_dim = config.observation_space
        self.act_dim = config.action_space

        self.input_proj = nn.Linear(self.obs_dim, self.d_model)

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer_dict = nn.ModuleDict()
            if config.use_controller:
                layer_dict['controller'] = HyperController(config)

            layer_dict['tracks'] = nn.ModuleList([
                LinearRecurrentUnit(config) for _ in range(config.track_count)
            ])

            # HoloLink for episodic memory
            if config.use_hololink:
                layer_dict['holo'] = HoloLink(config, input_dim=config.state_dim * config.track_count)

            self.layers.append(layer_dict)

        self.norm = nn.LayerNorm(self.d_model)

        # Actor-Critic heads
        self.actor = nn.Linear(self.d_model, self.act_dim)
        self.critic = nn.Linear(self.d_model, 1)

    def forward(self, obs, prev_hidden=None):
        # obs: (B, ObsDim) -> (B, 1, ObsDim) sequence length 1
        x = self.input_proj(obs).unsqueeze(1)

        # Running statefully? RL usually runs step-by-step.
        # Here we assume we process a sequence or just one step.
        # For simplicity, let's assume we process one step and ignore passing hidden state explicitly
        # (relying on internal recurrence if we were processing a sequence, but for step-by-step
        # we need to maintain state external to this forward if we want memory).

        # BUT, standard RL loops (like PPO) often process a rollout sequence during update.
        # During inference (interaction), it's step-by-step.

        # Let's assume this forward is for a sequence (training) or single step (inference).

        # Simplified: just run tracks on the input.

        for layer in self.layers:
            track_outs = []
            for track in layer['tracks']:
                yt, _ = track.forward_sequence(x)
                track_outs.append(yt)

            x = x + torch.stack(track_outs).mean(dim=0)

        x = self.norm(x)

        policy_logits = self.actor(x)
        value = self.critic(x)

        return policy_logits, value

if __name__ == "__main__":
    config = ANAConfig(observation_space=10, action_space=4, d_model=32)
    agent = ANARLAgent(config)
    obs = torch.randn(1, 10)
    logits, val = agent(obs)
    print(f"Agent Output: {logits.shape}, Value: {val.item():.4f}")
