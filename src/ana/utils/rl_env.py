import numpy as np
import torch

class SimpleGridWorld:
    """
    Simple Grid World Environment for RL testing.
    Goal: Reach the bottom-right corner (grid_size-1, grid_size-1).
    """
    def __init__(self, grid_size=4, max_steps=20):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.state = None
        self.steps = 0

        self.action_space = 4 # Up, Down, Left, Right
        self.observation_space = grid_size * grid_size # One-hot

    def reset(self):
        self.state = (0, 0)
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        r, c = self.state

        if action == 0: # Up
            r = max(0, r - 1)
        elif action == 1: # Down
            r = min(self.grid_size - 1, r + 1)
        elif action == 2: # Left
            c = max(0, c - 1)
        elif action == 3: # Right
            c = min(self.grid_size - 1, c + 1)

        self.state = (r, c)
        self.steps += 1

        done = False
        reward = -0.1

        if self.state == (self.grid_size - 1, self.grid_size - 1):
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        r, c = self.state
        obs = np.zeros(self.observation_space, dtype=np.float32)
        idx = r * self.grid_size + c
        obs[idx] = 1.0
        return torch.tensor(obs)
