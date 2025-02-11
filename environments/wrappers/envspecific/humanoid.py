import gym
import numpy as np
import torch
from tqdm import tqdm


class HumanoidObsCrop(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

        self.obs_space_shape = (self.observation_space.shape[0], 45)
        self.observation_space = gym.spaces.Box(low=np.ones(self.obs_space_shape) * -np.inf,
                                                high=np.ones(self.obs_space_shape) * np.inf)

    def obs(self, obs):
        return obs[:,:45]

    def reset(self):
        return self.obs(super().reset())

    def step(self, action):
        obs, rew, done, info = self.inner_step(self.action_state)

        obs = self.obs(obs)

        return obs, rew, done, info