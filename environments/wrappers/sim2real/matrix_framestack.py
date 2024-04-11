import gym.spaces
import numpy as np
import torch
from gym import Wrapper


class MatFrameStackEnv(Wrapper):
    def __init__(self, env, device, num_stack):
        super().__init__(env)
        self.device = device
        self.num_stack = num_stack


        assert len(self.observation_space.shape[1:]) == 1

        self.obs_space_shape = (self.observation_space.shape[0], self.num_stack, *self.observation_space.shape[1:])
        self.observation_space = gym.spaces.Box(low=np.ones(self.obs_space_shape) * -np.inf,
                                                high=np.ones(self.obs_space_shape) * np.inf)

    def reset_buf(self):
        self.buf = torch.zeros(self.obs_space_shape).to(self.device)
        self.buf_idx = self.num_stack-1

    def add_obs_to_buf(self, obs):
        self.buf[self.buf_idx] = obs

        self.buf_idx -= 1
        if self.buf_idx < 0:
            self.buf_idx = 0

    def reset(self, **kwargs):
        obs = super(MatFrameStackEnv, self).reset(**kwargs)

        self.add_obs_to_buf(obs)

        return self.buf

    def step(self, action):
        obs, rew, done, info = super(MatFrameStackEnv, self).step(action)

        self.add_obs_to_buf(obs)

        return self.buf, rew, done, info
