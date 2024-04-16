import gym.spaces
import numpy as np
import torch
from gym import Wrapper


class MatFrameStackEnv(Wrapper):
    def __init__(self, env, device, num_stack):
        super().__init__(env)

        print(device)

        self.device = device
        self.num_stack = num_stack


        assert len(self.observation_space.shape[1:]) == 1

        self.obs_space_shape = (self.observation_space.shape[0], self.num_stack, *self.observation_space.shape[1:])
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_space_shape)
        #low=np.ones(self.obs_space_shape) * -np.inf,
        #                                        high=np.ones(self.obs_space_shape) * np.inf)
        self.reset_buf(None)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_space_shape)

    def reset_buf(self, mask):
        if mask is None:
            self.buf = torch.zeros(self.obs_space_shape, device=self.device)
            self.buf_plex = torch.arange(self.obs_space_shape[0], dtype=torch.int32, device=self.device)
        else:
            mask = mask.bool()
            self.buf[mask] = self.buf[mask] * 0

    def add_obs_to_buf(self, obs):
        def roll_per_idx(buf):
            return torch.roll(buf, 1, dims=(0,))

        buf = torch.vmap(roll_per_idx)(self.buf)
        buf[self.buf_plex, 0] = obs

        self.buf = buf

    def reset(self, **kwargs):
        self.reset_buf(None)

        obs = super(MatFrameStackEnv, self).reset(**kwargs)
        self.add_obs_to_buf(obs)
        return self.buf

    def step(self, action):
        obs, rew, done, info = super(MatFrameStackEnv, self).step(action)

        self.reset_buf(mask=done)

        self.add_obs_to_buf(obs)

        return self.buf, rew, done, info
