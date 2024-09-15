import gym.spaces
import numpy as np
import torch
from gym import Wrapper

from environments.wrappers.sim2real.matrix_framestack import MatFrameStackEnv


class VecFrameStackEnv(Wrapper):
    def __init__(self, env, device, num_stack):
        super().__init__(
            MatFrameStackEnv(env, device, num_stack)
        )

        assert len(env.observation_space.shape[1:]) == 1
        NUM_OBS = env.observation_space.shape[1]

        self.obs_space_shape = (self.observation_space.shape[0], self.num_stack * NUM_OBS)
        self.observation_space = gym.spaces.Box(low=np.ones(self.obs_space_shape) * -np.inf,
                                                high=np.ones(self.obs_space_shape) * np.inf)

    def reset(self, **kwargs):
        obs = super(VecFrameStackEnv, self).reset(**kwargs)

        return flatten_obs(obs)

    def step(self, action):
        obs, rew, done, info = super(VecFrameStackEnv, self).step(action)

        return flatten_obs(obs), rew, done, info

def flatten_obs(obs):
    return torch.vmap(torch.flatten)(obs)
