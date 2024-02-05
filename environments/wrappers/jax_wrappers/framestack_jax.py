import gym.spaces
import jax
import numpy as np
import torch

from brax.envs.base import Env, State, Wrapper

class FrameStack(Wrapper):
    def __init__(self, env, num_frames, replace_obs):
        super().__init__(env)
        self.num_frames = num_frames

        self.stacks_shape = (env.num_envs, num_frames, self.observation_space.shape[-1])

        self.stacks = jax.zeros(self.stacks_shape)

        self.og_obs_space = env.observation_space
        if replace_obs:
            self.framestack_obs_space = gym.spaces.Box(low=np.ones(self.stacks_shape)*-np.inf, high=np.ones(self.stacks_shape)*np.inf)

        obs_size = self.og_obs_space.shape[-1]

    def reset_stacks(self, env_mask):
        if env_mask is None:
            env_mask = torch.ones(self.env.num_envs, dtype=torch.bool, requires_grad=False, device=self.device)
        env_mask = env_mask.bool()
        self.stacks[env_mask] = 0

    def reset(self, **kwargs):
        self.reset_stacks(None)
        return super(FrameStack, self).reset(**kwargs)

    def add_to_stacks(self, obs):
        # gather is the fastest way to move all elements "down" one row
        # https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch
        # first, move all elements in stacks "down" by one index
        self.stacks = torch.gather(self.stacks, 1, self.roll_indices)
        # then, write current obs to index 0
        self.stacks[self.env_indices, -1] = obs
        # that's it!

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.reset_stacks(done)
        self.add_to_stacks(obs)

        info['framestack'] = self.stacks[:,:,:] # copy
        info['obs'] = obs
        return obs, rew, done, info

