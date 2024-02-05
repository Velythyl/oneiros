import gym
import numpy as np
import torch
from gym import ObservationWrapper, spaces, Wrapper


class LastAct(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = self.observation_space.low
        hig = self.observation_space.high

        act_space_low = self.action_space.low
        act_space_high = self.action_space.high

        low = np.hstack(low, act_space_low)
        high = np.hstack(hig, act_space_high)
        self.observation_space = gym.spaces.Box(low=low, high=high)

    def make_obs(self, obs, last_action):
        ret = torch.vstack((obs, last_action))
        return ret

    def step(self, action):
        last_action = self.get_stacks().action[-1]
        obs, rew, done, info = super(LastAct, self).step(action)
        obs = self.make_obs(obs, last_action)
        return obs, rew, done, info

    def reset(self, **kwargs):
        last_action = self.get_stacks().action[-1]
        obs = super(LastAct, self).reset(**kwargs)
        return self.make_obs(obs, last_action)