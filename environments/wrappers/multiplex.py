import gym.spaces
import numpy as np
import torch
from gym import Wrapper


class MultiPlexEnv(Wrapper):
    def __init__(self, env_list, device):
        super().__init__(env_list[0])
        self.env_list = env_list
        self.env_list_len = len(env_list)
        self.device = device
        self.num_envs_per_env = env_list[0].observation_space.shape[0]

        obs_space_shape = (self.num_envs_per_env * len(env_list), *self.observation_space.shape[1:])
        self.observation_space = gym.spaces.Box(low=np.ones(obs_space_shape) * -np.inf, high=np.ones(obs_space_shape) * np.inf)
        act_space_shape = (self.num_envs_per_env * len(env_list), *self.action_space.shape[1:])
        self.action_space = gym.spaces.Box(low=np.ones(act_space_shape) * -np.inf, high=np.ones(act_space_shape) * np.inf)

        self.num_envs = self.num_envs_per_env * self.env_list_len

        self.env_map_to_name = []
        for env in self.env_list:
            self.env_map_to_name.append(env.ONEIROS_METADATA.env_key)

    def reset(self):
        obs_s = []
        for env in self.env_list:
            obs_s.append(env.reset())
        return torch.concat((obs_s))

    def render(self, *args, **kwargs):
        obs_s = []
        for env in self.env_list:
            obs_s.append(env.render(*args, **kwargs))
        return obs_s

    def step(self, action):
        obs_s = []
        rew_s = []
        done_s = []
        info_s = {}

        start = 0
        for i, stop in enumerate(range(self.num_envs_per_env, (1+self.env_list_len) * self.num_envs_per_env, self.num_envs_per_env)):
            acts = action[start:stop]
            start = stop

            obs, rew, done, info = self.env_list[i].step(acts)
            obs_s.append(obs)
            rew_s.append(rew)
            done_s.append(done)
            info_s.update(info)

        obs_s = torch.concat(obs_s)
        rew_s = torch.concat(rew_s)
        done_s = torch.concat(done_s)
        return obs_s, rew_s, done_s, info_s
