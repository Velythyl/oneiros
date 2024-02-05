import numpy as np
import torch
import gym

from environments.wrappers.utils import augment_box


class OneHotBehaviour(gym.Wrapper):
    def __init__(self, env, n_behaviours, initial_behaviour):
        super().__init__(env)
        self.n_behaviours = n_behaviours
        self.num_envs = env.num_envs

        # import pdb; pdb.set_trace()

        # self.behaviour_obs_space = gym.spaces.Box(low=marshall2np(0), high=marshall2np(self.n_behaviours))
        behaviour_oh_low = np.zeros((self.num_envs, self.n_behaviours))
        behaviour_oh_high = np.ones((self.num_envs, self.n_behaviours))

        self.behaviour_obs_space = gym.spaces.Box(low=behaviour_oh_low, high=behaviour_oh_high)
        self.og_obs_space = self.observation_space
        
        self.observation_space = augment_box(self.observation_space, self.behaviour_obs_space)
        self.single_observation_space = gym.spaces.Box(low=self.observation_space.low.min(), high=self.observation_space.high.max(), shape=self.observation_space.shape[1:])
        
        # self.current_behaviour = torch.ones((1,self.num_envs), dtype=torch.int32, requires_grad=False) * initial_behaviour
        self.current_behaviour = torch.zeros((self.num_envs,self.n_behaviours), dtype=torch.int32, requires_grad=False)
        self.current_behaviour[:,initial_behaviour] = 1
        self.current_behaviour = self.current_behaviour.to(device=self.device)


    def observation(self, observation):
        concat_obs = torch.concat((observation, self.current_behaviour),axis=1)
        return concat_obs

    # def resample_behaviour(self):
    #     self.current_behaviour = self.current_behaviour.uniform(0, self.n_behaviours)

    def reset(self, **kwargs):
        # self.resample_behaviour()
        # obs = super(OneHotBehaviour, self).reset(**kwargs)
        obs = self.observation(self.env.reset(**kwargs))

        return obs

    def step(self, action):
        observation, rewards, dones, infos = self.env.step(action)
        return self.observation(observation), rewards, dones, infos
