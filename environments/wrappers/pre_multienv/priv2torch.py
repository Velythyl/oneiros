import gym
import numpy as np
import torch

class Priv2Torch(gym.Wrapper):
    def __init__(self, env, priv_key, device):
        super().__init__(env)
        self.priv_key = priv_key
        self.device = device

    def step(self, action):
        ret = super(Priv2Torch, self).step(action)

        if self.priv_key in ret[-1] and isinstance(ret[-1][self.priv_key], np.ndarray):
            ret[-1][self.priv_key] = torch.tensor(ret[-1][self.priv_key]).to(device=self.device)
        #ret[-1]["priv"] = ret[-1][self.priv_key]

        return ret
