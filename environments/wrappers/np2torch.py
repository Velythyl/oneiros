import numpy as np
import torch
from gym import Wrapper

def np2torch(np_arr):
    np_arr = np.copy(np_arr)

    return torch.from_numpy(np_arr)

def torch2np(torch_t):
    torch_t = torch.clone(torch_t)
    return torch_t.detach().cpu().numpy()

def dict2torch(np_dict, device):
    ret = {}
    for k, v in np_dict.items():
        if isinstance(v, np.ndarray):
            ret[k] = np2torch(v).to(device)
        else:
            ret[k] = v
    return ret

class Np2TorchWrapper(Wrapper):
    def __init__(self, env, device):
        super(Np2TorchWrapper, self).__init__(env)
        self.device = device

    def reset(self, **kwargs):
        ret = super(Np2TorchWrapper, self).reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            return np2torch(ret[0]), ret[1]
        return np2torch(ret).to(self.device)

    def step(self, action):
        action = torch2np(action)
        obs, rew, done, info = super(Np2TorchWrapper, self).step(action)
        return np2torch(obs).to(self.device), np2torch(rew).to(self.device), np2torch(done).to(self.device).int(), info #dict2torch(info, self.device)

if __name__ == "__main__":
    np2torch