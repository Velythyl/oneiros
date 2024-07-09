import numpy as np
import torch
from gym import Wrapper

def np2torch(np_arr):
    #np_arr = np.copy(np_arr)

    ret = torch.from_numpy(np_arr).detach()
    ret.requires_grad = False
    return ret

def torch2np(torch_t):
    #torch_t = torch.clone(torch_t)
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

        final_info = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                try:
                    if v.dtype == object:
                        new_v = []
                        non_none = None
                        for _v in v:
                            new_v.append(_v)
                            if _v is not None:
                                non_none = _v
                        new_v2 = []
                        for _v in new_v:
                            if _v is None:
                                _v = non_none * 0
                            new_v2.append(_v)
                        v = np.vstack(new_v2)
                    v = np2torch(v).to(self.device)
                except:
                    pass
            final_info[k] = v

        return np2torch(obs).to(self.device), np2torch(rew).to(self.device), np2torch(done).to(self.device).int(), final_info #dict2torch(info, self.device)

if __name__ == "__main__":
    np2torch