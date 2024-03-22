import numpy as np
import torch
from gym import Wrapper


class VectorIndexMapWrapper(Wrapper):
    def __init__(self, env, mujoco_name):
        super(VectorIndexMapWrapper, self).__init__(env)

        self.obs_func, self.act_func = map_func_lookup(mujoco_name)

    def reset(self, **kwargs):
        return self.obs_func(super(VectorIndexMapWrapper, self).reset(**kwargs))

    def step(self, action):
        action = self.act_func(action)
        ret = super(VectorIndexMapWrapper, self).step(action)
        return self.obs_func(ret), *ret[1:]

class _Mapping:
    act: dict = None
    obs: dict = None

class Ant(_Mapping):
    act: dict = {
        0: 6,
        1: 7,
        2: 0,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5
    }

    obs: dict = {
        0: 0,
        1: 2,
        2: 3,
        3: 4,
        4: 1,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        20: 20,
        21: 21,
        22: 22,
        23: 23,
        24: 24,
        25: 25,
        26: 26
    }


def map_func_lookup(mujoco_name: str):
    get_mapping_class = list(_Mapping.__subclasses__())
    get_mapping_class = {str(c).split(".")[-1].split("'")[0]: c for c in get_mapping_class}
    mapping_class = get_mapping_class[mujoco_name]

    def make_mapping_matrix(dico):
        keyset = set(dico.keys())
        valset = set(dico.values())
        assert keyset == valset

        mapping = torch.zeros(len(valset))
        for i in range(len(valset)):
            mapping[i] = dico[i]

        def func(vec):
            return mapping[vec]
        return func

    return make_mapping_matrix(mapping_class.obs), make_mapping_matrix(mapping_class.act)


if __name__ == "__main__":
    obs, act = map_func_lookup("Ant")

    x = act(np.arange(8))
    i=0






