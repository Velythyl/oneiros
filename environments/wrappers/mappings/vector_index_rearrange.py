import gymnasium
import numpy as np
from gym import Wrapper


class VectorIndexMapWrapper(Wrapper):
    def __init__(self, env, mapping):
        super(VectorIndexMapWrapper, self).__init__(env)

        self.mapping = mapping

        self.observation_space = gymnasium.spaces.Box(
            low=self.mapping.obs2brax(self.observation_space.low),
            high=self.mapping.obs2brax(self.observation_space.high)
        )

        self.action_space = gymnasium.spaces.Box(
            low=self.mapping.act2brax(self.action_space.low),
            high=self.mapping.act2brax(self.action_space.high)
        )

    def read_mass(self):
        return self.mapping._read_mass(self.unwrapped)

    def reset(self, **kwargs):
        obs = super(VectorIndexMapWrapper, self).reset(**kwargs)

        if isinstance(obs, tuple) and len(obs) == 2:
            return self.mapping.obs2brax(obs[0]), obs[1]

        return self.mapping.obs2brax(obs)

    def step(self, action):
        # receives a BRAX action, makes it into a MUJOCO action, obtains a MUJOCO obs, returns a BRAX obs
        action = self.mapping.brax2act(action)
        ret = super(VectorIndexMapWrapper, self).step(action)
        return self.mapping.obs2brax(ret[0]), *ret[1:]


class _Mapping:
    """
    mappings are ALWAYS mujo -> brax

    act: dict = None    # mujo -> brax
    obs: dict = None    # mujo -> brax
    body_mass: dict = None # mujo -> brax
    """

    def __init__(self,
                 obs: dict,
                 act: dict,
                 mass: dict
                 ):

        self.act = act
        self.obs = obs
        self.mass = mass

        def assert_dico(dico):
            keyset = set(dico.keys())
            valset = set(dico.values())
            assert keyset == valset

        obs2brax, act2brax = self.obs, self.act
        assert_dico(obs2brax)
        assert_dico(act2brax)
        # DONT assert mass

        def invert_dico(dico):
            return {v: k for k, v in dico.items()}

        brax2act = invert_dico(act2brax)
        brax2obs = invert_dico(obs2brax)

        def make_mapping_matrix(dico):
            keyset = set(dico.keys())
            valset = set(dico.values())

            mapping = np.zeros(len(valset), dtype=int)
            for i in range(len(valset)):
                mapping[i] = dico[i]

            return mapping

        self._obs2brax = make_mapping_matrix(obs2brax)
        self._brax2obs = make_mapping_matrix(brax2obs)
        self._act2brax = make_mapping_matrix(act2brax)
        self._brax2act = make_mapping_matrix(brax2act)

        def read_mass(env):#, return_contributors=False):
            masses = np.zeros(len(np.unique(np.array(list(self.mass.values())))))
            #contributors = [[] for _ in range(len(self.mass)]
            #contributors_masses = [[] * len(self.mass)]
            for mujo_key, brax_key in self.mass.items():
                masses[brax_key] = masses[brax_key] + env.unwrapped.model.body_mass[mujo_key]
            #    contributors[brax_key].append(mujo_key)
            #    contributors_masses[brax_key].append(env.unwrapped.model.body_mass[mujo_key])

            #if return_contributors:
            #    return masses, contributors, contributors_masses
            #else:
            return masses

        self._read_mass = read_mass

    def obs2brax(self, obs):
        return obs[self._obs2brax]

    def act2brax(self, obs):
        return obs[self._act2brax]

    def brax2act(self, obs):
        return obs[self._brax2act]

    def brax2obs(self, obs):
        return obs[self._brax2obs]

class _MujocoMapping(_Mapping):
    pass

def gen_identity_dict(len):
    return {k:k for k in range(len)}

class Ant(_MujocoMapping):
    #"""
    act: dict = gen_identity_dict(8)
    # https://wandb.ai/velythyl/oneiros_framestack_sweep/runs/pkmeeyi2?nw=nwuservelythyl
    #"""

    """
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
    # https://wandb.ai/velythyl/oneiros_framestack_sweep/runs/wuxc732q?nw=nwuservelythyl
    #"""

    obs: dict = gen_identity_dict(27)

    """
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
    """

    mass: dict = {
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 3,
        5: 1,
        6: 4,
        7: 5,
        8: 1,
        9: 6,
        10: 7,
        11: 1,
        12: 8,
        13: 9
    }

def map_func_lookup(parent_class, mujoco_name: str) -> _Mapping:
    get_mapping_class = list(parent_class.__subclasses__())
    get_mapping_class = {str(c).split(".")[-1].split("'")[0].lower(): c for c in get_mapping_class}
    mapping_class = get_mapping_class[mujoco_name]
    mapping_class = mapping_class(mapping_class.obs, mapping_class.act, mapping_class.mass)

    return mapping_class


if __name__ == "__main__":
    obs, act = map_func_lookup("ant")

    x = act(np.arange(8))
    i = 0
