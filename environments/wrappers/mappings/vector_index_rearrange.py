import brax
import gymnasium
import jax.random
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

        def assert_dico(dico):
            keyset = set(dico.keys())
            valset = set(dico.values())
            assert keyset == valset

        def invert_dico(dico):
            return {v: k for k, v in dico.items()}

        def make_mapping_matrix(dico):
            keyset = set(dico.keys())
            valset = set(dico.values())

            mapping = np.zeros(len(valset), dtype=int)
            for i in range(len(valset)):
                mapping[i] = dico[i]

            return mapping

        self.act = act
        self.obs = obs
        self.mass = mass

        obs2brax, act2brax = self.obs, self.act

        if act2brax is None:
            self._act2brax = None
            self._brax2act = None
        else:
            assert_dico(act2brax)

            brax2act = invert_dico(act2brax)

            self._brax2act = make_mapping_matrix(brax2act)
            self._act2brax = make_mapping_matrix(act2brax)

        if obs2brax is None:
            self._obs2brax = None
            self._brax2obs = None
        else:
            #assert_dico(obs2brax)

            brax2obs = invert_dico(obs2brax)

            self._brax2obs = make_mapping_matrix(brax2obs)
            self._obs2brax = make_mapping_matrix(obs2brax)

        if self.mass is None:
            def read_mass(env):
                return env.unwrapped.model.body_mass
        else:
            def read_mass(env):
                masses = np.zeros(len(np.unique(np.array(list(self.mass.values())))))
                for mujo_key, brax_key in self.mass.items():
                    masses[brax_key] = masses[brax_key] + env.unwrapped.model.body_mass[mujo_key]
                return masses

        self._read_mass = read_mass

    def obs2brax(self, obs):
        if self._obs2brax is None:
            return obs
        return obs[self._obs2brax]

    def act2brax(self, obs):
        if self._act2brax is None:
            return obs
        return obs[self._act2brax]

    def brax2act(self, obs):
        if self._brax2act is None:
            return obs
        return obs[self._brax2act]

    def brax2obs(self, obs):
        if self._brax2obs is None:
            return obs
        return obs[self._brax2obs]

class _MujocoMapping(_Mapping):
    pass

def gen_identity_dict(len):
    return {k:k for k in range(len)}

class Ant(_MujocoMapping):
    act: dict = None
    obs: dict = None

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

class Hopper(_MujocoMapping):
    act: dict = None
    obs: dict = None

    mass: dict = None

class inverted_double_pendulum(_MujocoMapping):
    act: dict = None
    obs: dict = gen_identity_dict(8) # note: this throws away the last 3 obs of the mujoco env, which has 11 obs

    mass: dict = None

class inverted_pendulum(_MujocoMapping):
    act: dict = None
    obs: dict = None

    mass: dict = None

"""
this does not work, i think brax includes the ball mass in the body_mass and mujoco doesnt (since index 8 of brax is not found anywhere in mujoco, even when summing up indices appropriately)
class Pusher(_MujocoMapping):
    act: dict = None
    obs: dict = None

    mass: dict = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 4,
        6: 5,
        7: 5,
        8: 6,
        9: 7,
        10: 7,
        11: 7, 12: 7    # these are both 0 in the mujoco mass, i.e. noop
    }
"""

class Reacher(_MujocoMapping):
    act: dict = None
    obs: dict = None

    mass: dict = {
        0: 0,
        1:1,
        2:2,
        3:2,
        4:3
    }

class Walker2d(_MujocoMapping):
    act: dict = None
    obs: dict = None

    mass: dict = None

class Go1(_MujocoMapping):
    act: dict = None
    obs: dict = None

    mass: dict = None

class Widow(_MujocoMapping):
    act: dict = None
    obs: dict = None

    mass: dict = None

def map_func_lookup(parent_class, brax_envname: str) -> _Mapping:
    get_mapping_class = list(parent_class.__subclasses__())
    get_mapping_class = {str(c).split(".")[-1].split("'")[0].lower(): c for c in get_mapping_class}
    mapping_class = get_mapping_class[brax_envname]
    mapping_class = mapping_class(mapping_class.obs, mapping_class.act, mapping_class.mass)

    return mapping_class


if __name__ == "__main__":
    # DONE: ant, hopper
    # INCOMPAT: cheetah, humanoid, humanoidstandup
    # BROKEN IN MUJOCO: swimmer

    import brax
    from brax.envs.wrappers.torch import TorchWrapper

    from environments.customenv.braxcustom.widow_reacher import WidowReacher
    from environments.customenv.mujococustom.widow_reacher import WidowReacher



    mujoco = gymnasium.make("Widow", max_episode_steps=1000, autoreset=True)

    brax_env = brax.envs.create(env_name="widow", episode_length=1000, backend="generalized",
                                batch_size=2, no_vsys=True)

    reset = (brax_env.reset)
    step = (brax_env.step)
    #brax_reset_state = reset(jax.random.PRNGKey(0))
    #brax_step_state = step(brax_reset_state, jax.numpy.zeros((2, brax_env.action_size)))

    mujoco_reset_state = mujoco.reset()
    mujoco_step_state = mujoco.step(action=mujoco.action_space.sample()*0)

    #mujoco_mass = mujoco.unwrapped.model.body_mass
    #brax_mass = state.sys.body_mass

    obs, act = map_func_lookup("ant")

    x = act(np.arange(8))
    i = 0
