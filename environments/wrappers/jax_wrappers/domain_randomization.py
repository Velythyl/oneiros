import jax.random
from brax.envs.wrappers.vsys import SysKeyRange
from gym import Wrapper

from environments.wrappers.jax_wrappers.vectorgym import VectorGymWrapper


def marshall_str(string):
    if isinstance(string, str):
        return {
            "none": None,
            "true": True,
            "false": False
        }[string.lower()]
    return string


def DomainRandWrapper(env, percent_below, percent_above, do_on_reset, do_on_N_step, do_at_creation, seed):
    from brax.envs.wrappers.vsys import DomainRandVSysWrapper

    mass = env.sys.body_mass
    min = mass * percent_below
    max = mass * percent_above
    skr = SysKeyRange(key="body_mass", base=mass, min=min, max=max)

    if isinstance(seed, int):
        seed = jax.random.PRNGKey(seed)

    do_on_reset = marshall_str(do_on_reset)
    do_on_N_step = marshall_str(do_on_N_step)
    do_at_creation = marshall_str(do_at_creation)

    return DomainRandVSysWrapper(env, [skr], seed, do_on_reset=do_on_reset, do_at_creation=do_at_creation, do_every_N_step=do_on_N_step)

class WritePrivilegedInformationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, VectorGymWrapper)

    @property
    def current_sys(self):
        return self.env._state.sys

    def read_mass(self):
        return self.current_sys.body_mass

    def step(self, action):
        ret = super(WritePrivilegedInformationWrapper, self).step(action)
        ret[-1]["priv"] = self.read_mass()
        return ret
