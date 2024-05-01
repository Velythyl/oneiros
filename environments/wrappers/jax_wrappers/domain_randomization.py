from typing import Union

import gymnasium
import jax.random
import numpy as np
from brax.envs.wrappers.vsys import SysKeyRange
from flax import struct
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

import jax.numpy as jp

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
        return self.env._state.current_sys

    def read_mass(self):
        return self.current_sys.body_mass

    def step(self, action):
        ret = super(WritePrivilegedInformationWrapper, self).step(action)
        ret[-1]["priv"] = self.read_mass()
        return ret

if __name__ == "__main__":
    import brax
    import brax.envs
    import jax.numpy as jp

    env = brax.envs.create(env_name="ant", episode_length=1000, backend="mjx",
                           batch_size=16) # EP LEN, NUM_ENV

    autoreset_env = env
    vmap_env = env.env
    episode_env = env.env.env
    base_env = env.env.env.env

    from brax.envs.wrappers import training
    assert isinstance(autoreset_env, training.AutoResetWrapper)
    assert isinstance(vmap_env, training.VmapWrapper)
    assert isinstance(episode_env, training.EpisodeWrapper)

    base_sys = base_env.sys

    from brax.envs.base import Env, State, Wrapper


    def set_mass(new_mass):
        new_sys = base_sys.replace(body_mass=new_mass)




    x = env.unwrapped.sys.body_mass
    print(x)
    print(jp.unique(x, return_counts=True))
    exit()
    x=0

    def eval(low, high):
        baselines = []

        for _ in range(1000):
            np.random.seed(1)
            env = gymnasium.make("Ant-v4", max_episode_steps=1000, autoreset=True)
            env = MujocoDomainRandomization(env, low, high, do_on_reset=False, do_on_N_step=10)
            x = env.reset(seed=1)
            for i in range(20):
                x = env.step(action=env.action_space.sample() * 0)[0]
            baselines.append(x)

        baselines = np.vstack(baselines)
        baselines_mean = baselines.mean(axis=0)
        baselines_std = baselines.std(axis=0)
        return baselines_mean[5], baselines_std[5]

    print(eval(1., 1.)) # EQ
    print(eval(1., 1000.)) # DIFF
    print(eval(1., 1.)) # EQ
