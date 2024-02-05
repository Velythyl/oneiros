import copy
import functools
import time
from typing import Dict

import jax
import jax.numpy as jp
from brax import math, base, envs
from brax.envs.base import Wrapper, Env, State
from brax.io import mjcf
from flax import struct



if __name__ == "__main__":


    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng, 2)
    # key = jp.reshape(jp.stack(key), (10, 2))
    ret = jax.random.normal(key)

    env = envs.create(
        "halfcheetah",
        backend="spring",
        episode_length=1000,
        auto_reset=True,
        batch_size=4000,
    )


    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    def randact():
        return jax.random.uniform(jax.random.PRNGKey(0), (4000, env.action_size,))

    start = time.time()
    for i in range(100):
        print(i)
        state = env.step(state, randact())
    end = time.time()
    print(end - start)
    i = 9
    exit()

# UTILS

def _write_sys(sys, attr, val):
    """"Replaces attributes in sys with val."""
    if not attr:
        return sys
    if len(attr) == 2 and attr[0] == 'geoms':
        geoms = copy.deepcopy(sys.geoms)
        if not hasattr(val, '__iter__'):
            for i, g in enumerate(geoms):
                if not hasattr(g, attr[1]):
                    continue
                geoms[i] = g.replace(**{attr[1]: val})
        else:
            sizes = [g.transform.pos.shape[0] for g in geoms]
            g_idx = 0
            for i, g in enumerate(geoms):
                if not hasattr(g, attr[1]):
                    continue
                size = sizes[i]
                geoms[i] = g.replace(**{attr[1]: val[g_idx:g_idx + size].T})
                g_idx += size
        return sys.replace(geoms=geoms)
    if len(attr) == 1:
        return sys.replace(**{attr[0]: val})
    return sys.replace(**{attr[0]:
                              _write_sys(getattr(sys, attr[0]), attr[1:], val)})


def set_sys(sys, params: Dict[str, jp.ndarray]):
    """Sets params in the System."""
    for k in params.keys():
        sys = _write_sys(sys, k.split('.'), params[k])
    return sys

def set_sys_capsules(sys, lengths, radii):
    """Sets the system with new capsule lengths/radii."""
    sys2 = set_sys(sys, {'geoms.length': lengths})
    sys2 = set_sys(sys2, {'geoms.radius': radii})

    # we assume inertia.transform.pos is (0,0,0), as is often the case for
    # capsules

    # get the new joint transform
    cur_len = sys.geoms[1].length[:, None]
    joint_dir = jax.vmap(math.normalize)(sys.link.joint.pos)[0]
    joint_dist = sys.link.joint.pos - 0.5 * cur_len * joint_dir
    joint_transform = 0.5 * lengths[:, None] * joint_dir + joint_dist
    sys2 = set_sys(sys2, {'link.joint.pos': joint_transform})

    # get the new link transform
    parent_idx = jp.array([sys.link_parents])
    sys2 = set_sys(
        sys2,
        {
            'link.transform.pos': -(
                    sys2.link.joint.pos
                    + joint_dist
                    + 0.5 * lengths[parent_idx].T * joint_dir
            )
        },
    )
    return sys2

# END UTILS

def util_vmap_set(sys, keys, vals):
    dico = dict(zip(keys, vals))

    return set_sys(sys, dico)

def randomize(sys, rng):
    return set_sys(sys, {'link.inertia.mass': sys.link.inertia.mass + jax.random.uniform(rng, shape=(sys.num_links(),))})


@jax.jit
def randomize_sys_capsules(
        rng: jp.ndarray,
        sys: base.System,
        min_length: float = 0.0,
        max_length: float = 0.0,
        min_radius: float = 0.0,
        max_radius: float = 0.0,
):
    """Randomizes joint offsets, assume capsule geoms appear in geoms[1]."""
    rng, key1, key2 = jax.random.split(rng, 3)
    length_u = jax.random.uniform(
        key1, shape=(sys.num_links(),), minval=min_length, maxval=max_length
    )
    radius_u = jax.random.uniform(
        key2, shape=(sys.num_links(),), minval=min_radius, maxval=max_radius
    )
    length = length_u + sys.geoms[1].length  # pytype: disable=attribute-error
    radius = radius_u + sys.geoms[1].radius  # pytype: disable=attribute-error
    return set_sys_capsules(sys, length, radius)

class DomainRandWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, seed):
        super().__init__(env)

        self.rng = jax.random.PRNGKey(seed)

        self.sys = env.unwrapped.sys

        self.num_envs = env.unwrapped.num_envs

        self.obs_size = self.env.observation_size

        #sys_v = jax.vmap(functools.partial(randomize, sys=env.unwrapped.sys))(rng=key)
        #env.sys = sys_v
        #self.generic_set()



        self.step_counters = jp.zeros(self.num_envs)

    @property
    def observation_size(self) -> int:
        return self.obs_size

    def generic_set(self, values=None):
        if values is None:
            self.rng, key = jax.random.split(self.rng, 2)
            #key = jp.reshape(jp.stack(key), (self.num_envs, 2))

            values = self.sys.link.inertia.mass + jax.random.uniform(key, shape=(self.sys.num_links(),))

        new_sys = util_vmap_set(sys=self.sys, keys=["link.inertia.mass"], vals=[values])
        self.env.unwrapped.sys = new_sys
        i=0

    def capsule_set(self):
        pass # todo

    def reset(self, rng):
        self.generic_set(values=None)
        return self.env.reset(rng)

    def step(self, state, action):
        x = self.env.step(state, action)

        #self.step_counters = self.step_counters + 1
        #self.step_counters = self.step_counters * (1-x.done)



        return self.step(state, action)

BATCH_SIZE = 4095

rng = jax.random.PRNGKey(0)
initial_sys = mjcf.load(
    "/home/charlie/Desktop/mb-rma/temp/external/brax/brax/envs/assets/half_cheetah.xml"
)

rng, key = jax.random.split(rng, 2)
# key = jp.reshape(jp.stack(key), (self.num_envs, 2))

values = initial_sys.link.inertia.mass + jax.random.uniform(key, shape=(BATCH_SIZE, initial_sys.num_links(),))
