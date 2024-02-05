import collections
import functools
from typing import List, Dict

import brax
import jax
import torch
from brax import State
from brax.envs import Env
from flax import struct
from gym import Wrapper
from brax.io.torch import torch_to_jax
from jax import numpy as jp

from environments.wrappers.jax_wrappers.gym import VectorGymWrapper


def debug_jax_arr(arr):
    return brax.io.torch.jax_to_torch(arr).cpu().numpy()


from gym import Wrapper
import brax.io.torch
import jax
import torch




class VectorBuf:
    def __init__(self, prototype):
        self.prototype = prototype
        self.data = prototype
        self.indices = jp.zeros(prototype.shape[0], dtype=jp.int32)
        self.num_stack = prototype.shape[1]

        def roll_add(old_buf, xi, index):
            def add_direct(old_buf, xi, index):
                return old_buf.at[index].set(xi)

            def add_rolled(old_buf, xi, index):
                old_buf = jp.roll(old_buf, -1, 0)
                old_buf = old_buf.at[-1].set(xi)
                return old_buf

            new_buf = jax.lax.cond(
                index < self.num_stack,
                add_direct,
                add_rolled,
                old_buf, xi, index
            )
            return new_buf, jax.lax.clamp(0, index + 1, self.num_stack).astype('int32')
        self.roll_add = jax.jit(jax.vmap(roll_add))

    def add(self, element):
        self.data, self.indices = self.roll_add(self.data, element, self.indices)

    def reset(self, done_mask):
        done_mask = done_mask.astype('bool')
        self.data = self.data.at[done_mask].set(self.prototype[done_mask])
        self.indices = (self.indices * (1 - done_mask)).astype('int32')

@struct.dataclass
class Stacks:
    action: jp.array
    obs: jp.array
    flat_state: List[jp.array]
    extra: Dict

class VectorGymWrapper_FrameActionStack(Wrapper):
    def __init__(self, env, num_envs: int, num_stack: int):
        super().__init__(env)
        assert isinstance(env, VectorGymWrapper)

        self.num_envs = num_envs
        self.num_stack = num_stack

        self.last_state = None
        self.last_obs = None

        # INIT ACTIONS
        action_size = self.env.action_space.shape[-1]
        action_prototype = jp.zeros((self.num_envs, self.num_stack, action_size))
        self.action_buf = VectorBuf(action_prototype)

        def identity(s, _):
            return s

        def stackit(s):
            return jax.vmap(functools.partial(identity, s))(jp.arange(self.num_stack))


        # INIT OBS
        obs_prototype = env.reset()
        obs_prototype = stackit(obs_prototype).reshape((num_envs, num_stack, -1))
        self.obs_buf = VectorBuf(obs_prototype)

        sample_pstate = self.get_hidden_state()
        state_prototype = jax.tree_util.tree_flatten(jax.vmap(stackit)(sample_pstate))[
            0]
        self.state_bufs = []
        for state_elem in state_prototype:
            self.state_bufs.append(VectorBuf(state_elem))

        self.extra_stacks = {}

    def _get_stacks(self):
        return Stacks(action=self.action_buf.data, obs=self.obs_buf.data, flat_state=[b.data for b in self.state_bufs], extra=self.extra_stacks)

    def _add_stack(self, key, prototype):
        buf = VectorBuf(prototype)
        self.extra_stacks[key] = buf
        setattr(self, key, buf)

    def _add_to_stack(self, key, data):
        getattr(self, key).add(data)

    def get_state_stacks(self):
        return self._remake_p_states(self.flat_p_stacks)

    def extract_current_skrs_vals(self):
        return self.get_state().info["skrs_vals"]

    def extract_current_skrs_resampled(self):
        return self.get_state().info["skrs_resampled"]

    def get_hidden_state(self):
        return self.get_state().pipeline_state

    def get_state(self):
        return self.env._state

    def stacks_add(self, action, obs, state):
        self.action_buf.add(action)
        self.obs_buf.add(obs)
        for b, s in zip(self.state_bufs, jax.tree_util.tree_flatten(state)[0]):
            b.add(s)

    def stacks_reset(self, mask: jp.ndarray):
        self.action_buf.reset(mask)
        self.obs_buf.reset(mask)
        for b in self.state_bufs:
            b.reset(mask)

    def reset(self, **kwargs):
        ret = super(VectorGymWrapper_FrameActionStack, self).reset(**kwargs)
        self.last_state = self.get_hidden_state()
        self.last_obs = ret
        self.stacks_reset(mask=jp.ones((self.num_envs,)))
        return ret

    def step(self, action: jp.ndarray) -> State:
        ret = super(VectorGymWrapper_FrameActionStack, self).step(action)

        self.stacks_add(action, self.last_obs, self.last_state)
        self.last_state = self.get_hidden_state()
        self.last_obs = ret[0]
        self.stacks_reset(ret[-2])

        return ret
