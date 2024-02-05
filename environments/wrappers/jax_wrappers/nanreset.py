from typing import Dict, Optional

from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp

class NaNResetWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, penalty=-10):
        super().__init__(env)
        self.penalty = penalty

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)

        return state
        return state.replace(done=done)
