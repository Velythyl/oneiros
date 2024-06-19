# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrappers to convert brax envs to gym envs."""
import functools
from typing import ClassVar, Optional

from brax.envs.base import PipelineEnv
from brax.io import image
import gym
from gym import spaces
from gym.vector import utils
import jax
import numpy as np

from environments.wrappers.jax_wrappers.utils import take0
from environments.wrappers.jax_wrappers.vectorgym import VectorGymWrapper


class RenderWrap(gym.Wrapper):
    """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env):

        assert isinstance(env, VectorGymWrapper)
        super().__init__(env)
        self.env = env

    def render(self, mode='human'):
        if mode == 'rgb_array':
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError('must call reset or step before rendering')

            state_for_rendering = take0(state.pipeline_state)
            return image.render_array(sys, state_for_rendering, 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception

    def close(self):
        return


