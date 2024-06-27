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




class VectorGymWrapper(gym.vector.VectorEnv):
    """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self,
                 env: PipelineEnv,
                 seed: int = 0,
                 backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 1 / self._env.dt
        }
        if not hasattr(self._env, 'batch_size'):
            raise ValueError('underlying env must be batched')

        self.num_envs = self._env.batch_size
        self.seed(seed)
        self.backend = backend
        self._state = None

        obs = np.inf * np.ones(self._env.observation_size, dtype='float32')
        obs_space = spaces.Box(-obs, obs, dtype='float32')
        self.observation_space = utils.batch_space(obs_space, self.num_envs)

        action = np.ones(self._env.action_size, dtype='float32')
        action_space = spaces.Box(-action, action, dtype='float32')
        self.action_space = utils.batch_space(action_space, self.num_envs)

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

        # find which camera works
        self._cameras = ["track", "tackcom", "tracking", -1]
        self._camera_index = 0
        self._camera_found = False

        self.reset()
        works = []
        for i, cam in enumerate(self._cameras):
            try:
                self._camera_index = i
                self.render("rgb_array")
                works.append(True)
            except:
                works.append(False)
        self._camera_index = np.argmax(np.array(works))
        self._camera_found = True

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        return obs

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError('must call reset or step before rendering')

            state_for_rendering = take0(state.pipeline_state)
            frame = image.render_array(sys, state_for_rendering, 480, 480, camera=self._cameras[self._camera_index])

            return frame
        else:
            return super().render(mode=mode)  # just raise an exception

    def close(self):
        return


