# Copyright 2024 The Brax Authors.
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

# pylint:disable=g-multiple-import
"""Trains an ant to run in the +x direction."""
import io
from typing import Optional, List

import brax
import cv2
from PIL import Image
from brax import base, System
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html
from brax.io.image import render, render_array
from etils import epath
import jax
from imageio.plugins.swf import HTML
from jax import numpy as jp
import mujoco
from tqdm import tqdm


class Go1(PipelineEnv):
  def __init__(
      self,
      ctrl_cost_weight=0.5,
      use_contact_forces=False,
      contact_cost_weight=5e-4,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.1, 3.0),
      contact_force_range=(-1.0, 1.0),
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs,
  ):
    path = epath.resource_path('environments') / 'customenv/braxcustom/assets/unitree_go1/go1.xml'
    sys = mjcf.load(path)

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 10

    if backend == 'mjx':
      sys = sys.tree_replace({
          'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
          'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
          'opt.iterations': 1,
          'opt.ls_iterations': 4,
      })

    if backend == 'positional':
      # TODO: does the same actuator strength work as in spring
      sys = sys.replace(
          actuator=sys.actuator.replace(
              gear=200 * jp.ones_like(sys.actuator.gear)
          )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')

  def reset(self, sys: System, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    q = sys.init_q + jax.random.uniform(
        rng1, (sys.q_size(),), minval=low, maxval=hi
    )
    qd = hi * jax.random.normal(rng2, (sys.qd_size(),))

    pipeline_state = self.pipeline_init(sys, q, qd)
    obs = self._get_obs(pipeline_state)

    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_survive': zero,
        'reward_ctrl': zero,
        'reward_contact': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'forward_reward': zero,
    }
    return State(pipeline_state, obs, reward , done, sys, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(state.sys, pipeline_state0, action)

    velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
    forward_reward = velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = 0.0

    obs = self._get_obs(pipeline_state)
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe ant body position and velocities."""
    qpos = pipeline_state.q
    qvel = pipeline_state.qd

    #if self._exclude_current_positions_from_observation:
    qpos = pipeline_state.q[2:]

    return jp.concatenate([qpos] + [qvel])


def register(name, clazz):
    brax.envs._envs[name] = clazz

register("go1", Go1)


def render(
    sys: brax.System,
    trajectory: List[base.State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
    fmt: str = 'png',
) -> bytes:
  """Returns an image of a brax System."""
  if not trajectory:
    raise RuntimeError('must have at least one state')

  frames = render_array(sys, trajectory, height, width, camera)

  #out = cv2.VideoWriter("/tmp/go1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[0], frames[0].shape[1]))
  #for frame in frames:
  #    out.write(frame)  # frame is a numpy.ndarray with shape (1280, 720, 3)
  #out.release()
  #return

  frames = [ Image.fromarray(arr) for arr in frames]
  f = io.BytesIO()
  if len(frames) == 1:
    frames[0].save(f, format=fmt)
  else:
    frames[0].save(
        "/tmp/go1.gif",
        format=fmt,
        append_images=frames[1:],
        save_all=True,
        #duration=sys.dt * 100,
        fps=30,
        loop=0)

  return f.getvalue()


if __name__ == "__main__":
    register("go1", Go1)

    env = brax.envs.create(env_name="go1", episode_length=1000, backend="spring",
                           batch_size=1, no_vsys=True)

    state = env.reset(jax.random.PRNGKey(0))
    traj = []
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    for i in tqdm(range(1000)):
        key, rng = jax.random.split(key)
        state = step(state, jax.random.uniform(rng, shape=(1,env.action_size,), minval=-1, maxval=1))
        #print(state.reward)
        traj.append(state.pipeline_state)

    def render_from_state(sys, pipeline_state):
        frame = render_array([pipeline_state])

    frames = render_array(state.sys, traj, camera="tracking")
    frames = [Image.fromarray(arr) for arr in frames]
    frames[0].save(
        "/tmp/go1_brax.gif",
        append_images=frames[1:],
        save_all=True,
        duration=state.sys.dt * 1000,
        #fps=30,
        loop=0)


    #HTML(html.render(env.sys.replace(dt=env.dt), traj))
    #rendered = render(state.sys, traj)


    x=0

