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
"""Trains a reacher to reach a target.

Based on the OpenAI Gym MuJoCo Reacher environment.
"""
import functools
from typing import Tuple

from brax import base, System
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp

from environments.customenv.common_utils import random_sphere_jax


class WidowReacher(PipelineEnv):



  # pyformat: disable
  """
  ### Description

  "Reacher" is a two-jointed robot arm. The goal is to move the robot's end
  effector (called *fingertip*) close to a target that is spawned at a random
  position.

  ### Action Space

  The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)`
  represents the torques applied at the hinge joints.

  | Num | Action                                                                          | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1          | 1           | joint0                         | hinge | torque (N m) |
  | 1   | Torque applied at the second hinge (connecting the two links)                   | -1          | 1           | joint1                         | hinge | torque (N m) |

  ### Observation Space

  Observations consist of:

  - The cosine of the angles of the two arms
  - The sine of the angles of the two arms
  - The coordinates of the target
  - The angular velocities of the arms
  - The vector between the target and the reacher's fingertip (3 dimensional
    with the last element being 0)

  The observation is a `ndarray` with shape `(11,)` where the elements
  correspond to the following:

  | Num | Observation                                                                                    | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|------------------------------------------------------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                    | hinge | unitless                 |
  | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                    | hinge | unitless                 |
  | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | cos(joint0)                    | hinge | unitless                 |
  | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | cos(joint1)                    | hinge | unitless                 |
  | 4   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                       | slide | position (m)             |
  | 5   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                       | slide | position (m)             |
  | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                         | hinge | angular velocity (rad/s) |
  | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                         | hinge | angular velocity (rad/s) |
  | 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                             | slide | position (m)             |
  | 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                             | slide | position (m)             |
  | 10  | z-value of position_fingertip - position_target (0 since reacher is 2d and z is same for both) | -Inf | Inf | NA                             | slide | position (m)             |

  ### Rewards

  The reward consists of two parts:

  - *reward_dist*: This reward is a measure of how far the *fingertip*
    of the reacher (the unattached end) is from the target, with a more negative
    value assigned for when the reacher's *fingertip* is further away from the
    target. It is calculated as the negative vector norm of (position of
    the fingertip - position of target), or *-norm("fingertip" - "target")*.
  - *reward_ctrl*: A negative reward for penalising the walker if it takes
    actions that are too large. It is measured as the negative squared
    Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

  Unlike other environments, Reacher does not allow you to specify weights for
  the individual reward terms. However, `info` does contain the keys
  *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms, you
  should create a wrapper that computes the weighted reward from `info`.

  ### Starting State

  All observations start in state (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0) with a noise added for stochasticity. A uniform noise in the range
  [-0.1, 0.1] is added to the positional attributes, while the target position
  is selected uniformly at random in a disk of radius 0.2 around the origin.

  Independent, uniform noise in the range of [-0.005, 0.005] is added to the
  velocities, and the last element ("fingertip" - "target") is calculated at the
  end once everything is set.

  The default setting has a *dt = 0.02*

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder (or by changing
  the path to a modified XML file in another folder)..

  ```
  env = gym.make('Reacher-v2')
  ```

  There is no v3 for Reacher, unlike the robot environments where a v3 and
  beyond take gym.make kwargs such as ctrl_cost_weight, reset_noise_scale etc.

  There is a v4 version that uses the mujoco-bindings

  ```
  env = gym.make('Reacher-v4')
  ```

  And a v5 version that uses Brax:

  ```
  env = gym.make('Reacher-v5')
  ```

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks (not including
    reacher, which has a max_time_steps of 50). Added reward_threshold to
    environments.
  * v0: Initial versions release (1.0.0)
  """
  # pyformat: enable


  def __init__(self, backend='generalized', **kwargs):
    if backend in ['mjx']:
      path = epath.resource_path('environments') / 'customenv/braxcustom/assets/trossen_wx250s/wx250s_boxes_mjx.xml'
      sys = mjcf.load(path)
    elif backend in ["spring"]:
      path = epath.resource_path('environments') / 'customenv/braxcustom/assets/trossen_wx250s/wx250s_boxes_pospring.xml'
      sys = mjcf.load(path)
    elif backend in ['positional']:
      path = epath.resource_path(
          'environments') / 'customenv/braxcustom/assets/trossen_wx250s/wx250s_boxes_pos.xml' # '
      sys = mjcf.load(path)
    elif backend in ['generalized']:
      path = epath.resource_path('environments') / 'customenv/braxcustom/assets/trossen_wx250s/wx250s_boxes_generalized.xml'
      sys = mjcf.load(path)
    else:
      raise AssertionError(f"Backend `{backend}` not found")

    #n_frames = 2
    sys = sys.replace(dt=0.016)
    n_frames = 1

    self.scaling = 1.0
    if backend in ['spring', "positional"]:
      sys = sys.replace(dt=0.002)
      n_frames = 8

    if backend in ['generalized', "mjx"]:
        sys = sys.replace(dt=0.002 * 4)
        n_frames = 2

    #if backend in ['mjx', "generalized"]:  #
    #  sys = sys.replace(dt=0.016 / 2)
    #  n_frames = 2

    kwargs['n_frames'] = n_frames

    self.initial_target = jp.array([0.0, 0.3, 0.3])

    super().__init__(sys=sys, backend=backend, **kwargs)


  def reset(self, sys: System, rng: jax.Array) -> State:
    key, rng = jax.random.split(rng)
    rng, rng1, rng2 = jax.random.split(rng, 3)

    q = sys.init_q + jax.random.uniform(
        rng1, (sys.q_size(),), minval=-0.1, maxval=0.1
    ) * 0
    qd = jax.random.uniform(
        rng2, (sys.qd_size(),), minval=-0.005, maxval=0.005
    ) * 0

    # set the target q, qd
    target = self.initial_target
    q = q.at[-3:].set(target)
    qd = qd.at[-3:].set(0)

    pipeline_state = self.pipeline_init(sys, q, qd)

    obs = self._get_obs( pipeline_state, target)
    reward, done, zero = jp.zeros(3)
    reward = -math.safe_norm(target - self.get_tip_pos(pipeline_state))
    metrics = {
        'reward_dist': reward,
        "last_action": jp.zeros(self.action_size)
    }

    state = State(pipeline_state, obs, reward, done, sys, metrics)

    return state

  def get_tip_pos(self, pipeline_state):
      return pipeline_state.x.pos[-4]

  def get_target_pos(self, state):
      return state.obs[-3:]

  def set_target_body(self, pipeline_state, target):
      q, qd = pipeline_state.q, pipeline_state.qd

      q = q.at[-3:].set(target)
      qd = qd.at[-3:].set(0)

      return pipeline_state.replace(q=q, qd=qd)

  def step(self, state: State, action: jax.Array) -> State:
    pipeline_state = self.pipeline_step(state.sys, state.pipeline_state, action)
    pipeline_state = self.set_target_body(pipeline_state, self.get_target_pos(state))

    # CALC REWARD
    reward_dist = -math.safe_norm(self.get_target_pos(state) - self.get_tip_pos(pipeline_state))
    reward = reward_dist

    state.metrics.update(
        reward_dist=reward_dist,
        last_action=action
    )

    obs = self._get_obs(pipeline_state, self.get_target_pos(state))

    state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    return state

  def _get_obs(self, pipeline_state: base.State, target_pos_raw) -> jax.Array:
    """Returns egocentric observation of target and arm body."""

    self_pos = pipeline_state.q[:-3]
    self_vel = pipeline_state.qd[:-3]
    return jp.concatenate([
        self_pos,
        self_vel,
        target_pos_raw,
    ])


  def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
    #return jp.array([0.3,0,0.3])

    """Returns a target location in a random circle slightly above xy plane."""
    key, rng = jax.random.split(rng)
    point = random_sphere_jax(rng, min_r=0.3, max_r=0.6, shape=(1,))[0]
    #point = point.at[-2].set(0)
    return point.at[-1].set(jp.clip(jp.abs(point[-1]), 0.1, 0.6))


import brax
def register(name, clazz):
    brax.envs._envs[name] = clazz

register("widow", WidowReacher)