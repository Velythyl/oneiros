import copy
import functools
import os
import shutil
import time
from typing import Dict

import brax as brax
import gymnasium
import jax
import jax.numpy as jp
import numpy as np
import brax.envs


if __name__ == "__main__":

    try:
        os.remove("mujoco-3.1.1-linux-x86_64.tar.gz")
    except:
        pass
    exit()

    mujoco = gymnasium.make("Ant-v4", max_episode_steps=1000, autoreset=True)
    mujoco_obs = mujoco.reset()

    brax = brax.envs.create("ant", backend="spring")
    brax_obs = mujoco.reset()

    stop = None

    def eval(func = lambda x: x):
        baselines = []

        for _ in range(1000):
            np.random.seed(1)
            env = gymnasium.make("Ant-v4", max_episode_steps=1000, autoreset=True)
            x = env.reset(seed=1)
            env = func(env)
            for i in range(10):
                x = env.step(action=env.action_space.sample() * 0)[0]
            baselines.append(x)

        baselines = np.vstack(baselines)
        baselines_mean = baselines.mean(axis=0)
        baselines_std = baselines.std(axis=0)
        return baselines_mean[5], baselines_std[5]

    print(eval())
    def perturb(env):
        env.model.body_mass = env.model.body_mass * 10000
        #for i in range(len(env.model.body_mass)):
        #    env.model.body_mass[i] = env.model.body_mass[i] * 100
        return env
    print(eval(perturb))
    def perturb(env):
        for i in range(len(env.model.body_mass)):
            env.model.body_mass[i] = env.model.body_mass[i] * 1000
        return env
    print(eval(perturb))
    def perturb(env):
        for i in range(len(env.model.body_mass)):
            env.model.body_mass[i] = env.model.body_mass[i] * 10000
        return env
    print(eval(perturb))
    #def perturb(env):
    #    env.model.body_mass = env.model.body_mass * 0.001
    #    return env
    #print(eval(perturb))

