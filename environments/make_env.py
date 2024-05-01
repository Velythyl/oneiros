import dataclasses
import functools
import gc
from typing import Any, Tuple

import jax.random
from flax import struct
import brax
from brax.envs.wrappers.torch import TorchWrapper
from gymnasium import Wrapper
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm

from environments.config_utils import envkey_multiplex, num_multiplex, slice_multiplex, monad_multiplex, \
    splat_multiplex, marshall_multienv_cfg, cfg_envkey_startswith, build_dr_dataclass
from environments.env_binding import get_envstack, traverse_envstack, bind
from environments.func_utils import monad_coerce
from environments.wrappers.infologwrap import InfoLogWrap
from environments.wrappers.jax_wrappers.domain_randomization import DomainRandWrapper, WritePrivilegedInformationWrapper
from environments.wrappers.jax_wrappers.vectorgym import VectorGymWrapper
from environments.wrappers.mujoco.domain_randomization import MujocoDomainRandomization
from environments.wrappers.multiplex import MultiPlexEnv
from environments.wrappers.np2torch import Np2TorchWrapper
from environments.wrappers.recordepisodestatisticstorch import RecordEpisodeStatisticsTorch
from environments.wrappers.renderwrap import RenderWrap
from environments.wrappers.mappings.vector_index_rearrange import VectorIndexMapWrapper, map_func_lookup, _MujocoMapping
from environments.wrappers.sim2real.last_act import LastActEnv
from environments.wrappers.sim2real.matrix_framestack import MatFrameStackEnv
from environments.wrappers.sim2real.vector_framestack import VecFrameStackEnv
from src.utils.eval import evaluate
from src.utils.every_n import EveryN2
from src.utils.record import record


@monad_coerce
def make_brax(brax_cfg, seed):
    if not cfg_envkey_startswith(brax_cfg, "brax"):
        return None

    BACKEND = envkey_multiplex(brax_cfg).split("-")[0].replace("brax", "")
    ENVNAME = envkey_multiplex(brax_cfg).split("-")[1]

    dr_config = build_dr_dataclass(brax_cfg)
    env = brax.envs.create(env_name=ENVNAME, episode_length=brax_cfg.max_episode_length, backend=BACKEND,
                           batch_size=brax_cfg.num_env, no_vsys=not dr_config.DO_DR)

    if isinstance(dr_config.do_on_N_step, tuple):
        val = dr_config.do_on_N_step
        def sample_num(rng):
            return jax.random.uniform(rng, minval=val[0], maxval=val[1])
        dr_config = dataclasses.replace(dr_config, do_on_N_step=sample_num)

    if dr_config.DO_DR:
        env = DomainRandWrapper(env,
                                percent_below=dr_config.percent_below,
                                percent_above=dr_config.percent_above,
                                do_on_reset=dr_config.do_on_reset,
                                do_on_N_step=dr_config.do_on_N_step,
                                do_at_creation=dr_config.do_at_creation,
                                seed=seed + 2
                                )
    env = VectorGymWrapper(env, seed=seed)
    env = WritePrivilegedInformationWrapper(env)
    env = TorchWrapper(env, device=brax_cfg.device)

    print(f"Brax env built: {envkey_multiplex(brax_cfg)}")

    return env





@monad_coerce
def make_mujoco(mujoco_cfg, seed):
    if not cfg_envkey_startswith(mujoco_cfg, "mujoco"):
        return None

    import gymnasium.wrappers as gym_wrap
    import gymnasium

    BRAX_ENVNAME = envkey_multiplex(mujoco_cfg).split("-")[-1]

    MUJOCO_ENVNAME = {
        "ant": "Ant-v4"
    }[BRAX_ENVNAME]

    class WritePrivilegedInformationWrapper(Wrapper):
        def step(self, action):
            ret = super(WritePrivilegedInformationWrapper, self).step(action)
            ret[-1]["priv_info"] = self.unwrapped.model.body_mass
            return ret

    class SeededEnv(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._seed = seed

        def reset(self, **kwargs):
            ret = super(SeededEnv, self).reset(seed=self._seed)
            import numpy as np
            new_seed = np.random.randint(0, 20000)
            np.random.seed(new_seed)
            self._seed = new_seed
            return ret

    dr_config = build_dr_dataclass(mujoco_cfg)

    def thunk():
        env = gymnasium.make(MUJOCO_ENVNAME, max_episode_steps=mujoco_cfg.max_episode_length, autoreset=True)
        env = SeededEnv(env)
        env = VectorIndexMapWrapper(env, map_func_lookup(_MujocoMapping, BRAX_ENVNAME))

        if dr_config.DO_DR:
            env = MujocoDomainRandomization(env,
                                            percent_below=dr_config.percent_below,
                                            percent_above=dr_config.percent_above,
                                            do_on_reset=dr_config.do_on_reset,
                                            do_at_creation=dr_config.do_at_creation,
                                            do_on_N_step=dr_config.do_on_N_step
                                            )

        env = WritePrivilegedInformationWrapper(env)

        return env

    print("Pre async")

    env = AsyncVectorEnv([thunk for _ in range(mujoco_cfg.num_env)], shared_memory=True, copy=False, context="fork")

    print("Post async")

    class AsyncVectorEnvActuallyCloseWrapper(Wrapper):
        def close(self):
            return self.env.close(terminate=True)

    env = AsyncVectorEnvActuallyCloseWrapper(env)

    env = gym_wrap.StepAPICompatibility(env, output_truncation_bool=False)

    class NoResetInfoWrapper(Wrapper):
        def reset(self, **kwargs):
            return super(NoResetInfoWrapper, self).reset(**kwargs)[0]

    env = NoResetInfoWrapper(env)
    env = Np2TorchWrapper(env, mujoco_cfg.device)

    print(f"Mujoco env built: {envkey_multiplex(mujoco_cfg)}")

    return env


@struct.dataclass
class ONEIROS_METADATA:
    cfg: Any

    single_action_space: Tuple
    single_observation_space: Tuple

    multi_action_space: Tuple
    multi_observation_space: Tuple

    @property
    def env_key(self):
        return self.cfg.env_key


def make_multiplex(multiplex_env_cfg, seed):
    base_envs = []
    for sliced_multiplex in splat_multiplex(multiplex_env_cfg):
        base_envs += make_brax(sliced_multiplex, seed)
        base_envs += make_mujoco(sliced_multiplex, seed)

    base_envs = list(filter(lambda x: x is not None, base_envs))
    assert len(base_envs) == num_multiplex(multiplex_env_cfg)

    def single_action_space(env):
        return env.action_space.shape[1:]

    def single_observation_space(env):
        return env.observation_space.shape[1:]

    def num_envs(env):
        assert env.observation_space.shape[0] == env.action_space.shape[0]
        return env.observation_space.shape[0]

    DO_FRAMESTACK = slice_multiplex(multiplex_env_cfg, 0).framestack > 1
    DO_MAT_FRAMESTACK = slice_multiplex(multiplex_env_cfg, 0).mat_framestack_instead
    LAST_ACTION = slice_multiplex(multiplex_env_cfg, 0).last_action

    for i, env in enumerate(base_envs):
        if DO_FRAMESTACK and not DO_MAT_FRAMESTACK:
            env = VecFrameStackEnv(env, device=multiplex_env_cfg.device[0],
                                   num_stack=slice_multiplex(multiplex_env_cfg, i).framestack)
            if LAST_ACTION:
                env = LastActEnv(env, device=multiplex_env_cfg.device[0])

        elif DO_FRAMESTACK and DO_MAT_FRAMESTACK:
            if LAST_ACTION:
                env = LastActEnv(env, device=multiplex_env_cfg.device[0])
            env = MatFrameStackEnv(env, device=multiplex_env_cfg.device[0],
                                   num_stack=slice_multiplex(multiplex_env_cfg, i).framestack)
        else:
            if LAST_ACTION:
                env = LastActEnv(env, device=multiplex_env_cfg.device[0])

        base_envs[i] = env

    PROTO_ACT = single_action_space(base_envs[0])
    PROTO_OBS = single_observation_space(base_envs[0])
    PROTO_NUM_ENV = num_envs(base_envs[0])


    def metadata_maker(cfg, num_env):
        return ONEIROS_METADATA(cfg, PROTO_ACT, PROTO_OBS, (num_env, *PROTO_ACT), (num_env, *PROTO_OBS))

    for i, env in enumerate(base_envs):
        env = RenderWrap(env)
        env = RecordEpisodeStatisticsTorch(env, device=multiplex_env_cfg.device[0],
                                           num_envs=slice_multiplex(multiplex_env_cfg, i).num_env)
        # TODO one prefix for each DR type...
        env = InfoLogWrap(env, prefix=envkey_multiplex(slice_multiplex(multiplex_env_cfg, i)))

        assert single_action_space(env) == PROTO_ACT
        assert single_observation_space(env) == PROTO_OBS
        assert num_envs(env) == PROTO_NUM_ENV

        def assigns(e):
            e.ONEIROS_METADATA = metadata_maker(slice_multiplex(multiplex_env_cfg, i), PROTO_NUM_ENV)
            bind(e, traverse_envstack)
            bind(e, get_envstack)

        traverse_envstack(env, assigns)
        base_envs[i] = env

    env = MultiPlexEnv(base_envs, multiplex_env_cfg.device[0])

    assert env.observation_space.shape[0] == PROTO_NUM_ENV * len(base_envs)

    env.ONEIROS_METADATA = metadata_maker(multiplex_env_cfg, PROTO_NUM_ENV * len(base_envs))

    return env


def make_sim2sim(multienv_cfg, seed: int, save_path: str):
    multienv_cfg = marshall_multienv_cfg(multienv_cfg)

    DOING_POWERSET = multienv_cfg.do_powerset

    print("Building training envs...")
    train_env = make_multiplex(multienv_cfg.train, seed)
    gc.collect()
    print("...done!")

    print("Building eval envs...")
    eval_and_video_envs = []
    for i, sliced_multiplex in enumerate(splat_multiplex(multienv_cfg.eval)):
        sliced_multiplex = monad_multiplex(sliced_multiplex)
        eval_and_video_envs += [make_multiplex(sliced_multiplex, seed + i + 1)]

        assert eval_and_video_envs[
                   -1].ONEIROS_METADATA.single_action_space == train_env.ONEIROS_METADATA.single_action_space
        assert eval_and_video_envs[
                   -1].ONEIROS_METADATA.single_observation_space == train_env.ONEIROS_METADATA.single_observation_space
        gc.collect()
    print("...done!")

    EVAL_FREQ = multienv_cfg.eval_freq
    if EVAL_FREQ and EVAL_FREQ != "None":
        eval_envs = eval_and_video_envs
    else:
        eval_envs = []

    VIDEO_FREQ = multienv_cfg.video_freq
    if VIDEO_FREQ and VIDEO_FREQ != "None":
        video_envs = eval_and_video_envs
    else:
        video_envs = []

    hook_steps = []
    hooks = []
    for _env in video_envs:
        hook_steps.append(VIDEO_FREQ)
        hooks.append(
            functools.partial(record, video_envs=_env, RUN_DIR=save_path, NUM_STEPS=multienv_cfg.num_video_steps))
    for _env in eval_envs:
        hook_steps.append(EVAL_FREQ)
        hooks.append(functools.partial(evaluate, eval_envs=_env, NUM_STEPS=multienv_cfg.num_eval_steps))
    all_hooks = EveryN2(hook_steps, hooks)

    def close_all_envs():
        def kill_asyncvectorenvs(e):
            if isinstance(e, AsyncVectorEnv):
                print("Pre-emptively killing processes in AsyncVectorEnv...")
                for process in tqdm(e.processes):
                    import signal
                    import os
                    os.kill(process.pid, signal.SIGKILL)
                print("...done!")

        def find_multiplex(e):
            if isinstance(e, MultiPlexEnv):
                for e in e.env_list:
                    traverse_envstack(e, kill_asyncvectorenvs)

        print("Traversing train envs looking for AsyncVectorEnvs...")
        traverse_envstack(train_env, [print, find_multiplex])
        print("...done!")

        for env in eval_and_video_envs:
            print(f"Traversing eval env {env.ONEIROS_METADATA.env_key} looking for AsyncVectorEnvs...")
            traverse_envstack(env, [print, kill_asyncvectorenvs])
            print("...done!")

        print("Closing train env...")
        train_env.close()
        print("...done!")

        print("Closing eval env...")
        for env in eval_and_video_envs:
            print(f"closing: {env.ONEIROS_METADATA.env_key}")
            env.close()
        print("...done!")

    return train_env, all_hooks, close_all_envs
