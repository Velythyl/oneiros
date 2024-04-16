import functools
import gc
from typing import Any, Tuple

from flax import struct
import brax # noqa
from brax.envs.wrappers.torch import TorchWrapper   # noqa
from gymnasium import Wrapper
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm

from environments.config_utils import envkey_multiplex, num_multiplex, slice_multiplex, monad_multiplex, \
    splat_multiplex, marshall_multienv_cfg, cfg_envkey_startswith
from environments.env_binding import get_envstack, traverse_envstack, bind
from environments.func_utils import monad_coerce
from environments.wrappers.infologwrap import InfoLogWrap
from environments.wrappers.jax_wrappers.gym import VectorGymWrapper
from environments.wrappers.multiplex import MultiPlexEnv
from environments.wrappers.np2torch import Np2TorchWrapper
from environments.wrappers.recordepisodestatisticstorch import RecordEpisodeStatisticsTorch
from environments.wrappers.renderwrap import RenderWrap
from environments.wrappers.mappings.vector_index_rearrange import VectorIndexMapWrapper, map_func_lookup, _MujocoMapping
from environments.wrappers.sim2real.last_act import LastActEnv
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

    env = brax.envs.create(env_name=ENVNAME, episode_length=brax_cfg.max_episode_length, backend=BACKEND,
                           batch_size=brax_cfg.num_env) # EP LEN, NUM_ENV
    env = VectorGymWrapper(env, seed=seed)
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

    def thunk():
        env = gymnasium.make(MUJOCO_ENVNAME, max_episode_steps=mujoco_cfg.max_episode_length, autoreset=True)
        env = SeededEnv(env)
        env = VectorIndexMapWrapper(env, map_func_lookup(_MujocoMapping, BRAX_ENVNAME))
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
        return (env.action_space.shape[-1],)

    def single_observation_space(env):
        return (env.observation_space.shape[-1],)

    def num_envs(env):
        assert env.observation_space.shape[0] == env.action_space.shape[0]
        return env.observation_space.shape[0]

    for i, env in enumerate(base_envs):
        if slice_multiplex(multiplex_env_cfg, i).framestack > 1:
            env = VecFrameStackEnv(env, device=multiplex_env_cfg.device[0],
                                   num_stack=slice_multiplex(multiplex_env_cfg, i).framestack)

        if slice_multiplex(multiplex_env_cfg, i).last_action is not False:
            env = LastActEnv(env, device=multiplex_env_cfg.device[0])

        base_envs[i] = env

    PROTO_ACT = single_action_space(base_envs[0])
    PROTO_OBS = single_observation_space(base_envs[0])
    PROTO_NUM_ENV = num_envs(base_envs[0])

    def metadata_maker(cfg, num_env):
        return ONEIROS_METADATA(cfg, PROTO_ACT, PROTO_OBS, (num_env, *PROTO_ACT), (num_env, *PROTO_OBS))


    for i, env in enumerate(base_envs):
        env = RenderWrap(env)
        env = RecordEpisodeStatisticsTorch(env, device=multiplex_env_cfg.device[0], num_envs=slice_multiplex(multiplex_env_cfg, i).num_env)
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

    print("Building training envs...")
    train_env = make_multiplex(multienv_cfg.train, seed)
    gc.collect()
    print("...done!")

    print("Building eval envs...")
    eval_and_video_envs = []
    for i, sliced_multiplex in enumerate(splat_multiplex(multienv_cfg.eval)):
        sliced_multiplex = monad_multiplex(sliced_multiplex)
        eval_and_video_envs += [make_multiplex(sliced_multiplex, seed+i+1)]

        assert eval_and_video_envs[-1].ONEIROS_METADATA.single_action_space == train_env.ONEIROS_METADATA.single_action_space
        assert eval_and_video_envs[-1].ONEIROS_METADATA.single_observation_space == train_env.ONEIROS_METADATA.single_observation_space
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
        hooks.append(functools.partial(record, video_envs=_env, RUN_DIR=save_path, NUM_STEPS=multienv_cfg.num_video_steps))
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
