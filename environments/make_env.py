import functools
from typing import Any, Tuple

from flax import struct
import brax # noqa
from brax.envs.wrappers.torch import TorchWrapper   # noqa
from gymnasium import Wrapper
from gymnasium.vector import AsyncVectorEnv

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
from environments.wrappers.mappings.vector_index_rearrange import VectorIndexMapWrapper
from src.utils.eval import evaluate
from src.utils.every_n import EveryN2
from src.utils.record import record


@monad_coerce
def make_brax(brax_cfg):
    if not cfg_envkey_startswith(brax_cfg, "brax"):
        return None

    BACKEND = envkey_multiplex(brax_cfg).split("-")[0].replace("brax", "")
    ENVNAME = envkey_multiplex(brax_cfg).split("-")[1]

    env = brax.envs.create(env_name=ENVNAME, episode_length=brax_cfg.max_episode_length, backend=BACKEND,
                           batch_size=brax_cfg.num_env) # EP LEN, NUM_ENV
    env = VectorGymWrapper(env, seed=0)  # todo
    env = TorchWrapper(env, device=brax_cfg.device)

    print(f"Brax env built: {envkey_multiplex(brax_cfg)}")

    return env


@monad_coerce
def make_mujoco(mujoco_cfg):
    if not cfg_envkey_startswith(mujoco_cfg, "mujoco"):
        return None

    import gymnasium.wrappers as gym_wrap
    import gymnasium

    BRAX_ENVNAME = envkey_multiplex(mujoco_cfg).split("-")[-1]

    MUJOCO_ENVNAME = {
        "ant": "Ant-v4"
    }[BRAX_ENVNAME]

    def thunk():
        env = gymnasium.make(MUJOCO_ENVNAME, max_episode_steps=mujoco_cfg.max_episode_length, autoreset=True)
        env = VectorIndexMapWrapper(env, BRAX_ENVNAME)
        return env

    env = AsyncVectorEnv([thunk for _ in range(mujoco_cfg.num_env)])

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
        base_envs += make_brax(sliced_multiplex)
        base_envs += make_mujoco(sliced_multiplex)

    base_envs = list(filter(lambda x: x is not None, base_envs))
    assert len(base_envs) == num_multiplex(multiplex_env_cfg)

    def single_action_space(env):
        return (env.action_space.shape[-1],)

    def single_observation_space(env):
        return (env.observation_space.shape[-1],)

    def num_envs(env):
        assert env.observation_space.shape[0] == env.action_space.shape[0]
        return env.observation_space.shape[0]

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
    print("...done!")

    print("Building eval envs...")
    many_eval_env = []
    for i, sliced_multiplex in enumerate(splat_multiplex(multienv_cfg.eval)):
        sliced_multiplex = monad_multiplex(sliced_multiplex)
        many_eval_env += [make_multiplex(sliced_multiplex, seed+i+1)]

        assert many_eval_env[-1].ONEIROS_METADATA.single_action_space == train_env.ONEIROS_METADATA.single_action_space
        assert many_eval_env[-1].ONEIROS_METADATA.single_observation_space == train_env.ONEIROS_METADATA.single_observation_space
    print("...done!")

    EVAL_FREQ = multienv_cfg.eval_freq
    if EVAL_FREQ and EVAL_FREQ != "None":
        eval_envs = many_eval_env
    else:
        eval_envs = []

    VIDEO_FREQ = multienv_cfg.video_freq
    if VIDEO_FREQ and VIDEO_FREQ != "None":
        video_envs = many_eval_env
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

    return train_env, all_hooks
