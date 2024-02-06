import copy
import functools
from typing import Any, Tuple

from flax import struct
import gym
import brax
from brax.envs.wrappers.torch import TorchWrapper

from environments.wrappers.infologwrap import InfoLogWrap
from environments.wrappers.jax_wrappers.gym import VectorGymWrapper
from environments.wrappers.multiplex import MultiPlexEnv
from environments.wrappers.recordepisodestatisticstorch import RecordEpisodeStatisticsTorch
from environments.wrappers.renderwrap import RenderWrap
from src.utils.eval import evaluate
from src.utils.every_n import EveryN2
from src.utils.record import record


# from environments.our_brax import our_brax_loader
# from environments.our_brax.our_brax_loader import register_environment

def get_envstack(_env, aslist=False):
    def iter(env):
        def get_next():
            if hasattr(env, "env"):
                return env.env
            if hasattr(env, "_env"):
                return env._env
            return None

        if not get_next():
            return
        yield get_next()
        yield from iter(get_next())

    if aslist:
        return list(iter(_env))
    else:
        return iter(_env)


def traverse_envstack(env, funcs):
    for e in get_envstack(env):
        for func in funcs:
            func(e)


def bind(e, method):
    # PLEASE only use this with envs or wrappers
    # This is a terrible code practice, but doing
    # "proper, clean" code would require changing
    # about a billion 3rd party dependencies...
    # especially since the whole goal of this lib
    # is to handle a bunch of different
    # simulators
    method_name = method.__name__
    assert not hasattr(e, method_name)
    setattr(e, method.__name__, functools.partial(method, e))


@struct.dataclass
class MakeEnv_Input:
    env_key: str
    num_env: int
    max_episode_length: int
    action_repeat: int
    framestack: int
    device: str

    def build(self, env_key, num_env, sim2sim_cfg):
        return MakeEnv_Input(
            env_key=env_key,
            num_env=num_env,
            max_episode_length=sim2sim_cfg.max_episode_length,
            action_repeat=sim2sim_cfg.action_repeat,
            framestack=sim2sim_cfg.action_repeat,
            device=sim2sim_cfg.device
        )


def make_env(env_cfg):
    pass


def startswith(cfg, name):
    return cfg.env_key.startswith(name)


def monad(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return [f(*args, **kwds)]

    return wrapper


@monad
def make_brax(brax_cfg):
    if not startswith(brax_cfg, "brax"):
        return None

    BACKEND = envkey_multiplex(brax_cfg).split("-")[0].replace("brax", "")
    ENVNAME = envkey_multiplex(brax_cfg).split("-")[1]

    env = brax.envs.create(env_name=ENVNAME, episode_length=brax_cfg.max_episode_length, backend=BACKEND,
                           batch_size=brax_cfg.num_env)
    env = VectorGymWrapper(env, seed=0)  # todo
    env = TorchWrapper(env, device=brax_cfg.device)

    return env


@monad
def make_mujoco(mujoco_cfg):
    if not startswith(mujoco_cfg, "mujoco"):
        return None

@struct.dataclass
class EnvMetaData:
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
        return EnvMetaData(cfg, PROTO_ACT, PROTO_OBS, (num_env, *PROTO_ACT), (num_env, *PROTO_OBS))

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
        traverse_envstack(env, [assigns])
        base_envs[i] = env

    env = MultiPlexEnv(base_envs, multiplex_env_cfg.device[0])

    assert env.observation_space.shape[0] == PROTO_NUM_ENV * len(base_envs)

    env.ONEIROS_METADATA = metadata_maker(multiplex_env_cfg, PROTO_NUM_ENV * len(base_envs))

    return env


def envkey_multiplex(multiplex_cfg):
    return multiplex_cfg.env_key


def num_multiplex(multiplex_cfg):
    return len(multiplex_cfg.env_key)


def keys_multiplex(mutiplex_cfg):
    return vars(mutiplex_cfg)["_content"].keys()


def slice_multiplex(multiplex_cfg, index):
    EVAL_CONFIG = copy.deepcopy(multiplex_cfg)

    for k in keys_multiplex(multiplex_cfg):
        EVAL_CONFIG[k] = multiplex_cfg[k][index]

    return EVAL_CONFIG

def monad_multiplex(multiplex_cfg):
    EVAL_CONFIG = copy.deepcopy(multiplex_cfg)
    for k in keys_multiplex(multiplex_cfg):
        EVAL_CONFIG[k] = [multiplex_cfg[k]]
    return EVAL_CONFIG


def splat_multiplex(multiplex_cfg):
    ret = []
    for i in range(num_multiplex(multiplex_cfg)):
        ret += [slice_multiplex(multiplex_cfg, i)]
    return ret


def make_sim2sim(multienv_cfg, seed: int, save_path: str):
    for key in ["num_env", "max_episode_length", "action_repeat", "framestack"]:
        if multienv_cfg[key] not in ["None", None]:
            multienv_cfg.train[key] = multienv_cfg[key]
            multienv_cfg.eval[key] = multienv_cfg[key]

    def coerce_traineval(train_or_eval_cfg):

        def coerce_possible_list(key):
            val = train_or_eval_cfg[key]

            if isinstance(val, str):
                if "," in val:
                    val = val.split(",")
                    val = list(map(lambda x: x.strip(), val))
                else:
                    val = [val]
            elif isinstance(val, int):
                val = [val]

            VALID = True
            for x in val:
                if isinstance(x, str) or isinstance(x, int):
                    continue
                else:
                    VALID = False
                    break

            if not VALID:
                raise ValueError(f"Input is wrong for key {key}")

            train_or_eval_cfg[key] = val

        counts = {}
        for key in vars(train_or_eval_cfg)["_content"].keys():
            coerce_possible_list(key)
            counts[key] = len(train_or_eval_cfg[key])

        if num_multiplex(train_or_eval_cfg) == 1:
            for key, c in counts.items():
                if c != 1:
                    raise ValueError("When there's only one env in multienv, all other params must also be of len 1")
        else:
            NUM_MULTIENV = num_multiplex(train_or_eval_cfg)
            assert NUM_MULTIENV > 1

            for key, count in counts.items():
                if key == "env_key":
                    continue

                if count == NUM_MULTIENV:
                    continue

                assert count == 1
                train_or_eval_cfg[key] = [*train_or_eval_cfg[key]] * NUM_MULTIENV

    coerce_traineval(multienv_cfg.train)
    coerce_traineval(multienv_cfg.eval)

    train_env = make_multiplex(multienv_cfg.train, seed)

    many_eval_env = []
    for i, sliced_multiplex in enumerate(splat_multiplex(multienv_cfg.eval)):
        sliced_multiplex = monad_multiplex(sliced_multiplex)
        many_eval_env += [make_multiplex(sliced_multiplex, seed+i+1)]

        assert many_eval_env[-1].ONEIROS_METADATA.single_action_space == train_env.ONEIROS_METADATA.single_action_space
        assert many_eval_env[-1].ONEIROS_METADATA.single_observation_space == train_env.ONEIROS_METADATA.single_observation_space

    EVAL_FREQ = multienv_cfg.eval_freq
    if EVAL_FREQ:
        eval_envs = many_eval_env
    else:
        eval_envs = []

    VIDEO_FREQ = multienv_cfg.video_freq
    if VIDEO_FREQ:
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
