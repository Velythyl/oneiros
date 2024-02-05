import copy
import functools

from flax import struct
import gym
import brax
from brax.envs.wrappers.torch import TorchWrapper

from environments.wrappers.infologwrap import InfoLogWrap
from environments.wrappers.jax_wrappers.gym import VectorGymWrapper
from environments.wrappers.multiplex import MultiPlexEnv
from environments.wrappers.recordepisodestatisticstorch import RecordEpisodeStatisticsTorch
from environments.wrappers.renderwrap import RenderWrap
from environments.wrappers.rundir import rundir
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

    env.ONEIROS_DEVICE = brax_cfg.device

    return env


@monad
def make_mujoco(mujoco_cfg):
    if not startswith(mujoco_cfg, "mujoco"):
        return None


def make_multiplex(multiplex_env_cfg):
    base_envs = []
    for sliced_multiplex in splat_multiplex(multiplex_env_cfg):
        base_envs += make_brax(sliced_multiplex)
        base_envs += make_mujoco(sliced_multiplex)

    base_envs = list(filter(lambda x: x is not None, base_envs))
    assert len(base_envs) == num_multiplex(multiplex_env_cfg)

    def single_action_space(env):
        return env.action_space.shape[-1]

    def single_observation_space(env):
        return env.observation_space.shape[-1]

    PROTO_ACT = single_action_space(base_envs[0])
    PROTO_OBS = single_observation_space(base_envs[0])
    PROTO_DEVICE = base_envs[0].ONEIROS_DEVICE

    for i, env in enumerate(base_envs):
        env = RenderWrap(env)
        env = InfoLogWrap(env, prefix=envkey_multiplex(slice_multiplex(multiplex_env_cfg, i)))

        assert single_action_space(env) == PROTO_ACT
        assert single_observation_space(env) == PROTO_OBS
        assert PROTO_DEVICE == env.ONEIROS_DEVICE

        base_envs[i] = env

    env = MultiPlexEnv(base_envs, PROTO_DEVICE)

    # "hopper"  # @param ['ant', 'fetch', 'grasp', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'pusher', 'reacher', 'walker2d', 'grasp', 'ur5e']

    from environments.wrappers.jax_wrappers.gym import VectorGymWrapper

    env = VectorGymWrapper(env, seed=env_config.seed)
    env = TorchWrapper(env_with_the_stacks, device=DEVICE)
    env = RenderWrap(env)
    env = RecordEpisodeStatisticsTorch(env, DEVICE, num_envs=NUM_ENVS)

    env = InfoLogWrap(env_traindata, prefix=prefix)

    env.action_space = env.action_space
    env.many_action_space = env.action_space
    env.single_action_space = gym.spaces.Box(low=env.action_space.low.min(), high=env.action_space.high.max(),
                                             shape=env.action_space.shape[1:])

    # envs.observation_space = env.observation_space
    env.many_observation_space = env.observation_space
    env.single_observation_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                                  high=env.observation_space.high.max(),
                                                  shape=env.observation_space.shape[1:])
    env.num_envs = NUM_ENVS
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    def assigns(e):
        e.env_name = env_config.env_id
        e.num_envs = NUM_ENVS
        bind(e, traverse_envstack)
        bind(e, get_envstack)
        e.MAX_EPISODE_LENGTH = EPISODE_LENGTH

    traverse_envstack(env, [assigns])
    x = env.reset()

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


def splat_multiplex(multiplex_cfg):
    ret = []
    for i in range(num_multiplex(multiplex_cfg)):
        ret += [slice_multiplex(multiplex_cfg, i)]
    return ret


def make_sim2sim(multienv_cfg):
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

    train_env = make_multiplex(multienv_cfg.train)

    many_eval_env = []
    for sliced_multiplex in splat_multiplex(multienv_cfg.eval):
        many_eval_env += [make_multiplex(sliced_multiplex)]

    DO_EVAL = False

    if DO_EVAL:
        eval_envs = [
            eval_wdr_atstart(),
            eval_wdr_every200()
        ]
    else:
        eval_envs = []

    if DO_EVAL:
        video_envs = [
            eval_wdr_every200()
        ]
    else:
        video_envs = []

    hook_steps = []
    hooks = []
    for _env in video_envs:
        hook_steps.append(25_000_000)
        hooks.append(functools.partial(record, video_envs=_env, RUN_DIR=rundir()))
    for _env in eval_envs:
        hook_steps.append(5_000_000)
        hooks.append(functools.partial(evaluate, eval_envs=_env))
    all_hooks = EveryN2(hook_steps, hooks)

    return train_env, all_hooks
