import functools
from typing import Union, Callable, List

from environments.wrappers.multiplex import MultiPlexEnv


def get_envstack(_env, aslist=False):
    def iter(env):
        def get_next(env):
            if hasattr(env, "env"):
                return env.env
            if hasattr(env, "_env"):
                return env._env
            return None

        yield env

        if not get_next(env):
            return

        if isinstance(env, MultiPlexEnv):
            for e in env.env_list:
                yield from iter(e)
        else:
            yield from iter(get_next(env))

    if aslist:
        return list(iter(_env))
    else:
        return iter(_env)


def traverse_envstack(env, funcs: Union[Callable, List[Callable]]):
    if not isinstance(funcs, list):
        funcs = [funcs]

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
