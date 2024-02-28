import functools
from typing import Iterable


def monad_coerce(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):

        ret = f(*args, **kwds)

        if isinstance(ret, list) or isinstance(ret, Iterable):
            pass
        else:
            ret = [ret]

        return ret

    return wrapper
