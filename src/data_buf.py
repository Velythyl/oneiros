import brax.io.torch
import jax
import torch

class _DataBuf:
    def __init__(self, key, dim):
        self.key = key
        self.dim = dim
        self.list = []

    def add(self, val):
        if isinstance(val, jax.Array):
            val = brax.io.torch.jax_to_torch(val)
        assert isinstance(val, torch.Tensor)
        self.list.append(val)

    def output(self):
        ret = torch.hstack(self.list)
        self.list = []
        return ret

class DataBuf:
    def __init__(self, keys, dims):
        self.keys = []
        for key, dim in zip(keys, dims):
            self.register(key, dim)

    def register(self, key, dim):
        assert key not in self.keys
        setattr(self, key, _DataBuf(key, dim))
        self.keys.append(key)

    def add(self, keys, vals):
        for key, val in zip(keys, vals):
            getattr(self, key).add(val)

    def output(self):
        ret = {}
        for key in self.keys:
            ret[key] = getattr(self, key).output()
        return ret
