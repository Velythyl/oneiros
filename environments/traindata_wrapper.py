from gym import Wrapper
import brax.io.torch
import jax
import torch

class _DataBuf:
    def __init__(self, key):
        self.key = key
        self.list = []

    def add(self, val):
        if isinstance(val, jax.Array):
            val = brax.io.torch.jax_to_torch(val)
        assert isinstance(val, torch.Tensor)
        self.list.append(val)

    def output(self):
        if len(self.list) > 0:
            return torch.hstack(self.list)
        return None

    def reset(self):
        self.list = []

class DataBuf:
    def __init__(self, keys):
        self.keys = []
        for key in keys:
            self.register(key)

    def register(self, key):
        assert key not in self.keys
        setattr(self, key, _DataBuf(key))
        self.keys.append(key)

    def add(self, dict_or_keys, vals=None):
        if isinstance(dict_or_keys, dict):
            assert vals is None
            vals = dict_or_keys.values()
            dict_or_keys = dict_or_keys.keys()

        for key, val in zip(dict_or_keys, vals):
            getattr(self, key).add(val)

    def reset(self):
        for key in self.keys:
            getattr(self, key).reset()

    def output(self):
        ret = {}
        for key in self.keys:
            ret[key] = getattr(self, key).output()
        return ret

class TraindataWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.buf = DataBuf(["next_obs", "reward", "done"])
        self.old_obs = None

    def _register_traindata_buf(self, key):
        self.buf.register(key)

    def _add_traindata_data(self, dico):
        self.buf.add(dico)

    def _get_traindata(self):
        ret = self.buf.output()
        self.buf.reset()
        return ret
