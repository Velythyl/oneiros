import brax.io.torch
import jax
import numpy as np
import torch


class TinyLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.list_logs = {}
        self.special_logs = {}

    def _add(self, key, val):

        # test if is a numeric type
        def handle_numeric(val):
            if isinstance(val, jax.Array):
                val = brax.io.torch.jax_to_torch(val)
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()
                val = val.numpy()
            if isinstance(val, float) or isinstance(val, int):
                val = np.array([val])
            if val.shape == tuple():
                val = val[None]
            return val

        try:
            val = handle_numeric(val)
            if key in self.list_logs:
                self.list_logs[key].append(val)
            else:
                self.list_logs[key] = [val]
        except:
            if key in self.special_logs:
                raise Exception(f"key {key} arleady found in TinyLogger's special logs")
            self.special_logs[key] = val

    def log(self, log_dict):
        for key, val in log_dict.items():
            self._add(key, val)

    def info(self, done, info, global_step):
        for key, val in info.items():
            if key.endswith("#rew"):
                key = key.replace("#", "/")

            if "/" in key:
                if "#" in key:
                    key = key.replace("#", "")
                self._add(key, val)

    def output(self):
        dico = {}
        for key, val in self.list_logs.items():
            dico[key] = np.concatenate(val).mean()
        dico.update(self.special_logs)
        return dico

