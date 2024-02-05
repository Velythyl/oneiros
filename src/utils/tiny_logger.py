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
        if done.any():
            episodic_return = info['train#r'][done.bool()].cpu().numpy()
            episodic_length = info['train#l'][done.bool()].float().cpu().numpy()
            self._add("perf/ep_r", episodic_return)
            self._add("perf/ep_l", episodic_length)
            episodic_return = episodic_return.mean()
            episodic_length = episodic_length.mean()
            print(
                f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}")

        for key, val in info.items():
            if "#" in key:
                prefix, key = key.split("#")
                out_key = f"{prefix.upper()}_{key}"
            else:
                prefix = ""
                out_key = key

            if key.startswith(f"mbrma/"):
                self._add(out_key, val)

    def output(self):
        dico = {}
        for key, val in self.list_logs.items():
            dico[key] = np.concatenate(val).mean()
        dico.update(self.special_logs)
        return dico

