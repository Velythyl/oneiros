from time import sleep

import numpy as np
import pandas
import torch
import wandb
from tqdm import tqdm


class WANDB_CSV:
    def __init__(self, metadata=None):
        self.dico = {"time": []}

        self.metadata = metadata

    @property
    def time(self):
        return len(self.dico["time"])

    def add_time(self):
        self.dico["time"].append(self.time)
        return self.time

    def log(self, dico):
        if self.metadata is not None:
            self.metadata.update(dico)
            dico = self.metadata
            self.metadata = None

        editted_keys = set(dico.keys())
        untouched_keys = set(self.dico.keys()) - editted_keys - set(["time"])

        def time_fill():
            ret = [None]
            for _ in range(self.time):
                ret.append(None)
            return ret

        for key in editted_keys:
            assert isinstance(key, str)

            val = dico[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().item()

            if key in self.dico:
                assert isinstance(self.dico[key], list)
                self.dico[key].append(val)
            else:
                self.dico[key] = time_fill()
                self.dico[key][-1] = val

        for key in untouched_keys:
            assert isinstance(key, str)
            assert isinstance(self.dico[key], list)

            self.dico[key].append(None)

        self.add_time()
        self._assert_business_logic()

    def _assert_business_logic(self):
        LEN = len(self.dico[list(self.dico.keys())[0]])

        for key, val in self.dico.items():
            assert isinstance(key, str)
            assert len(val) == LEN

        assert LEN - 1 == self.dico["time"][-1]

    def get_np(self):
        ret = {}
        for key, val in self.dico.items():
            ret[key] = np.array(val)
        return ret

    def get_pd(self):
        as_np = self.get_np()
        return pandas.DataFrame(as_np)


def encapsulate(other_metadata={}):
    import wandb
    WANDB_INIT = wandb.init

    def _init(**wandb_init_kwargs):
        init(other_metadata, **wandb_init_kwargs)
        WANDB_INIT(**wandb_init_kwargs)

        WANDB_LOG = wandb.log

        def _log(dico):
            log(dico)
            WANDB_LOG(dico)

        wandb.log = _log

        WANDB_FINISH = wandb.finish

        def _finish(**wandb_finish_kwargs):
            finish(**wandb_finish_kwargs)
            WANDB_FINISH(**wandb_finish_kwargs)

        wandb.finish = _finish

    wandb.init = _init


instance = None


def init(other_metadata, **wandb_init_kwargs):
    global instance
    assert instance is None

    metadata = {
        "project": wandb_init_kwargs["project"],
        "name": wandb_init_kwargs["name"],
    }
    for tag in wandb_init_kwargs["tags"]:
        metadata[tag] = "tag"
    other_metadata.update(metadata)

    instance = WANDB_CSV(other_metadata)
    return _get_instance()


def finish(**wandb_finish_kwargs):
    wandb_run_dir = "/".join(wandb.run.dir.split("/")[:-1])
    global instance
    pd = instance.get_pd()
    pd.to_csv(f"{wandb_run_dir}/pd_logs.csv")
    instance = None


def _get_instance():
    assert instance is not None
    return instance


def log(dico):
    return _get_instance().log(dico)
