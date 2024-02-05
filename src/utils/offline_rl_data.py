import dataclasses
import functools
import itertools
import os
import random
import timeit
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler

from torch.utils.data import DataLoader
from torch.utils.data import random_split


#@functools.lru_cache(maxsize=16)
#def load(path):
#    # print("CALLED")
#    # print(load.cache_info().hits)
#    return torch.load(path)

def load(path):
    # print("CALLED")
    # print(load.cache_info().hits)
    ret = torch.load(path)
    for key, val in ret.items():
        val.requires_grad = False
    return ret

class Cache:
    def __init__(self, size, all_paths, only_in_cache=False):
        self.cach = {}

        randperm = np.random.permutation(len(all_paths))

        for i, idx in enumerate(randperm):

            if i >= size:
                break

            path = all_paths[idx]
            self.cach[path] = load(path)
        self.saved_paths = list(self.cach.keys())

        self.only_in_cache = only_in_cache

    def __call__(self, path):
        if path in self.cach:
            return self.cach[path], True

        if self.only_in_cache:
            rand_choice = random.choice(self.saved_paths)
            return self.cach[rand_choice], True

        return load(path), False


class OfflineExpertDataset(Dataset):
    def __init__(self, expert_paths):
        self.expert_paths = expert_paths
        self.file_paths = {expert: [] for expert in expert_paths}
        self.all_files = []
        for expert in expert_paths:
            subfiles = os.listdir(expert)
            subfiles = list(
                map(lambda x: f"{expert}/{x}",
                    filter(lambda x: x.startswith("rb_"),
                           subfiles)
                    )
            )

            self.all_files.extend(subfiles)
            self.file_paths[expert] = subfiles

        self.load = Cache(44, self.all_files, only_in_cache=True)
        temp_buf = load(self.all_files[0])["action"]
        self.single_buf_shape = temp_buf.shape
        self.n_elements_in_single_buf = self.single_buf_shape[0] * (self.single_buf_shape[1])
        self.n_elements_per_expert = OrderedDict()
        for expert, paths in self.file_paths.items():
            _n_paths = len(paths)
            self.n_elements_per_expert[expert] = _n_paths * self.n_elements_in_single_buf

        summed = 0
        self.sum_n_elements_per_expert = OrderedDict()
        for expert, num in self.n_elements_per_expert.items():
            summed += num
            self.sum_n_elements_per_expert[expert] = summed

        self.num_datapoints = sum(self.n_elements_per_expert.values())

    def __len__(self):
        return self.num_datapoints

    def parse_idx(self, idx):
        expert_found = False
        buffer_adjustment = 0
        for expert_path, n_elements in self.sum_n_elements_per_expert.items():
            if idx < n_elements:
                expert_found = True
                break
            buffer_adjustment = n_elements
        assert expert_found, "Could not find valid expert. Something's wrong with your dataset, or paths, or requested idx, or something."

        idx -= buffer_adjustment

        buf_idx, item_idx = divmod(idx, self.n_elements_in_single_buf)

        # env_idx = item_idx % self.single_buf_shape[0]

        env_idx, inner_idx = divmod(item_idx, self.single_buf_shape[1])  # n times you can fit BUF_LEN in idx
        if inner_idx == self.single_buf_shape[1] - 1:
            # is this 100% accurate? no. But who cares.
            # this ensure that we will NEVER have to cross envs or buffers
            # at the cost of NUM_ENVS * NUM_BUFFERS lost elements (order of 16 * 4000)
            inner_idx -= 1
        elif inner_idx > self.single_buf_shape[1] - 1:
            raise AssertionError("Something went wrong with the inner idx?")

        return (expert_path, buf_idx, env_idx, inner_idx)

    def __getitem__(self, idx):
        (exper_path, buf_idx, env_idx, inner_idx) = self.parse_idx(idx)

        buf, cache_hit = self.load(f"{exper_path}/rb_{buf_idx}.pth")

        ret = []
        for key in ["obs", "action", "next_obs", "rew", "done"]:
            _key = key
            _inner_idx = inner_idx

            if key == "next_obs":
                _key = "obs"
                _inner_idx = inner_idx + 1

            keyed_buf = buf[_key]

            # if (keyed_buf == -torch.inf).any() or (keyed_buf == torch.inf).any():
            #    raise AssertionError("Some elements of the offline data were marked by a sentinel value. Something's wrong with the data collection.")
            elem = keyed_buf[env_idx, _inner_idx]
            if not cache_hit:
                elem = torch.clone(elem)

            ret.append(elem)
        ret = tuple(ret)
        return ret




class TrueRandomSampler(BatchSampler):
    r"""Samples elements randomly WITH replacement. Just returns a uniformly sampled int.. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, length, batch_size):
        self.len = length
        self.batch_size = batch_size

    @property
    def num_samples(self):
        return self.len

    def __iter__(self):
        class Rand:
            def __init__(self, n_to_gen):
                self.n_to_gen = n_to_gen
                self.n_genned = 0

            def __iter__(self):
                return self

            def __next__(self):  # Python 2: def next(self)
                if self.n_genned > self.n_to_gen:
                    raise StopIteration
                self.n_genned += 1
                return random.randint(0, self.n_to_gen - 1)

        return Rand(self.num_samples)

    def __len__(self):
        return self.num_samples


def get_dataset():
    root_path = f"/home/charlie/Desktop/offline-adaptation/wandb"
    squid_path = f"{root_path}/halfsquid_expert"
    cheetah_path = f"{root_path}/halfcheetah_expert"
    crawler_path = f"{root_path}/halfcrawler_expert"
    x = OfflineExpertDataset(expert_paths=[squid_path, cheetah_path, crawler_path])

    loader = DataLoader(x, batch_size=256, shuffle=False, sampler=TrueRandomSampler(len(x), batch_size=256),
                        batch_sampler=None, num_workers=0, collate_fn=None,
                        pin_memory=False, drop_last=False, timeout=0,
                        worker_init_fn=None, prefetch_factor=None,
                        persistent_workers=False)
    return loader

if __name__ == "__main__":

    loader = get_dataset()

    # for batch_idx, batch in enumerate(loader):
    #    print(f"BATCH {batch_idx}")
    #    break
    # exit()

    def profile():
        i = 0
        for batch_idx, batch in enumerate(loader):
            print(f"BATCH {batch_idx}")
            i += 1
            if i > 20:
                break
        return i


    print(timeit.timeit(stmt=profile, number=100))
