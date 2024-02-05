import os

from typing import Optional

import brax.io.torch
import jax
import jax.numpy as jp
from flax import struct


def debug_jax_arr(arr):
    return brax.io.torch.jax_to_torch(arr).cpu().numpy()
@struct.dataclass
class _Data:
    pass

@struct.dataclass
class DummyData:
    i: jp.array


@struct.dataclass
class Node:
    prev: int
    data: Optional[_Data]
    next: jp.array  # array of IDs
    id: int
    is_sentinel: bool = False


def add_to_arr(arr, a):
    return jp.concatenate((arr, jp.array([a])))


@struct.dataclass
class Paths:
    id: int
    trunk: jp.array
    stars: jp.array

    data: _Data
    data_prototype: _Data

    @classmethod
    def init(cls, data):
        return Paths(0, jp.array([], dtype=jp.uint32), jp.array([[],[]], dtype=jp.uint32), data=data, data_prototype=data)

    def grow(self, data):
        @jax.jit
        def expand_merge(mother, data):
            yo_mama_flat, make_her_big_again = jax.tree_util.tree_flatten(mother)
            yo_data_flat, _ = jax.tree_util.tree_flatten(data)

            new_mom = []
            for m, d in zip(yo_mama_flat, yo_data_flat):
                new_mom.append(
                    jp.vstack((m, d[None]))
                )
            mother = jax.tree_util.tree_unflatten(make_her_big_again, new_mom)
            return mother

        return self.replace(
            data=expand_merge(self.data, data),
            trunk=add_to_arr(self.trunk, self.id),
            id=self.id + 1
        )

    @property
    def maxlen(self):
        return self.trunk.shape[0]  # by def, trunk is maxlen

    @property
    def main_branch_end(self):
        return len(self.trunk)

    def star(self, prev_ids, stars):
        # NOTE: ONCE YOU CALL THIS, YOU CAN'T GROW AGAIN!!! because itll mess up the paths

        @jax.jit
        def direct_merge(mother, data):
            yo_mama_flat, make_her_big_again = jax.tree_util.tree_flatten(mother)
            yo_data_flat, _ = jax.tree_util.tree_flatten(data)

            new_mom = []
            for m, d in zip(yo_mama_flat, yo_data_flat):
                new_mom.append(
                    jp.vstack((m, d))
                )
            mother = jax.tree_util.tree_unflatten(make_her_big_again, new_mom)
            return mother
        new_data = direct_merge(self.data, stars)
        num_added_data = prev_ids.shape[0]

        newly_minted_ids = jp.arange(num_added_data) + self.id
        stars = jp.vstack((prev_ids, newly_minted_ids))
        new_stars = jp.concatenate((self.stars, stars),axis=1)

        return self.replace(
            data=new_data,
            stars=new_stars.astype(jp.uint32),
            id=self.id+num_added_data
        )

    def clear(self):
        return Paths.init(self.data_prototype)

    def get_data_paths(self):
        def collect_traces(trunk, stars):
            traces = []
            # trunk path
            traces.append(trunk)

            for star in stars.T:
                trace = add_to_arr(trunk[:star[0]+1], star[1])
                traces.append(trace)
            return traces
            """
            find_ancestor = 0
            unique_branches = jp.unique(stars[0])
            ancestors = stars[0]
            for ancestor in unique_branches:
                find_ancestor = int(jp.nonzero(trunk[find_ancestor:] == ancestor)[0]) + find_ancestor
                ancestors = ancestors.at[ancestors == ancestor].set(find_ancestor)
            stars = stars.at[0].set(ancestors)

            for star in stars.T:
                trace = add_to_arr(trunk[:star[0]+1], star[1])
                traces.append(trace)
            return traces
            """
        traces = collect_traces(self.trunk, self.stars)

        @jax.jit
        def one_trace(indices, mother):
            yo_mama_flat, make_her_big_again = jax.tree_util.tree_flatten(mother)
            ret = []
            for item in yo_mama_flat:
                ret.append(item[indices])
            return jax.tree_util.tree_unflatten(make_her_big_again, ret)

        paths = []
        for trace in traces:
            paths.append(one_trace(trace+1, self.data))

        return paths

if __name__ == "__main__":
    paths = Paths.init(DummyData(jp.array([[-1]])))

    prev_ids = []
    adds = []
    for i in range(10):
        paths = paths.grow(DummyData(jp.array([i])))
        if i % 4 == 0:
            add = []
            for j in range(4):
                add.append(i*10 + j)
            adds.extend(add)
            prev_ids.extend([paths.main_branch_end] * 4)
    paths = paths.star(jp.array(prev_ids), DummyData(jp.array(adds)[:,None]))

    hehe = paths.get_data_paths()
    i=0

