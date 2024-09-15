import jax
from jax.tree_util import tree_flatten, tree_unflatten

@jax.jit
def take0(state):
    flat, treedef = tree_flatten(state)

    transformed = [fla[0] for fla in flat]

    unflat = tree_unflatten(treedef, transformed)
    return unflat
