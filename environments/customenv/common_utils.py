import functools

import jax.random
import jax.numpy as jp
import numpy as np


@functools.partial(jax.jit, static_argnums=3)
def random_sphere_jax(key, min_r, max_r, shape=None):
    # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume#

    key, rng = jax.random.split(key)
    phi = jax.random.uniform(rng, shape=shape, minval=0, maxval=2*jp.pi)
    key, rng = jax.random.split(key)
    costheta = jax.random.uniform(rng,shape=shape, minval=-1, maxval=1)
    key, rng = jax.random.split(key)
    u = jax.random.uniform(rng, shape=shape, minval=0, maxval=1)

    theta = jp.arccos(costheta)
    r = jp.cbrt((u * max_r ** 3) + ((1 - u) * min_r ** 3))

    x = r * jp.sin(theta) * jp.cos(phi)
    y = r * jp.sin(theta) * jp.sin(phi)
    z = r * jp.cos(theta)

    return jp.stack((x,y,z),axis=1)


def random_sphere_numpy(min_r, max_r, shape=None):
    # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume#

    phi = np.random.uniform(low=0, high=2 * np.pi, size=shape)
    costheta = np.random.uniform(low=-1, high=1, size=shape)
    u = np.random.uniform(low=0, high=1, size=shape)

    theta = np.arccos(costheta)
    r = np.cbrt((u * max_r ** 3) + ((1 - u) * min_r ** 3))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack((x, y, z), axis=1)

def split_key(key, num):
    reset_rng, *set_rng = jax.random.split(key, num + 1)
    set_rng = jp.reshape(jp.stack(set_rng), (num, 2))
    return reset_rng, set_rng

if __name__ == '__main__':
    NUM = 10000
    key, keys = split_key(jax.random.PRNGKey(0), NUM)

   # points = []
    #for i in range(NUM):
    #    points.append(random_sphere_jax_minradius(keys[i], 1.0, 0.3, NUM_TRIALS=10000))
    #points = random_sphere_numpy_minradius(1.0, 0.3, NUM_TRIALS=10000)
    #points = random_points_with_min_component_jax(0.3, 0.66, 1000)

    #points = jp.stack(points)

    points = random_sphere_jax(key, 0.3, 0.6, shape=(10000,))

    """
    def gen(key, NUM, l, h):
        return jax.random.uniform(key, shape=(NUM, 1), minval=l, maxval=)

    _, keys = split_key(key, 4 * 3)
    neg_x_pos_y = jp.concatenate((-gen(keys[0], NUM // 4), gen(keys[1], NUM // 4), gen(keys[2], NUM // 4)), axis=1) #jax.random.uniform(key, shape=(NUM // 2, 1), minval=0.3, maxval=0.6)
    pos_x_pos_y = jp.concatenate((gen(keys[3], NUM // 4), gen(keys[4], NUM // 4), gen(keys[5], NUM // 4)), axis=1)
    neg_x_neg_y = jp.concatenate((-gen(keys[6], NUM // 4), -gen(keys[7], NUM // 4), gen(keys[8], NUM // 4)), axis=1)
    pos_x_neg_y = jp.concatenate((gen(keys[9], NUM // 4), -gen(keys[10], NUM // 4), gen(keys[11], NUM // 4)), axis=1)

    points = jp.concatenate((neg_x_pos_y, pos_x_pos_y, pos_x_neg_y, neg_x_neg_y))
    """
    #points = jax_random_target(key, absmincomp=0.3, absmaxcomp=0.6, NUM=10000)

    #points = jax.vmap(random_sphere_jax)(keys, jp.ones(NUM) )

    print(jp.abs(points).max())
    print(jp.abs(points).min())

    def mse(p):
        return jp.linalg.norm(p)
    dists = jax.vmap(mse)(points)

    print(dists.max())
    print(dists.min())

    argmin = dists.argmin()
    print(points[argmin])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Extracting the coordinates
    points = jp.abs(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Creating the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')

    # Setting labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
