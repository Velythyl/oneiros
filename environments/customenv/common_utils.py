import functools

import jax.random
import jax.numpy as jp
import numpy as np

@functools.partial(jax.jit, static_argnums=2)
def random_sphere_jax(key, R, shape=None):
    # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume#

    key, rng = jax.random.split(key)
    phi = jax.random.uniform(rng, shape=shape, minval=0, maxval=2*jp.pi)
    key, rng = jax.random.split(key)
    costheta = jax.random.uniform(rng,shape=shape, minval=-1, maxval=1)
    key, rng = jax.random.split(key)
    u = jax.random.uniform(rng,shape=shape, minval=0, maxval=1)

    theta = jp.arccos(costheta)
    r = R * jp.cbrt(u)

    x = r * jp.sin(theta) * jp.cos(phi)
    y = r * jp.sin(theta) * jp.sin(phi)
    z = r * jp.cos(theta)

    return jp.stack((x,y,z),axis=1)

@functools.partial(jax.jit, static_argnums=3)
def random_sphere_jax_minradius(key, R, min_component, NUM_TRIALS=10000):

    def sample(kpm):
        key, rng = jax.random.split(kpm[0])
        points = random_sphere_jax(rng, R, shape=(NUM_TRIALS,))
        mask = jp.any(points < min_component, axis=1)
        return (key, points, mask)

    init_val = sample((key, None, None))
    kpm = jax.lax.while_loop(
        lambda kpm: kpm[-1].all(),
        sample,
        init_val
    )

    _, points, mask = kpm

    return points[jp.argmin(mask)]


def random_sphere_numpy(R, shape=None):
    # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume#

    phi = np.random.uniform(low=0, high=2 * np.pi, size=shape)
    costheta = np.random.uniform(low=-1, high=1, size=shape)
    u = np.random.uniform(low=0, high=1, size=shape)

    theta = np.arccos(costheta)
    r = R * np.cbrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack((x, y, z), axis=1)

def random_sphere_numpy_minradius(R, min_component, NUM_TRIALS=10000):

    def sample():
        points = random_sphere_numpy(R, shape=(NUM_TRIALS,))
        mask = np.any(points < min_component, axis=1)
        return points, mask

    p, m = sample()
    while m.all():
        p, m = sample()
    return p[m.argmin()]


def split_key(key, num):
    reset_rng, *set_rng = jax.random.split(key, num + 1)
    set_rng = jp.reshape(jp.stack(set_rng), (num, 2))
    return reset_rng, set_rng

if __name__ == '__main__':
    NUM = 10000
    key, keys = split_key(jax.random.PRNGKey(0), NUM)

    points = random_sphere_jax_minradius(key, 1.0, 0.3, NUM_TRIALS=10000)
    points = random_sphere_numpy_minradius(1.0, 0.3, NUM_TRIALS=10000)

    #points = jax.vmap(random_sphere_jax)(keys, jp.ones(NUM) )

    print(points.max())

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Extracting the coordinates
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
