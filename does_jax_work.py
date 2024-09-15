import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
import jax


@jax.jit
def matmul(a, b):
    return a @ b

def random_mat(shape, seed):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, shape)

SHAPE = (1000,1000)

mat1 = random_mat(SHAPE, 0)
mat2 = random_mat(SHAPE, 1)

print("If you get something like:")
print("\t(EXAMPLE OUTPUT) jaxlib.xla_extension.XlaRuntimeError: FAILED_PRECONDITION: DNN library initialization failed. Look at the errors above for more details.")
print("... or some other error message; it's because something broken ;) reinstall some other way.")
print("This is a problem with the way you install jax+torch for your platform, not necessarily with this library.")
print(matmul(mat1, mat2))
