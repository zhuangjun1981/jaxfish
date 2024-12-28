import jax
import timeit
import jax.numpy as jnp


def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x = jnp.arange(1000000)
timeit.timeit("selu(x).block_until_ready()")

# if __name__ == "__main__":
#     x = jnp.arange(1000000)
#     timeit.timeit("selu(x).block_until_ready()")
