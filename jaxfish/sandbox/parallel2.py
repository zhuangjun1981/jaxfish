import jax
import jax.numpy as jnp
import os

# simulate multicore gpu on cpu
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# jax.distributed.initialize(
#     coordinator_address="localhost",
#     num_processes=2,
#     process_id=0,
# )


def f(x):
    return jnp.dot(x, x)


x = jnp.ones((8, 100))
parallel_f = jax.pmap(f)

# Execute on multiple devices
result = parallel_f(x)

print(result)
