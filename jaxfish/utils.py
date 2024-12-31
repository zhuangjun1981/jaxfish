import numpy as np
from jaxfish.data_classes import Terrain
import jax.numpy as jnp


def generate_terrain_map(terrain: Terrain):
    if terrain.should_use_minimap:
        terrain_map = np.zeros(terrain.minimap_size, dtype=np.uint8)
        margin = terrain.minimap_margin
        terrain_map[:margin, :] = 1
        terrain_map[-margin:, :] = 1
        terrain_map[:, :margin] = 1
        terrain_map[:, -margin:] = 1
    else:
        raise NotImplementedError("regular map other than minimap not implemented")

    return jnp.array(terrain_map)
