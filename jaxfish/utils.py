import jax
import numpy as np
import jax.numpy as jnp
import scipy.ndimage as ni
from jaxfish.data_classes import Terrain, Connection
from functools import partial


def generate_terrain_map(terrain: Terrain) -> jnp.ndarray:
    """
    Given a dataclass terrain, return a 2d terrain map
    currently only minimap is supported.

    Args:
        terrain: Terrain dataclass
    Returns:
        terrain_map: jnp.ndarray
    """
    if terrain.should_use_minimap:
        terrain_map = np.zeros(terrain.minimap_size, dtype=np.uint8)
        margin = terrain.minimap_margin
        terrain_map[:margin, :] = 1
        terrain_map[-margin:, :] = 1
        terrain_map[:, :margin] = 1
        terrain_map[:, -margin:] = 1
    else:
        raise NotImplementedError("regular map other than minimap not implemented")

    return terrain_map


def get_starting_fish_position(terrain_map: np.ndarray, rng: jax.Array):
    """
    Return an (row, col) array that can be a starting position of the fish.
    the starting position is the center of the 3 x 3 fish body and ensures
    that the fish body will not overlap with land in the terrain_map (with
    value of 1)
    """

    position_map = ni.binary_dilation(
        terrain_map,
        structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        border_value=1,
    )

    locations = np.array(np.where(position_map == 0)).T

    return jax.random.choice(key=rng, a=locations)


def update_food_positions(
    terrain_map: jnp.ndarray,
    fish_position: jnp.ndarray,
    food_positions: jnp.ndarray,
    rng: jax.Array,
) -> tuple[jnp.ndarray, int]:
    """
    update food positions, given the state of simulation.
    if no food is eaten keep the same food positions.
    if there is food being eaten, remove those food and
    spawn new food at random locations, these locations will
    not overlap with the land in terrain_map or the fish body

    note: this assume that there is food being eaten. If no food was
    eaten, this function can be by pass by lax.cond in the simulation
    loop

    Args:
        terrain_map: jnp.ndarray, 2d binary map, 0: water, 1: land
        fish_position: jnp.ndarray, (row, col), body center of the fish
        food_positions: jnp.ndarray, shape = (food_num, 2), food_positions
          in the previous time point. each row is [row, col] of each food
          pellet
        eaten_food_positions: jnp.ndarray, shape = (eaten_food_num, 2),
          food_positions that were eaten in the current time point. each row
          is [row, col] of each eaten food pellet, this should be a subset
          of food_positions
        rng: a jax.random key

    Returns:
        food_positions: jnp.ndarray, shape = (food_num, 2), updated food
          positions.
        eaten_food_num: int, number of food eaten
    """

    fish_pixels = set(
        [
            (row, col)
            for row in range(fish_position[0] - 1, fish_position[0] + 2)
            for col in range(fish_position[1] - 1, fish_position[1] + 2)
        ]
    )
    food_pixels = set([tuple(p) for p in food_positions.tolist()])
    eaten_food_pixels = fish_pixels.intersection(food_pixels)
    eaten_food_num = len(eaten_food_pixels)

    print(food_pixels)

    if eaten_food_num == 0:
        return food_positions, 0

    left_food_pixels = food_pixels - eaten_food_pixels
    available_food_pixels = jnp.array(jnp.where(terrain_map == 0)).T
    available_food_pixels = set([tuple(p) for p in available_food_pixels.tolist()])
    available_food_pixels = available_food_pixels - fish_pixels - left_food_pixels

    new_food_pixels = jax.random.choice(
        rng, jnp.array(list(available_food_pixels)), (eaten_food_num,)
    )
    new_food_pixels = set([tuple(p) for p in new_food_pixels.tolist()])

    food_pixels = left_food_pixels.union(new_food_pixels)
    food_positions = jnp.array(list(food_pixels))

    return food_positions, eaten_food_num


def generate_psp_waveform(connection: Connection) -> jnp.ndarray:
    """
    generate unit post synaptic probability wave form for a given connection
    """

    psp = np.zeros(connection.latency + connection.rise_time + connection.decay_time)
    psp[connection.latency : connection.latency + connection.rise_time] = (
        connection.amplitude
        * (np.arange(connection.rise_time) + 1).astype(np.float32)
        / float(connection.rise_time)
    )

    psp[-connection.decay_time :] = (
        connection.amplitude
        * (np.arange(connection.decay_time, 0, -1) - 1).astype(np.float32)
        / float(connection.decay_time)
    )

    return jnp.array(psp, dtype=jnp.float32)


if __name__ == "__main__":
    seed = 0
    key = jax.random.key(seed)
    fish_key, food_key = jax.random.split(key, 2)

    terrain = Terrain(minimap_size=(7, 7))
    minimap = generate_terrain_map(terrain)
    fish_position = get_starting_fish_position(minimap, rng=key)
    food_positions = jnp.array([[1, 1], [1, 5], [3, 1]])

    updated_food_positions, eaten_food_num = update_food_positions(
        terrain_map=minimap,
        fish_position=fish_position,
        food_positions=food_positions,
        rng=food_key,
    )

    print(f"{food_positions=}")
    print(f"{updated_food_positions=}")
    print(f"{eaten_food_num=}")
