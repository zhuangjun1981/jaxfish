import jax
import numpy as np
import jax.numpy as jnp
import scipy.ndimage as ni
from jaxfish.data_classes import Terrain, Connection
from functools import partial


@jax.jit
def get_fish_pixels(fish_position: jnp.ndarray) -> jnp.ndarray:
    """
    given [row, col] of fish center return the [row, col] of the 9 fish pixels

    Args:
      fish_position: jnp.ndarray, shape=(2,)
    Returns:
      fish_pixels: jnp.ndarray, shape=(9, 2)
    """
    indices = jnp.array(
        [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ],
        dtype=jnp.int32,
    )
    fish_pixels = fish_position[jnp.newaxis, :] + indices
    return fish_pixels


@jax.jit
def pick_one_zero(arr: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    pick a random zero pixel from the input array

    Args:
      arr: jnp.ndarray, 2d binary array
      key: random number generator

    Returns:
      position: jnp.ndarray, shape=(2,), [row, col] of a random pixel with value 0
    """

    # Get array shape
    height, width = arr.shape

    # Create array of all positions
    rows = jnp.arange(height)[:, None]
    cols = jnp.arange(width)[None, :]
    all_rows = jnp.broadcast_to(rows, (height, width)).ravel()
    all_cols = jnp.broadcast_to(cols, (height, width)).ravel()
    all_positions = jnp.stack([all_rows, all_cols], axis=1)

    all_positions = all_positions[jax.random.permutation(key, all_positions.shape[0])]

    init_val = 0
    cond_fun = lambda i: arr[*all_positions[i]] != 0
    body_fun = lambda i: i + 1
    zero_idx = jax.lax.while_loop(
        cond_fun=cond_fun, body_fun=body_fun, init_val=init_val
    )
    zero_position = all_positions[zero_idx]
    return zero_position


@partial(jax.jit, static_argnames="terrain")
def generate_terrain_map(terrain: Terrain) -> jnp.ndarray:
    """
    Given a dataclass terrain, return a 2d terrain map
    currently only minimap is supported.

    Args:
        terrain: Terrain dataclass
    Returns:
        terrain_map: jnp.ndarray
    """
    assert (
        terrain.should_use_minimap
    ), "Only minimap is supported. terrain.should_use_minimap must be True."

    terrain_map = jnp.zeros(terrain.minimap_size, dtype=jnp.uint8)
    margin = terrain.minimap_margin
    terrain_map = terrain_map.at[:margin, :].set(1)
    terrain_map = terrain_map.at[-margin:, :].set(1)
    terrain_map = terrain_map.at[:, :margin].set(1)
    terrain_map = terrain_map.at[:, -margin:].set(1)

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


@jax.jit
def update_food_positions(
    food_map: jnp.ndarray,
    food_positions: jnp.ndarray,
    is_eaten: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """
    given a simulation state generate valid food_positions

    Args:
      food_map: jnp.ndarray, shape=(m, n), uint8, 2d binary array. 1: invalid position to spawn food; 0: valid position to spawn food
      food_positions: jnp.ndarray, shape=(food_num, 2), int32, [row, col] for each food in previous time point
      is_eaten: jnp.ndarray, shape=(food_num,), bool, if the given food is eaten
      key: random number generator.
    """

    food_num = len(is_eaten)

    def update_one_eaten_food(operand):
        food_map, food_positions, key, i = operand

        pos = pick_one_zero(arr=food_map, key=key)
        food_positions = food_positions.at[i].set(pos)
        food_map = food_map.at[*pos].set(1)

        return food_map, food_positions

    def update_step(carry, i):
        food_map, food_positions, is_eaten, key = carry

        key, subkey = jax.random.split(key)

        food_map, food_positions = jax.lax.cond(
            is_eaten[i],
            update_one_eaten_food,
            lambda x: (x[0], x[1]),
            operand=(food_map, food_positions, subkey, i),
        )

        return (food_map, food_positions, is_eaten, key), None

    (_, food_positions, _, _), _ = jax.lax.scan(
        update_step, (food_map, food_positions, is_eaten, key), jnp.arange(food_num)
    )

    return food_positions


@jax.jit
def update_food_positions_in_simulation(
    terrain_map: jnp.ndarray,
    fish_position: jnp.ndarray,
    food_positions: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.int32]:
    def is_fish_pixel(pos):
        return jnp.all(
            (pos[0] >= fish_position[0] - 1)
            & (pos[0] <= fish_position[0] + 1)
            & (pos[1] >= fish_position[1] - 1)
            & (pos[1] <= fish_position[1] + 1)
        )

    is_eaten = jax.vmap(is_fish_pixel)(food_positions)
    eaten_food_num = jnp.sum(is_eaten)

    # generate food map, binary, 0: valid position to spawn food, 1: invalid position to spawn food
    # the invalid positions are
    #   1. land pixel
    #   2. pixels occupied by fish
    #   3. pixels occupied by food
    food_map = jnp.array(terrain_map)
    fish_pixels = get_fish_pixels(fish_position)
    food_map = food_map.at[food_positions.T[0], food_positions.T[1]].set(1)
    food_map = food_map.at[fish_pixels.T[0], fish_pixels.T[1]].set(1)

    food_positions = update_food_positions(
        food_map=food_map, food_positions=food_positions, is_eaten=is_eaten, key=key
    )

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

    updated_food_positions, eaten_food_num = update_food_positions_in_simulation(
        terrain_map=minimap,
        fish_position=fish_position,
        food_positions=food_positions,
        key=food_key,
    )

    print(f"{food_positions=}")
    print(f"{updated_food_positions=}")
    print(f"{eaten_food_num=}")
