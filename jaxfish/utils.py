import jax
import numpy as np
import jax.numpy as jnp
import scipy.ndimage as ni
from jaxfish.data_classes import Terrain


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
    eaten_food_positions: jnp.ndarray,
    rng: jax.Array,
) -> jnp.ndarray:
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
    """
    if len(eaten_food_positions) == 0:
        return food_positions
    else:
        mask = jnp.any(
            jnp.all(food_positions[:, None] == eaten_food_positions[None, :], axis=-1),
            axis=-1,
        )
        new_food_num = jnp.sum(mask)

        position_map = jnp.array(terrain_map)
        position_map = position_map.at[
            fish_position[0] - 1 : fish_position[0] + 2,
            fish_position[1] - 1 : fish_position[1] + 2,
        ].set(1)

        left_food_positions = food_positions[~mask]

        if len(left_food_positions) > 0:
            position_map = position_map.at[
                left_food_positions.T[0],
                left_food_positions.T[1],
            ].set(1)

        available_food_positions = jnp.array(jnp.where(position_map == 0)).T

        new_food_positions = jax.random.choice(
            rng,
            available_food_positions,
            (new_food_num,),
        )

        indices = jnp.where(mask)[0]
        for i, idx in enumerate(indices):
            food_positions = food_positions.at[idx].set(new_food_positions[i])

        return food_positions


if __name__ == "__main__":
    seed = 0
    key = jax.random.key(seed)
    fish_key, food_key = jax.random.split(key, 2)

    terrain = Terrain(minimap_size=(7, 7))
    minimap = generate_terrain_map(terrain)
    fish_position = get_starting_fish_position(minimap, rng=key)

    updated_food_positions = update_food_positions(
        terrain_map=minimap,
        fish_position=fish_position,
        food_positions=jnp.array([[1, 1], [2, 1], [3, 1]]),
        eaten_food_positions=jnp.array([[1, 1]]),
        rng=food_key,
    )

    print(updated_food_positions)
