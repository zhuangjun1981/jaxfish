import jax
import numpy as np
import jax.numpy as jnp
from typing import Union
from jaxfish.data_classes import (
    Terrain,
    ConnectionFrozen,
    EyeFrozen,
    NeuronFrozen,
    MuscleFrozen,
    BrainFrozen,
)
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
def update_fish_position(
    curr_fish_position: jnp.ndarray,
    move_attempt: jnp.ndarray,
    terrain_map: jnp.ndarray,
) -> jnp.ndarray:
    """
    given current fish position and move attempt and the terrain map,
    return updated fish position, mostly making sure it will not move
    out of boundary
    """
    n_rows, n_cols = terrain_map.shape
    updated_fish_position = curr_fish_position + move_attempt

    updated_fish_position = jnp.vstack((updated_fish_position, jnp.array([1, 1])))
    updated_fish_position = jnp.max(updated_fish_position, axis=0)
    updated_fish_position = jnp.vstack(
        (updated_fish_position, jnp.array([n_rows - 2, n_cols - 2]))
    )
    updated_fish_position = jnp.min(updated_fish_position, axis=0)

    return updated_fish_position


# def get_land_overlap(
#     terrain_map,
#     fish_position,
# ):
#     fish_pixels = get_fish_pixels(fish_position)


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


@jax.jit
def get_starting_fish_position(terrain_map: np.ndarray, key: jax.Array):
    """
    Return an (row, col) array that can be a starting position of the fish.
    the starting position is the center of the 3 x 3 fish body and ensures
    that the fish body will not overlap with land in the terrain_map (with
    value of 1)
    """

    # Get array shape
    height, width = terrain_map.shape

    # Create array of all positions
    rows = jnp.arange(1, height - 1)[:, None]
    cols = jnp.arange(1, width - 1)[None, :]
    all_rows = jnp.broadcast_to(rows, (height - 2, width - 2)).ravel()
    all_cols = jnp.broadcast_to(cols, (height - 2, width - 2)).ravel()
    all_positions = jnp.stack([all_rows, all_cols], axis=1)

    all_positions = all_positions[jax.random.permutation(key, all_positions.shape[0])]

    # jax.debug.print("{all_positions}", all_positions=all_positions)

    init_val = 0

    def cond_fun(i):
        fish_position = all_positions[i]
        fish_pixels = get_fish_pixels(fish_position)

        def is_zero(fish_pixel):
            return terrain_map[*fish_pixel] == 0

        is_zero_arr = jax.vmap(is_zero)(fish_pixels)

        return ~is_zero_arr.all()

    body_fun = lambda i: i + 1
    fish_pos_idx = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=init_val,
    )
    fish_position = all_positions[fish_pos_idx]
    return fish_position


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


@partial(jax.jit, static_argnames="psp_waveform_length")
def generate_psp_waveform(
    latency: int,
    rise_time: int,
    decay_time: int,
    amplitude: float,
    psp_waveform_length: int,
):
    # assert psp_waveform_length >= (latency + rise_time + decay_time)

    psp = jnp.zeros(psp_waveform_length)

    # add rising phase
    psp = jax.lax.fori_loop(
        latency,
        latency + rise_time,
        lambda i, psp: psp.at[i].set(amplitude * (i - latency + 1) / rise_time),
        psp,
    )

    # add decay phase
    psp = jax.lax.fori_loop(
        latency + rise_time,
        latency + rise_time + decay_time,
        lambda i, psp: psp.at[i].set(
            amplitude - (amplitude / decay_time) * (i - latency - rise_time + 1)
        ),
        psp,
    )

    return psp


@jax.jit
def get_input_terrain_eye(
    rf_positions: jnp.ndarray,
    rf_weights: jnp.ndarray,
    gain: float,
    terrain_map: jnp.ndarray,
    fish_position: jnp.ndarray,
) -> float:
    map_pixels = terrain_map[
        rf_positions[0] + fish_position[0],
        rf_positions[1] + fish_position[1],
    ]

    input = jnp.dot(rf_weights, map_pixels) * gain

    return input


def get_input_food_eye(
    rf_positions: jnp.ndarray,
    rf_weights: jnp.ndarray,
    gain: float,
    food_positions: jnp.ndarray,
    fish_position: jnp.ndarray,
):
    # Broadcast fish_position to match rf_positions shape
    fish_pos_broadcasted = fish_position[:, jnp.newaxis]

    # Calculate pixel positions for all receptive fields
    pix_positions = (fish_pos_broadcasted + rf_positions).T

    # Reshape the arrays for broadcasting
    rf_pix_reshaped = pix_positions[:, jnp.newaxis, :]
    food_pix_reshaped = food_positions[jnp.newaxis, :, :]

    # matches, shape = (num_rf_pixels, )
    matches = jnp.any(jnp.all(rf_pix_reshaped == food_pix_reshaped, axis=2), axis=1)
    return jnp.dot(matches, rf_weights) * gain


@partial(jax.jit, static_argnames=["neuron"])
def get_input_neuron(
    neuron: Union[EyeFrozen, NeuronFrozen, MuscleFrozen],
    terrain_map: jnp.ndarray,
    food_positions: jnp.ndarray,
    fish_position: jnp.ndarray,
):
    if neuron.type == "eye_frozen":
        if neuron.input_type == "terrain":
            input = get_input_terrain_eye(
                rf_positions=jnp.array(neuron.rf_positions),
                rf_weights=jnp.array(neuron.rf_weights),
                gain=neuron.gain,
                terrain_map=terrain_map,
                fish_position=fish_position,
            )
        elif neuron.input_type == "food":
            input = get_input_food_eye(
                rf_positions=jnp.array(neuron.rf_positions),
                rf_weights=jnp.array(neuron.rf_weights),
                gain=neuron.gain,
                food_positions=food_positions,
                fish_position=fish_position,
            )
    else:
        input = 0.0

    return input


@jax.jit
def find_last_firing_time(firing_history: jnp.ndarray) -> int:
    """
    return the index of last firing in the 1d firing history array

    Args:
      firing_history: jnp.ndarray, 1d, binary, firing or not in each time point

    Returns:
      firing_time: int, the index of last firing time, if no firing, return -1
    """
    indices = jnp.arange(firing_history.shape[0])
    masked_indices = jnp.where(firing_history == 1, indices, -1)
    return jnp.max(masked_indices).astype(jnp.int32)


@partial(jax.jit, static_argnames="neuron")
def evaluate_neuron(
    neuron_idx: int,
    t: int,
    neuron: Union[EyeFrozen, NeuronFrozen, MuscleFrozen],
    terrain_map: jnp.ndarray,
    food_positions_history: jnp.ndarray,
    fish_position_history: jnp.ndarray,
    firing_history: jnp.ndarray,
    psp_history: jnp.ndarray,
    firing_keys: jax.Array,
):
    input = get_input_neuron(
        neuron=neuron,
        terrain_map=terrain_map,
        food_positions=food_positions_history[t],
        fish_position=fish_position_history[t],
    )

    base_val = input + neuron.baseline_rate + psp_history[neuron_idx][t]
    last_firing_time = find_last_firing_time(firing_history[neuron_idx])

    def true_fn(operand):
        is_firing, firing_history = operand
        firing_history = firing_history.at[neuron_idx, t].set(1)
        return True, firing_history

    condition = (last_firing_time + neuron.refractory_period < t) & (
        jax.random.uniform(firing_keys[neuron_idx, t]) < base_val
    )
    is_firing, firing_history = jax.lax.cond(
        pred=condition,
        true_fun=true_fn,
        fals_fun=lambda x: x,
        operand=(False, firing_history),
    )

    return is_firing, firing_history


@jax.jit
def update_psp_history(
    t: int,
    is_firing: int,
    pre_neuron_idx: int,
    post_neuron_idx: int,
    psp_waveforms: jnp.ndarray,
    psp_history: jnp.ndarray,
) -> jnp.ndarray:
    psp_length = psp_waveforms.shape[1]
    length = psp_history.shape[1]
    end_idx = jnp.minimum(t + psp_length, length)

    psp_history = jax.lax.fori_loop(
        0,
        end_idx - t,
        lambda i, val: val.at[post_neuron_idx, i + t].add(
            psp_waveforms[pre_neuron_idx, i] * is_firing
        ),
        psp_history,
    )

    return psp_history


if __name__ == "__main__":
    psp_history = jnp.zeros((4, 20))
    psp_waveforms = jnp.arange(6, dtype=jnp.float32).reshape((2, 3))

    psp_history = update_psp_history(
        t=2,
        is_firing=True,
        pre_neuron_idx=1,
        post_neuron_idx=3,
        psp_waveforms=psp_waveforms,
        psp_history=psp_history,
    )

    print(psp_history)

    # # ===========================================================
    # seed = 0
    # key = jax.random.key(seed)
    # fish_key, food_key = jax.random.split(key, 2)

    # terrain = Terrain(minimap_size=(6, 6))
    # minimap = generate_terrain_map(terrain)
    # fish_position = get_starting_fish_position(minimap, key=key)
    # food_positions = jnp.array([[1, 1], [1, 4], [3, 1]])

    # updated_food_positions, eaten_food_num = update_food_positions_in_simulation(
    #     terrain_map=minimap,
    #     fish_position=fish_position,
    #     food_positions=food_positions,
    #     key=food_key,
    # )

    # print(minimap)
    # print(f"{fish_position=}")
    # print(f"{food_positions=}")
    # print(f"{updated_food_positions=}")
    # print(f"{eaten_food_num=}")

    # # =======================================================
    # from jaxfish.data_classes import freeze, MINIMUM_BRAIN

    # brain = freeze(MINIMUM_BRAIN)
    # psp_waveform = generate_psp_waveform(brain.connections[0])

    # print(brain.connections)
    # print(type(brain.connections))
    # print(len(brain.connections))

    # psp_waveforms = jax.vmap(generate_psp_waveform, 0, 0)(brain.connections)

    # # =======================================================
    # amplitude = 4.0
    # latency = 3
    # rise_time = 2
    # decay_time = 4
    # psp_waveform_length = 10
    # psp_waveform = generate_psp_waveform(
    #     latency=latency,
    #     rise_time=rise_time,
    #     decay_time=decay_time,
    #     amplitude=amplitude,
    #     psp_waveform_length=psp_waveform_length,
    # )
    # print(psp_waveform)

    # =======================================================
    # from jaxfish.data_classes import EIGHT_EYES, freeze

    # seed = 0
    # key = jax.random.key(seed)
    # fish_key, food_key = jax.random.split(key, 2)

    # terrain = Terrain(minimap_size=(6, 6))
    # minimap = generate_terrain_map(terrain)
    # food_positions = jnp.array([[1, 1], [1, 4], [3, 1]])
    # fish_position = get_starting_fish_position(minimap, key=key)

    # print(minimap)
    # print(f"{fish_position=}")
    # print(f"{food_positions=}")

    # eye_terr = EIGHT_EYES["south"]
    # eye_terr.input_type = "terrain"
    # eye_terr = freeze(eye_terr)
    # input_terr = get_input_terrain_eye(
    #     eye=eye_terr,
    #     terrain_map=minimap,
    #     fish_position=fish_position,
    # )
    # print(f"{input_terr=}")  # should be 0.45

    # eye_terr = EIGHT_EYES["north"]
    # eye_terr.input_type = "terrain"
    # eye_terr = freeze(eye_terr)
    # input_terr = get_input_terrain_eye(
    #     eye=eye_terr,
    #     terrain_map=minimap,
    #     fish_position=fish_position,
    # )
    # print(f"{input_terr=}")  # should be 0.9

    # eye_food = EIGHT_EYES["west"]
    # eye_food.input_type = "food"
    # eye_food = freeze(eye_food)
    # input_food = get_input_food_eye(
    #     eye=eye_food,
    #     food_positions=food_positions,
    #     fish_position=fish_position,
    # )
    # print(f"{input_food=}")  # should be 0.3

    # eye_food = EIGHT_EYES["northeast"]
    # eye_food.input_type = "food"
    # eye_food = freeze(eye_food)
    # input_food = get_input_food_eye(
    #     eye=eye_food,
    #     food_positions=food_positions,
    #     fish_position=fish_position,
    # )
    # print(f"{input_food=}")  # should be 0.1

    # eye_terr_0 = EIGHT_EYES["south"]
    # eye_terr_0.input_type = "terrain"
    # eye_terr_0 = freeze(eye_terr_0)
    # input = get_input_neuron(
    #     neuron=eye_terr_0,
    #     terrain_map=minimap,
    #     food_positions=food_positions,
    #     fish_position=fish_position,
    # )
    # print(input)  # should be 0.45

    # eye_terr_1 = EIGHT_EYES["north"]
    # eye_terr_1.input_type = "terrain"
    # eye_terr_1 = freeze(eye_terr_1)
    # input = get_input_neuron(
    #     neuron=eye_terr_1,
    #     terrain_map=minimap,
    #     food_positions=food_positions,
    #     fish_position=fish_position,
    # )
    # print(input)  # should be 0.9

    # eye_food_0 = EIGHT_EYES["west"]
    # eye_food_0.input_type = "food"
    # eye_food_0 = freeze(eye_food_0)
    # input = get_input_neuron(
    #     neuron=eye_food_0,
    #     terrain_map=minimap,
    #     food_positions=food_positions,
    #     fish_position=fish_position,
    # )
    # print(input)  # should be 0.3

    # eye_food_0 = EIGHT_EYES["northeast"]
    # eye_food_0.input_type = "food"
    # eye_food_0 = freeze(eye_food_0)
    # input = get_input_neuron(
    #     neuron=eye_food_0,
    #     terrain_map=minimap,
    #     food_positions=food_positions,
    #     fish_position=fish_position,
    # )
    # print(input)  # should be 0.1

    # # =======================================================
    # terrain_map = jnp.zeros((6, 6))
    # print(update_fish_position(
    #     curr_fish_position = jnp.array([1, 1]),
    #     move_attempt=jnp.array([-1, 1]),
    #     terrain_map=terrain_map,
    # ))
    # # =======================================================

    # # =======================================================
    # from jaxfish.data_classes import EIGHT_EYES, freeze, MINIMUM_BRAIN

    # seed = 0
    # length = 50
    # key = jax.random.key(seed)
    # food_key, fish_key, firing_key = jax.random.split(key, 3)
    # brain = freeze(MINIMUM_BRAIN)
    # n_neurons = len(brain.neurons)
    # firing_keys = jax.random.split(firing_key, (n_neurons, length))

    # terrain = Terrain(minimap_size=(6, 6))
    # minimap = generate_terrain_map(terrain)

    # food_positions_history = jnp.zeros((2, length, 2), dtype=jnp.int32)
    # food_position_history = food_positions_history.at[:, 0, :].set(jnp.array([[1, 1], [1, 4]]))

    # fish_position_history = jnp.zeros((length, 2), dtype=jnp.int32)
    # fish_position_history = fish_position_history.at[0].set(jnp.array([2, 3]))

    # firing_history = jnp.zeros((n_neurons, length), dtype=jnp.uint8)
    # psp_history = jnp.zeros((n_neurons, length), dtype=jnp.float32)

    # move_attempt = jnp.zeros(2, dtype=jnp.int32)

    # for neuron_idx, neuron in enumerate(brain.neurons):

    #     # evaluate neuron firing and move attempt
    #     is_firing, firing_history = evaluate_neuron(
    #         neuron_idx=neuron_idx,
    #         t=0,
    #         neuron=neuron,
    #         terrain_map=minimap,
    #         food_positions_history=food_positions_history,
    #         fish_position_history=fish_position_history,
    #         firing_history=firing_history,
    #         psp_history=psp_history,
    #         firing_keys=firing_keys
    #     )

    #     if is_firing and neuron.type == "muscle_frozen":
    #         move_attempt = move_attempt + jnp.array(neuron.step_motion)

    #     print(neuron)
    #     print(is_firing)
    #     print(move_attempt)
