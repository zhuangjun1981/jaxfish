import jax
import jax.numpy as jnp
from jaxfish.data_classes import (
    Terrain,
    FishForzen,
    BrainFrozen,
    SimulationFrozen,
    MINIMUM_BRAIN,
    frozen,
)
import jaxfish.utils as ut
from functools import partial


@partial(jax.jit, static_argnames=["terrain", "simulation", "fish", "brain"])
def initiate_simulation(
    terrain: Terrain,
    simulation: SimulationFrozen,
    fish: FishForzen,
    brain: BrainFrozen,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray]]:
    """
    Initiate one simulation of one fish,
      1. generate terrain map
      2. generate random number generaters,
      3. allocate memory for various simulation cache
        - psp_waveforms for each connection
        - food_positions_history
        - fish_position_history
        - firing_history for each neuron
        - health_history
        - psp_hisory for each neuron
      4. initiate fish and food positions

    Args:
      terrain: Terrain dataclass
      simulation: SimulationFrozen dataclass
      fish: FishFrozen dataclass
      brain: BrainFrozen dataclass

    Return:
        firing_keys: jnp.ndarray, shape=(n_neurons, length), dtype=uint32 ("key<fry>")
        food_keys: jnp.ndarray, shape=(length,), dtype=uint32 ("key<fry>")
        terrain_map: jnp.ndarray, shape=(n_rows, n_cols), dtype=uint8
        food_positions_history,: jnp.ndarray, shape=(length, food_num, 2), dtype=int32
        fish_position_history, jnp.ndarray, shape=(length, 2), dtype=int32
        health_history: jnp.ndarray, shape=(length,), dtype=float32
        firing_history: jnp.ndarray, shape=(n_neurons, length), dtype=uint8
        psp_waveforms: tuple[jnp.ndarray], len=n_connections, each element is 1d, with dtyp float32
        psp_history: jnp.ndarray, shape=(n_neurons, length), dtype=float32
    """

    length = simulation.max_simulation_length
    food_num = simulation.food_num
    n_neurons = len(brain.neurons)

    # generate rngs
    main_key = jax.random.key(simulation.main_seed)
    food_key, fish_key, firing_key = jax.random.split(main_key, 3)

    food_keys = jax.random.split(food_key, length)
    firing_keys = jax.random.split(firing_key, (n_neurons, length))

    # generate terrain map
    terrain_map = jnp.array(ut.generate_terrain_map(terrain))

    # initiate fish position history
    fish_position_history = jnp.zeros((length, 2), dtype=jnp.int32)

    # update initial fish position
    starting_fish_position = ut.get_starting_fish_position(terrain_map, fish_key)
    fish_position_history = fish_position_history.at[0].set(starting_fish_position)

    # initiate food position history
    food_positions_history = jnp.zeros(
        (length, simulation.food_num, 2), dtype=jnp.int32
    )

    # update initial food positions
    food_map = jnp.array(terrain_map)
    fish_pixels = ut.get_fish_pixels(starting_fish_position)
    food_map = food_map.at[fish_pixels.T[0], fish_pixels.T[1]].set(1)
    food_positions_history = food_positions_history.at[0].set(
        ut.update_food_positions(
            food_map=food_map,
            food_positions=food_positions_history[0],
            is_eaten=jnp.ones(food_num),
            key=food_keys[0],
        )
    )

    # initiate fish health history
    health_history = jnp.zeros(length, dtype=jnp.float32)
    health_history = health_history.at[0].set(fish.start_health)

    # initiate neuron firing history
    firing_history = jnp.zeros((n_neurons, length), dtype=jnp.uint8)

    # generate psp waveforms
    psp_waveforms = []
    for connection in brain.connections:
        psp_waveforms.append(ut.generate_psp_waveform(connection))
    psp_waveforms = tuple(psp_waveforms)

    # initiate neuron post-synaptic potential waveforms
    baselines = jnp.expand_dims(jnp.array([n.baseline_rate for n in brain.neurons]), 1)
    psp_history = jnp.ones((len(brain.neurons), length), dtype=float) * baselines

    return (
        firing_keys,
        food_keys,
        terrain_map,
        food_positions_history,
        fish_position_history,
        health_history,
        firing_history,
        psp_waveforms,
        psp_history,
    )


def step_simulation(
    t: int,
    fish: FishForzen,
    brain: BrainFrozen,
    simulation: SimulationFrozen,
    firing_keys: jnp.ndarray,
    food_keys: jnp.ndarray,
    terrain_map: jnp.ndarray,
    food_positions_history: jnp.ndarray,
    fish_position_history: jnp.ndarray,
    health_history: jnp.ndarray,
    firing_history: jnp.ndarray,
    psp_waveforms: tuple[jnp.ndarray],
    psp_history: tuple[jnp.ndarray],
):
    """
    given the current terrain map, fish position, food positions,
    evaluate events in following steps:
      1. if fish's body overlaps with food, eat food
      2. if there is food eaten, spaw new food, otherwise keep old food, set food positions in next time point
      3. get input to the brain with terrain map and food position in next time point
      4. update neuron firing, psp history, and fish movement (jit compile this step?)
      5. move fish and set fish position in next time point
    """

    curr_health = health_history[t]
    if curr_health < 0.0:
        return

    curr_fish_position = fish_position_history[t]
    curr_food_positions = food_positions_history[t]

    # update food positions and find out number of food eaten
    updated_food_positions, eaten_food_num = ut.update_food_positions(
        terrain_map=terrain_map,
        fish_position=curr_fish_position,
        food_positions=curr_food_positions,
        rng=food_keys[t],
    )

    # set food positions at t + 1
    food_positions_history = food_positions_history.at[t + 1].set(
        updated_food_positions
    )

    # updata health
    curr_health = curr_health + eaten_food_num * fish.food_rate

    # get input to the eye

    # update firing of all neurons, get movement

    # update psp histories

    # set fish_position at t + 1

    # set health at t + 1
    curr_health = curr_health - fish.health_decay_rate
    health_history = health_history.at[t + 1].set(curr_health)
    pass


def save_simulation():
    pass


if __name__ == "__main__":
    _ = initiate_simulation(
        terrain=Terrain(),
        brain=frozen(MINIMUM_BRAIN),
        fish=FishForzen(),
        simulation=SimulationFrozen(
            simulation_ind=0,
            max_simulation_length=100,
        ),
    )

    (
        firing_keys,
        food_keys,
        terrain_map,
        food_positions_history,
        fish_position_history,
        health_history,
        firing_history,
        psp_waveforms,
        psp_history,
    ) = _

    # print(firing_keys)
    print(firing_keys[0].dtype)
    # print(terrain_map)
    # print(psp_history)
    # print(fish_position_history)
