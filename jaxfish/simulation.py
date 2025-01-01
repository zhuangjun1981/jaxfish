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


def initiate_simulation(
    terrain: Terrain,
    simulation: SimulationFrozen,
    fish: FishForzen,
    brain: BrainFrozen,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray]]:
    """ """
    length = simulation.max_simulation_length
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
    fish_position_history = fish_position_history.at[0].set(
        ut.get_starting_fish_position(terrain_map, fish_key)
    )

    # initiate food position history
    food_positions_history = jnp.zeros(
        (length, simulation.food_num, 2), dtype=jnp.int32
    )

    # update initial food positions
    food_positions_history = food_positions_history.at[0].set(
        ut.update_food_positions(
            terrain_map=terrain_map,
            fish_position=fish_position_history[0],
            food_positions=food_positions_history[0],
            eaten_food_positions=food_positions_history[0],
            rng=food_keys[0],
        )
    )

    # initiate fish health history
    health_history = jnp.zeros(length, dtype=jnp.float32)
    health_history = health_history.at[0].set(fish.start_health)

    # initiate neuron firing history
    firing_history = jnp.zeros((n_neurons, length), dtype=jnp.uint8)

    # generate psp waveforms
    psp_waveforms = ()

    # initiate neuron post-synaptic potential waveforms
    baselines = jnp.expand_dims(jnp.array([n.baseline_rate for n in brain.neurons]), 1)
    psp_history = jnp.ones((len(brain.neurons), length), dtype=float) * baselines

    return (
        terrain_map,
        food_positions_history,
        fish_position_history,
        health_history,
        firing_history,
        psp_waveforms,
        psp_history,
    )


def step_simulation():
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
        terrain_map,
        food_positions_history,
        fish_position_history,
        health_history,
        firing_history,
        psp_waveforms,
        psp_history,
    ) = _

    # print(terrain_map)
    # print(psp_history)
    print(fish_position_history)
