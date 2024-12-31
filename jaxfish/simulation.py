from jaxfish.data_classes import Terrain, FishForzen, BrainFrozen, SimulationFrozen
from jaxfish.utils import generate_terrain_map
import jax.numpy as jnp


def initiate_simulation(
    terrain: Terrain,
    simulation: SimulationFrozen,
    fish: FishForzen,
    brain: BrainFrozen,
):
    # generate terrain map
    terrain_map = generate_terrain_map(terrain)

    length = simulation.max_simulation_length

    # initiate food position history
    food_positions_history = jnp.zeros((length, simulation.food_num, 2), dtype=jnp.uint)

    # get initial food positions

    # initiate fish position history
    fish_position_history = jnp.zeros((length, 2), dtype=jnp.uint)

    # get initial fish position

    # initiate fish health history
    health_history = jnp.zeros(length, dtype=jnp.float)
    health_history = health_history.at[0].set(fish.start_health)

    # initiate neuron firing history
    firing_history = jnp.zeros((len(brain.neurons), length), dtype=jnp.uint8)

    # initiate neuron post-synaptic potential waveforms
    psp_waveforms = jnp.zeros((len(brain.neurons), length), dtype=float)

    return (
        terrain_map,
        food_positions_history,
        fish_position_history,
        health_history,
        firing_history,
        psp_waveforms,
    )


def step_simulation():
    pass


def save_simulation():
    pass
