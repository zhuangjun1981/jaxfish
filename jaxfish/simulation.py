import jax
import jax.numpy as jnp
from jaxfish.data_classes import (
    Terrain,
    FishForzen,
    BrainFrozen,
    SimulationFrozen,
)
import jaxfish.utils as ut
from functools import partial


@partial(jax.jit, static_argnames=("terrain", "simulation", "fish", "brain"))
def initiate_simulation(
    terrain: Terrain,
    simulation: SimulationFrozen,
    fish: FishForzen,
    brain: BrainFrozen,
) -> tuple[jnp.ndarray]:
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
    n_connections = len(brain.connections)

    # generate rngs
    main_key = jax.random.key(simulation.main_seed)
    food_key, fish_key, firing_key = jax.random.split(main_key, 3)

    food_keys = jax.random.split(food_key, length)
    firing_keys = jax.random.split(firing_key, (n_neurons, length))

    # generate terrain map
    terrain_map = ut.generate_terrain_map(terrain)

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

    # generate psp waveforms,
    psp_waveforms = jnp.zeros(
        (n_connections, simulation.psp_waveform_length), dtype=jnp.float32
    )
    for c_i, connection in enumerate(brain.connections):
        psp_waveforms = psp_waveforms.at[c_i].set(
            ut.generate_psp_waveform(
                latency=connection.latency,
                rise_time=connection.rise_time,
                decay_time=connection.decay_time,
                amplitude=connection.amplitude,
                psp_waveform_length=simulation.psp_waveform_length,
            )
        )

    # initiate neuron post-synaptic potential waveforms, this part is jit compatable, if brain ahd simulation is set to be static
    psp_history = jnp.zeros((n_neurons, length), dtype=jnp.float32)

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


@partial(jax.jit, static_argnames=["fish", "brain", "simulation"])
def step_simulation(
    simulation_params,
    fish: FishForzen,
    brain: BrainFrozen,
    simulation: SimulationFrozen,
):
    """
    given the current terrain map, fish position, food positions,
    evaluate events in following steps:
      1. if fish's body overlaps with food, eat food
      2. if there is food eaten, spaw new food, otherwise keep old food, set food positions in next time point
      3. get input to the brain with terrain map and food position in next time point
      4. update neuron firing, psp history, and fish movement (jit compile this step?)
      5. move fish and set fish position in next time point

    Args:
      simulation_params:
        t: int,
        firing_keys: jnp.ndarray,
        food_keys: jnp.ndarray,
        terrain_map: jnp.ndarray,
        food_positions_history: jnp.ndarray,
        fish_position_history: jnp.ndarray,
        health_history: jnp.ndarray,
        firing_history: jnp.ndarray,
        psp_waveforms: jnp.ndarray,
        psp_history: jnp.ndarray,
      fish: FishForzen,
      brain: BrainFrozen,
      simulation: SimulationFrozen,
    """

    (
        t,
        firing_keys,
        food_keys,
        terrain_map,
        food_positions_history,
        fish_position_history,
        health_history,
        firing_history,
        psp_waveforms,
        psp_history,
    ) = simulation_params

    curr_health = health_history[t]
    curr_fish_position = fish_position_history[t]
    curr_food_positions = food_positions_history[t]

    # update food positions and find out number of food eaten
    updated_food_positions, eaten_food_num = ut.update_food_positions_in_simulation(
        terrain_map=terrain_map,
        fish_position=curr_fish_position,
        food_positions=curr_food_positions,
        key=food_keys[t],
    )

    # set food positions at t + 1
    food_positions_history = food_positions_history.at[t + 1].set(
        updated_food_positions
    )

    # updata health
    curr_health = curr_health + eaten_food_num * fish.food_rate

    # update firing of all neurons, get movement attempt
    move_attempt = jnp.zeros(2, dtype=jnp.int32)

    for neuron_idx, neuron in enumerate(brain.neurons):
        # evaluate neuron firing and move attempt
        is_firing, firing_history = ut.evaluate_neuron(
            neuron_idx=neuron_idx,
            t=t,
            neuron=neuron,
            terrain_map=terrain_map,
            food_positions_history=food_positions_history,
            fish_position_history=fish_position_history,
            firing_history=firing_history,
            psp_history=psp_history,
            firing_keys=firing_keys,
        )

        # update move_attempt
        move_attempt = jax.lax.cond(
            pred=is_firing & (neuron.type == "muscle_frozen"),
            true_fun=lambda x: x + jnp.array(neuron.step_motion),
            false_fun=lambda x: x,
            operand=move_attempt,
        )

        # update psp histories
        for pre_idx, post_idx in brain.connection_directions:
            if pre_idx == neuron_idx:

                def true_fn(operand):
                    t, pre_idx, post_idx, psp_waveforms, psp_history = operand

                    psp_history = ut.update_psp_history(
                        t=t,
                        pre_neuron_idx=pre_idx,
                        post_neuron_idx=post_idx,
                        psp_waveforms=psp_waveforms,
                        psp_history=psp_history,
                    )
                    return t, pre_idx, post_idx, psp_waveforms, psp_history

                _, _, _, _, psp_history = jax.lax.cond(
                    pred=is_firing,
                    true_fun=true_fn,
                    false_fun=lambda x: x,
                    operand=(t, pre_idx, post_idx, psp_waveforms, psp_history),
                )

    # set fish_position at t + 1
    updated_fish_position = ut.update_fish_position(
        curr_fish_position=curr_fish_position,
        move_attempt=move_attempt,
        terrain_map=terrain_map,
    )
    fish_position_history = fish_position_history.at[t + 1].set(updated_fish_position)

    # evaluate land penalty
    land_overlap = ut.get_land_overlap(
        terrain_map=terrain_map, fish_position=curr_fish_position
    )
    curr_health = curr_health - land_overlap * fish.land_penalty_rate

    # evaluate movement penalty
    curr_health = curr_health - jnp.sum(jnp.abs(move_attempt)) * fish.move_penalty_rate

    # set health at t + 1
    curr_health = curr_health - fish.health_decay_rate
    health_history = health_history.at[t + 1].set(curr_health)

    return (
        t + 1,
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


@partial(jax.jit, static_argnames=["fish", "brain", "simulation"])
def cond_fn_out(
    simulation_params: tuple[jnp.ndarray],
    fish: FishForzen,
    brain: BrainFrozen,
    simulation: SimulationFrozen,
):
    """
    condition check for the simulation while loop
    if t is less than simulation_max_lenght and if fish's health >=0
    continue simulation
    """
    (
        t,
        _,
        _,
        _,
        _,
        _,
        health_history,
        _,
        _,
        _,
    ) = simulation_params

    # if t is not reachin the end of the simulation and fish is not dead
    return (t < simulation.max_simulation_length - 1) & (health_history[t] >= 0.0)


@partial(jax.jit, static_argnames=("terrain", "simulation", "fish", "brain"))
def run_simulation(
    terrain: Terrain,
    simulation: SimulationFrozen,
    fish: FishForzen,
    brain: BrainFrozen,
) -> tuple[jnp.ndarray]:
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
    ) = initiate_simulation(
        terrain=terrain,
        simulation=simulation,
        fish=fish,
        brain=brain,
    )

    init_params = (
        0,
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

    def cond_fn_in(simulation_params):
        return cond_fn_out(
            simulation_params, fish=fish, brain=brain, simulation=simulation
        )

    def body_fn(simulation_params):
        return step_simulation(
            simulation_params, fish=fish, brain=brain, simulation=simulation
        )

    params = jax.lax.while_loop(cond_fn_in, body_fn, init_params)

    return params


def save_simulation():
    pass


if __name__ == "__main__":
    from jaxfish.data_classes import MINIMUM_BRAIN, freeze, SimulationFrozen

    terrain = Terrain()
    fish = FishForzen()
    brain = freeze(MINIMUM_BRAIN)
    simulation = SimulationFrozen(
        simulation_ind=0,
        max_simulation_length=50,
    )

    simulation_result = run_simulation(
        terrain=terrain,
        simulation=simulation,
        fish=fish,
        brain=brain,
    )

    (
        _,
        _,
        _,
        terrain_map,
        food_positions_history,
        fish_position_history,
        health_history,
        firing_history,
        _,
        psp_history,
    ) = simulation_result

    print(f"\n{food_positions_history=}")
    print(f"\n{fish_position_history=}")
    print(f"\n{health_history=}")
    print(f"\n{firing_history=}")
    print(f"\n{psp_history=}")
