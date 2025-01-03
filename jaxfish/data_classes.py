import jax.numpy as jnp
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Terrain:
    should_use_minimap: bool = True
    minimap_size: tuple = (9, 9)
    minimap_margin: int = 1


@dataclass
class Fish:
    name: str = "unknown"
    mother_name: str = "unknown"
    max_health: float = 50.0
    start_health: float = 10.0
    food_rate: float = 20
    land_penalty_rate: float = 5.0
    move_penalty_rate: float = 0.001
    health_decay_rate: float = 0.01
    type: str = "fish"


@dataclass(frozen=True)
class FishForzen:
    name: str = "unknown"
    mother_name: str = "unknown"
    max_health: float = 50.0
    start_health: float = 10.0
    food_rate: float = 20
    land_penalty_rate: float = 5.0
    move_penalty_rate: float = 0.001
    health_decay_rate: float = 0.01
    type: str = "fish_frozen"


@dataclass
class Neuron:
    type: str = "neuron"
    refractory_period: float = 1.2
    baseline_rate: float = 0.0


@dataclass(frozen=True)
class NeuronFrozen:
    type: str = "neuron_frozen"
    refractory_period: float = 1.2
    baseline_rate: float = 0.0


@dataclass
class Eye:
    eye_direction: str
    rf_positions: tuple[
        tuple[int]
    ]  # row and col positions for each receptive field pixel location relative to fish's body center
    rf_weights: tuple[float] = (
        0.1,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
    )  # weight of each receptive field pixel location
    type: str = "eye"
    input_type: str = "terrain"
    gain: float = 1.0
    refractory_period: float = 1.2
    baseline_rate: float = 0.0


@dataclass(frozen=True)
class EyeFrozen:
    eye_direction: str
    rf_positions: tuple[
        tuple[int]
    ]  # row and col positions for each receptive field pixel location relative to fish's body center
    rf_weights: tuple[float] = (
        0.1,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
    )  # weight of each receptive field pixel location
    type: str = "eye_frozen"
    input_type: str = "terrain"
    gain: float = 1.0
    refractory_period: float = 1.2
    baseline_rate: float = 0.0


@dataclass
class Muscle:
    direction: str
    step_motion: tuple
    type: str = "muscle"
    refractory_period: float = 1.2
    baseline_rate: float = 0.0


@dataclass(frozen=True)
class MuscleFrozen:
    direction: str
    step_motion: tuple
    type: str = "muscle_frozen"
    refractory_period: float = 1.2
    baseline_rate: float = 0.0


@dataclass
class Connection:
    amplitude: float = 0.001
    latency: int = 3
    rise_time: int = 5
    decay_time: int = 10
    type: str = "connection"


@dataclass(frozen=True)
class ConnectionFrozen:
    amplitude: float = 0.001
    latency: int = 3
    rise_time: int = 5
    decay_time: int = 10
    type: str = "connection_frozen"


@dataclass
class Brain:
    neurons: tuple[Neuron, Eye, Muscle]
    connection_directions: tuple[
        tuple[int]
    ]  # (n_connections, 2), int, each row is (pre_neuron_index, pos_neuron_index)
    connections: tuple[Connection]
    type: str = "brain"


@dataclass(frozen=True)
class BrainFrozen:
    neurons: tuple[NeuronFrozen, EyeFrozen, MuscleFrozen]
    connection_directions: tuple[tuple[int]]
    connections: tuple[ConnectionFrozen]
    type: str = "brain_frozen"


@dataclass(frozen=True)
class SimulationFrozen:
    simulation_ind: int
    max_simulation_length: int = 20000
    food_num: int = 1
    main_seed: int = 42
    psp_waveform_length: int = 100  # maximum length to save unit psp waveforms


EIGHT_EYES = {
    "southeast": Eye(
        eye_direction="southeast",
        rf_positions=(
            (1, 1, 2, 2, 2, 3, 3),
            (1, 2, 2, 1, 3, 3, 2),
        ),
    ),
    "northwest": Eye(
        eye_direction="northwest",
        rf_positions=(
            (-1, -1, -2, -2, -2, -3, -3),
            (-1, -2, -2, -1, -3, -3, -2),
        ),
    ),
    "northeast": Eye(
        eye_direction="northeast",
        rf_positions=(
            (-1, -1, -2, -2, -2, -3, -3),
            (1, 2, 2, 1, 3, 3, 2),
        ),
    ),
    "southwest": Eye(
        eye_direction="southwest",
        rf_positions=(
            (1, 1, 2, 2, 2, 3, 3),
            (-1, -2, -2, -1, -3, -3, -2),
        ),
    ),
    "north": Eye(
        eye_direction="north",
        rf_positions=(
            (-1, -2, -2, -2, -3, -3, -3),
            (0, -1, 0, 1, -1, 0, 1),
        ),
    ),
    "south": Eye(
        eye_direction="south",
        rf_positions=(
            (1, 2, 2, 2, 3, 3, 3),
            (0, -1, 0, 1, -1, 0, 1),
        ),
    ),
    "east": Eye(
        eye_direction="east",
        rf_positions=(
            (0, -1, 0, 1, -1, 0, 1),
            (1, 2, 2, 2, 3, 3, 3),
        ),
    ),
    "west": Eye(
        eye_direction="west",
        rf_positions=(
            (0, -1, 0, 1, -1, 0, 1),
            (-1, -2, -2, -2, -3, -3, -3),
        ),
    ),
}


FOUR_MUSCLES = {
    "north": Muscle(direction="north", step_motion=(-1, 0)),
    "south": Muscle(direction="south", step_motion=(1, 0)),
    "east": Muscle(direction="east", step_motion=(0, 1)),
    "west": Muscle(direction="west", step_motion=(0, -1)),
}


MINIMUM_BRAIN = Brain(
    neurons=(EIGHT_EYES["north"], FOUR_MUSCLES["north"]),
    connection_directions=((0, 1),),
    connections=(Connection(),),
)


def Freeze(var: Union[Fish, Neuron, Eye, Muscle, Connection, Brain]):
    if var.type.endswith("_frozen"):
        return var
    elif var.type == "brain":
        return BrainFrozen(
            neurons=tuple(frozen(n) for n in var.neurons),
            connection_directions=var.connection_directions,
            connections=tuple(frozen(c) for c in var.connections),
        )
    else:
        frozen_class = var.type.split("_")[0]
        frozen_class = frozen_class[0].upper() + frozen_class[1:] + "Frozen"
        frozen_class_type = globals()[frozen_class]
        var_dict = dict(var.__dict__)
        del var_dict["type"]
        return frozen_class_type(**var_dict)


if __name__ == "__main__":
    print(EIGHT_EYES["north"])

    eye = EIGHT_EYES["north"]
    eye_frozen = frozen(eye)
    print(eye_frozen)

    brain = MINIMUM_BRAIN
    brain_frozen = frozen(brain)
    print(brain_frozen)
