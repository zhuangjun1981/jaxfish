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
    step_motion: tuple = (0, 0)


@dataclass(frozen=True)
class NeuronFrozen:
    type: str = "neuron_frozen"
    refractory_period: float = 1.2
    baseline_rate: float = 0.0
    step_motion: tuple = (0, 0)


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
    step_motion: tuple = (0, 0)


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
    step_motion: tuple = (0, 0)


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
    psp_waveform_length: int = 100  # maximum length to save unit psp waveforms


def freeze(var: Union[Fish, Neuron, Eye, Muscle, Connection, Brain]):
    if var.type.endswith("_frozen"):
        return var
    elif var.type == "brain":
        return BrainFrozen(
            neurons=tuple(freeze(n) for n in var.neurons),
            connection_directions=var.connection_directions,
            connections=tuple(freeze(c) for c in var.connections),
        )
    else:
        frozen_class = var.type.split("_")[0]
        frozen_class = frozen_class[0].upper() + frozen_class[1:] + "Frozen"
        frozen_class_type = globals()[frozen_class]
        var_dict = dict(var.__dict__)
        del var_dict["type"]
        return frozen_class_type(**var_dict)
