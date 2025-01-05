from jaxfish.data_classes import Eye, Muscle, Connection, Brain


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


def get_eight_eye_no_hidden_brain(
    eye_baseline_rate: float = 0.1,
    eye_gain: float = 1.0,
    muscle_baseline_rate: float = 0.5,
    connection_amplitude: float = 0.1,
    connection_latency: int = 3,
    connection_rise_time: int = 5,
    connection_decay_time: int = 10,
):
    neurons = []
    eye_keys = sorted(list(EIGHT_EYES.keys()))
    for eye_key in eye_keys:
        curr_eye = EIGHT_EYES[eye_key]
        curr_eye_terr = Eye(
            eye_direction=curr_eye.eye_direction,
            rf_positions=curr_eye.rf_positions,
            rf_weights=curr_eye.rf_weights,
            input_type="terrain",
            gain=eye_gain,
            baseline_rate=eye_baseline_rate,
        )

        curr_eye_food = Eye(
            eye_direction=curr_eye.eye_direction,
            rf_positions=curr_eye.rf_positions,
            rf_weights=curr_eye.rf_weights,
            input_type="food",
            gain=eye_gain,
            baseline_rate=eye_baseline_rate,
        )

        neurons.extend([curr_eye_terr, curr_eye_food])

    muscle_keys = sorted(list(FOUR_MUSCLES.keys()))
    for muscle_key in muscle_keys:
        curr_muscle = Muscle(
            direction=FOUR_MUSCLES[muscle_key].direction,
            step_motion=FOUR_MUSCLES[muscle_key].step_motion,
            baseline_rate=muscle_baseline_rate,
        )
        neurons.append(curr_muscle)

    eye_inds = list(range(16))
    muscle_inds = list(range(16, 20))

    connection_directions = tuple(
        [(pre_ind, post_ind) for post_ind in muscle_inds for pre_ind in eye_inds]
    )

    connections = [
        Connection(
            amplitude=connection_amplitude,
            latency=connection_latency,
            rise_time=connection_rise_time,
            decay_time=connection_decay_time,
        )
        for _ in connection_directions
    ]

    return Brain(
        neurons=neurons,
        connection_directions=connection_directions,
        connections=connections,
    )


BRAIN_EIGHT_EYE_NO_HIDDEN = get_eight_eye_no_hidden_brain()


if __name__ == "__main__":
    from jaxfish.data_classes import freeze

    # print(EIGHT_EYES["north"])

    # eye = EIGHT_EYES["north"]
    # eye_frozen = freeze(eye)
    # print(eye_frozen)

    # brain = MINIMUM_BRAIN
    # brain_frozen = freeze(brain)
    # print(brain_frozen)

    brain = get_eight_eye_no_hidden_brain()
    print(brain)
