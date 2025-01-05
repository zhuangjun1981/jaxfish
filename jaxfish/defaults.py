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


if __name__ == "__main__":
    from jaxfish.data_classes import freeze

    print(EIGHT_EYES["north"])

    eye = EIGHT_EYES["north"]
    eye_frozen = freeze(eye)
    print(eye_frozen)

    brain = MINIMUM_BRAIN
    brain_frozen = freeze(brain)
    print(brain_frozen)
