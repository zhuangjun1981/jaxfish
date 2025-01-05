import numpy as np
import matplotlib.pyplot as plt


LAND_RGB = np.array([46, 204, 113], dtype=int)
SEA_RGB = np.array([52, 152, 219], dtype=int)
FISH_RGB = np.array([241, 196, 15], dtype=int)
FOOD_RGB = np.array([157, 32, 45], dtype=int)


def get_terrain_map_rgb(terrain_map_binary):
    terrain_map_rgb = np.zeros(
        (terrain_map_binary.shape[0], terrain_map_binary.shape[1], 3), dtype=int
    )
    land_rgb = LAND_RGB
    sea_rgb = SEA_RGB
    terrain_map_rgb[terrain_map_binary == 1, :] = land_rgb
    terrain_map_rgb[terrain_map_binary == 0, :] = sea_rgb

    return terrain_map_rgb


def add_fish_rgb(terrain_map_rgb, body_position):
    fish_rgb = FISH_RGB
    show_map_rgb = np.array(terrain_map_rgb)
    show_map_rgb[
        body_position[0] - 1 : body_position[0] + 2,
        body_position[1] - 1 : body_position[1] + 2,
        0,
    ] = fish_rgb[0]
    show_map_rgb[
        body_position[0] - 1 : body_position[0] + 2,
        body_position[1] - 1 : body_position[1] + 2,
        1,
    ] = fish_rgb[1]
    show_map_rgb[
        body_position[0] - 1 : body_position[0] + 2,
        body_position[1] - 1 : body_position[1] + 2,
        2,
    ] = fish_rgb[2]
    return show_map_rgb


def add_foods_rgb(show_map_rgb, food_poss):
    food_rgb = FOOD_RGB
    for food_pos in food_poss:
        show_map_rgb[food_pos[0], food_pos[1], :] = food_rgb
    return show_map_rgb


def get_map_rgb(t, simulation_result):
    terrain_map = simulation_result[3]
    food_positions = simulation_result[4][t]
    fish_position = simulation_result[5][t]

    map_rgb = get_terrain_map_rgb(terrain_map)
    map_rgb = add_fish_rgb(map_rgb, fish_position)
    map_rgb = add_foods_rgb(map_rgb, food_positions)

    return map_rgb


if __name__ == "__main__":
    simulation_result = (
        None,
        None,
        None,
        np.zeros((6, 6), dtype=np.uint8),
        np.zeros((1, 1, 2), dtype=np.int32),
        np.ones((1, 2), dtype=np.int32) * 3,
        np.ones((1,)) * 100,
    )

    map_rgb = get_map_rgb(0, simulation_result)
    plt.imshow(map_rgb)
    plt.show()
