from __future__ import annotations

import numpy as np

import settings as s


DIRECTIONS = (
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (0, 1),   # DOWN
    (-1, 0),  # LEFT
)


def blast_tiles(
    field: np.ndarray,
    bomb_position: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Return all tiles affected by a bomb.

    Stone walls stop the explosion without being included.
    Crates are included, but stop the explosion afterward.
    """
    bomb_x, bomb_y = bomb_position
    width, height = field.shape

    affected_tiles = [(bomb_x, bomb_y)]

    for dx, dy in DIRECTIONS:
        for distance in range(1, s.BOMB_POWER + 1):
            position = (
                bomb_x + dx * distance,
                bomb_y + dy * distance,
            )
            x, y = position

            if not (0 <= x < width and 0 <= y < height):
                break

            tile = field[position]

            if tile == -1:  # stone wall
                break

            affected_tiles.append(position)

            if tile == 1:  # crate
                break

    return affected_tiles

def earliest_danger_times(
    game_state: dict,
) -> np.ndarray:
    """
    Return the earliest explosion time for every board tile.

    np.inf means the tile is currently not threatened.
    0 means an explosion is already active.
    """
    field = game_state["field"]

    danger_times = np.full(
        field.shape,
        np.inf,
        dtype=np.float64,
    )

    explosion_map = game_state.get("explosion_map")

    if explosion_map is not None:
        danger_times[
            np.asarray(explosion_map) > 0
        ] = 0.0

    for bomb_position, timer in game_state.get("bombs", []):
        for position in blast_tiles(field, bomb_position):
            danger_times[position] = min(
                danger_times[position],
                float(timer),
            )

    return danger_times