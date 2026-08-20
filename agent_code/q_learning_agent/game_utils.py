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
    Calculate all tiles affected by a bomb.

    A stone wall blocks the explosion completely.
    A crate is destroyed, but blocks propagation behind it.
    """
    affected_tiles = [bomb_position]

    bomb_x, bomb_y = bomb_position
    width, height = field.shape

    for dx, dy in DIRECTIONS:
        for distance in range(1, s.BOMB_POWER + 1):
            x = bomb_x + dx * distance
            y = bomb_y + dy * distance

            if not (0 <= x < width and 0 <= y < height):
                break

            tile_value = field[x, y]

            # Stone wall: stop before including the wall.
            if tile_value == -1:
                break

            affected_tiles.append((x, y))

            # Crate: destroy it, then stop.
            if tile_value == 1:
                break

    return affected_tiles


def earliest_danger_times(
    game_state: dict,
) -> np.ndarray:
    """
    Calculate the earliest known explosion time for each tile.

    Values:
        np.inf: no currently known danger
        0: dangerous now
        1: dangerous in one step
        2: dangerous in two steps
        ...
    """
    field = game_state["field"]

    danger_times = np.full(
        field.shape,
        np.inf,
        dtype=np.float64,
    )

    # Existing explosions are dangerous immediately.
    explosion_map = game_state["explosion_map"]
    danger_times[explosion_map > 0] = 0.0

    # Predict the blast area of each active bomb.
    for bomb_position, timer in game_state["bombs"]:
        for position in blast_tiles(field, bomb_position):
            danger_times[position] = min(
                danger_times[position],
                float(timer),
            )

    return danger_times