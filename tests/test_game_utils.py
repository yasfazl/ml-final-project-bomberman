import numpy as np

from agent_code.q_learning_agent.game_utils import (
    blast_tiles,
    earliest_danger_times,
    nearest_opponent_path,
)


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)

    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return field


def test_blast_extends_in_four_directions():
    field = open_field()

    result = set(blast_tiles(field, (4, 4)))

    expected = {
        (4, 4),
        (4, 3), (4, 2), (4, 1),
        (5, 4), (6, 4), (7, 4),
        (4, 5), (4, 6), (4, 7),
        (3, 4), (2, 4), (1, 4),
    }

    assert result == expected


def test_stone_wall_blocks_explosion():
    field = open_field()
    field[4, 2] = -1

    result = set(blast_tiles(field, (4, 4)))

    assert (4, 3) in result
    assert (4, 2) not in result
    assert (4, 1) not in result


def test_crate_is_hit_and_blocks_explosion():
    field = open_field()
    field[6, 4] = 1

    result = set(blast_tiles(field, (4, 4)))

    assert (5, 4) in result
    assert (6, 4) in result
    assert (7, 4) not in result



def make_state(field, bombs=None, explosion_map=None):
    if bombs is None:
        bombs = []

    if explosion_map is None:
        explosion_map = np.zeros_like(field)

    return {
        "field": field,
        "bombs": bombs,
        "explosion_map": explosion_map,
    }


def test_bomb_creates_danger_times():
    field = open_field()

    state = make_state(
        field,
        bombs=[((4, 2), 1)],
    )

    danger = earliest_danger_times(state)

    assert danger[4, 2] == 1.0
    assert danger[4, 4] == 1.0
    assert np.isinf(danger[5, 5])


def test_earliest_bomb_timer_is_used():
    field = open_field()

    state = make_state(
        field,
        bombs=[
            ((4, 2), 3),
            ((4, 6), 1),
        ],
    )

    danger = earliest_danger_times(state)

    assert danger[4, 4] == 1.0


def test_active_explosion_has_zero_danger_time():
    field = open_field()
    explosion_map = np.zeros_like(field)
    explosion_map[3, 3] = 1

    state = make_state(
        field,
        explosion_map=explosion_map,
    )

    danger = earliest_danger_times(state)

    assert danger[3, 3] == 0.0


def test_nearest_opponent_path_points_to_adjacent_tile():
    field = open_field(size=11)
    state = {
        "field": field,
        "self": ("q_learning_agent", 0, True, (5, 5)),
        "others": [("opponent", 0, True, (9, 5))],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "round": 1,
        "step": 1,
    }

    direction, distance = nearest_opponent_path(state)

    assert direction == 1  # RIGHT
    assert distance == 3