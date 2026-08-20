import numpy as np

from agent_code.q_learning_agent.game_utils import (
    blast_tiles,
    earliest_danger_times,
)


def create_empty_field(size: int = 9) -> np.ndarray:
    """
    Create an empty field surrounded by stone walls.
    """
    field = np.zeros((size, size), dtype=int)

    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return field


def test_open_blast_has_expected_range():
    field = create_empty_field()
    bomb_position = (4, 4)

    affected = set(blast_tiles(field, bomb_position))

    assert bomb_position in affected

    assert (4, 1) in affected
    assert (4, 7) in affected
    assert (1, 4) in affected
    assert (7, 4) in affected

    assert len(affected) == 13


def test_stone_wall_blocks_explosion():
    field = create_empty_field()
    bomb_position = (4, 4)

    field[4, 2] = -1

    affected = set(blast_tiles(field, bomb_position))

    assert (4, 3) in affected
    assert (4, 2) not in affected
    assert (4, 1) not in affected


def test_crate_is_destroyed_and_blocks_explosion():
    field = create_empty_field()
    bomb_position = (4, 4)

    field[6, 4] = 1

    affected = set(blast_tiles(field, bomb_position))

    assert (5, 4) in affected
    assert (6, 4) in affected
    assert (7, 4) not in affected


def test_danger_map_uses_bomb_timer():
    field = create_empty_field()
    explosion_map = np.zeros_like(field)

    game_state = {
        "field": field,
        "bombs": [((4, 4), 2)],
        "explosion_map": explosion_map,
    }

    danger_times = earliest_danger_times(game_state)

    assert danger_times[4, 4] == 2
    assert danger_times[4, 2] == 2
    assert np.isinf(danger_times[2, 2])


def test_existing_explosion_is_dangerous_now():
    field = create_empty_field()
    explosion_map = np.zeros_like(field)
    explosion_map[2, 3] = 1

    game_state = {
        "field": field,
        "bombs": [],
        "explosion_map": explosion_map,
    }

    danger_times = earliest_danger_times(game_state)

    assert danger_times[2, 3] == 0