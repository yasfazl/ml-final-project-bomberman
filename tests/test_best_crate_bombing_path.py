import numpy as np

from agent_code.q_learning_agent.game_utils import (
    best_crate_bombing_path,
)


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(field, position=(4, 4)):
    return {
        "field": field,
        "self": ("q_learning_agent", 0, True, position),
        "others": [],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "round": 1,
        "step": 1,
    }


def test_no_crate_target_exists():
    direction, distance, crate_count = best_crate_bombing_path(
        make_state(open_field())
    )

    assert direction is None
    assert distance is None
    assert crate_count == 0


def test_moves_one_step_to_hit_more_crates():
    field = open_field()

    # A bomb at (4, 4) hits only the crate at (4, 2).
    field[4, 2] = 1

    # Moving LEFT to (3, 4) creates a two-crate blast line.
    field[3, 2] = 1
    field[3, 6] = 1

    direction, distance, crate_count = best_crate_bombing_path(
        make_state(field)
    )

    assert direction == 3  # LEFT
    assert distance == 1
    assert crate_count == 2


def test_prefers_nearer_position_when_yield_is_equal():
    field = open_field()

    # The current tile already hits two crates.
    field[4, 2] = 1
    field[4, 6] = 1

    direction, distance, crate_count = best_crate_bombing_path(
        make_state(field)
    )

    assert direction is None
    assert distance == 0
    assert crate_count == 2


def test_respects_search_distance():
    field = open_field()
    field[4, 2] = 1
    field[1, 2] = 1
    field[1, 6] = 1

    # With radius zero, only the current position may be considered.
    direction, distance, crate_count = best_crate_bombing_path(
        make_state(field),
        max_search_distance=0,
    )

    assert direction is None
    assert distance == 0
    assert crate_count == 1