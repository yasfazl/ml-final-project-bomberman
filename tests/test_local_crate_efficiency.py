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


def make_state(field):
    return {
        "field": field,
        "self": ("q_learning_agent", 0, True, (4, 4)),
        "others": [],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "round": 1,
        "step": 1,
    }


def test_default_search_avoids_two_step_detour():
    field = open_field()

    # Bombing now reaches one crate.
    field[4, 2] = 1

    # Two steps LEFT would reach a position that targets two crates, but the
    # local default must not take that detour.
    field[2, 2] = 1
    field[2, 6] = 1

    direction, distance, crate_count = best_crate_bombing_path(
        make_state(field)
    )

    assert direction is None
    assert distance == 0
    assert crate_count == 1


def test_explicit_radius_can_still_find_two_step_position():
    field = open_field()
    field[4, 2] = 1
    field[2, 2] = 1
    field[2, 6] = 1

    direction, distance, crate_count = best_crate_bombing_path(
        make_state(field),
        max_search_distance=2,
    )

    assert direction == 3  # LEFT
    assert distance == 2
    assert crate_count == 2
