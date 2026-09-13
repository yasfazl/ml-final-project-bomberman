import numpy as np

from agent_code.q_learning_agent.game_utils import (
    bomb_target_counts,
    earliest_danger_times,
    has_escape_route_after_bomb,
    nearest_safe_path,
    tile_is_safe_at_time,
)


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(field=None, position=(5, 5), bombs=None, others=None):
    if field is None:
        field = open_field()
    if bombs is None:
        bombs = []
    if others is None:
        others = []

    return {
        "field": field,
        "self": ("test", 0, True, position),
        "bombs": bombs,
        "others": others,
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_bomb_timer_allows_movement_before_explosion():
    assert tile_is_safe_at_time(3.0, 3)
    assert not tile_is_safe_at_time(3.0, 4)
    assert not tile_is_safe_at_time(3.0, 5)
    assert tile_is_safe_at_time(3.0, 6)


def test_exact_four_step_escape_is_found():
    # A one-tile corridor forces the agent to use all four moves:
    # three tiles through the blast range and one tile beyond it.
    field = np.full((11, 11), -1, dtype=int)
    field[1:10, 5] = 0

    state = make_state(
        field=field,
        bombs=[((5, 5), 3)],
    )

    direction, distance = nearest_safe_path(state)

    assert direction == 1
    assert distance == 4


def test_has_escape_route_after_bomb_on_open_board():
    assert has_escape_route_after_bomb(make_state())


def test_no_escape_route_from_closed_area():
    field = np.full((11, 11), -1, dtype=int)
    field[5, 5] = 0
    field[5, 4] = 0

    assert not has_escape_route_after_bomb(
        make_state(field=field)
    )


def test_chain_reaction_uses_earlier_timer():
    state = make_state(
        bombs=[
            ((3, 5), 1),
            ((6, 5), 3),
        ]
    )

    danger = earliest_danger_times(state)

    assert danger[8, 5] == 1.0


def test_bomb_target_counts_crates_and_opponents():
    field = open_field()
    field[5, 3] = 1
    others = [("enemy", 0, True, (7, 5))]

    crates, opponents = bomb_target_counts(
        make_state(field=field, others=others)
    )

    assert crates == 1
    assert opponents == 1