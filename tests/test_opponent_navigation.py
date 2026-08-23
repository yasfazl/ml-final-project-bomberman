import numpy as np

from agent_code.q_learning_agent.game_utils import (
    nearest_opponent_path,
)


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(
    position=(2, 4),
    opponents=(),
    bombs=(),
    field=None,
):
    if field is None:
        field = open_field()

    others = [
        (f"opponent_{index}", 0, True, opponent_position)
        for index, opponent_position in enumerate(opponents)
    ]

    return {
        "field": field,
        "self": ("q_learning_agent", 0, True, position),
        "others": others,
        "bombs": list(bombs),
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "round": 1,
        "step": 1,
    }


def test_no_opponent_path_without_opponents():
    direction, distance = nearest_opponent_path(
        make_state()
    )

    assert direction is None
    assert distance is None


def test_path_toward_opponent():
    direction, distance = nearest_opponent_path(
        make_state(opponents=[(6, 4)])
    )

    # Reach (5, 4), directly beside the opponent.
    assert direction == 1  # RIGHT
    assert distance == 3


def test_already_adjacent_to_opponent():
    direction, distance = nearest_opponent_path(
        make_state(
            position=(3, 4),
            opponents=[(4, 4)],
        )
    )

    assert direction is None
    assert distance == 0


def test_unreachable_opponent():
    field = open_field()

    # A complete vertical stone wall separates the two sides.
    field[4, 1:8] = -1

    direction, distance = nearest_opponent_path(
        make_state(
            field=field,
            opponents=[(6, 4)],
        )
    )

    assert direction is None
    assert distance is None