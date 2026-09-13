import numpy as np

from agent_code.q_learning_agent.callbacks import (
    FEATURE_DIM,
    state_to_features,
)


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(field=None, bombs=None):
    if field is None:
        field = open_field()

    if bombs is None:
        bombs = []

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test_agent", 0, True, (4, 4)),
        "others": [],
        "bombs": bombs,
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_safe_board_features():
    features = state_to_features(make_state())

    assert FEATURE_DIM == 39
    assert features.shape == (FEATURE_DIM,)
    assert np.allclose(features[11:17], 0.0)


def test_bomb_danger_features():
    features = state_to_features(
        make_state(bombs=[((4, 2), 1)])
    )

    # Current tile, UP, DOWN, and WAIT are inside the blast line.
    assert np.isclose(features[11], 0.5)
    assert np.isclose(features[12], 0.5)
    assert np.isclose(features[13], 0.0)
    assert np.isclose(features[14], 0.5)
    assert np.isclose(features[15], 0.0)
    assert np.isclose(features[16], 0.5)


def test_blocked_movement_is_marked_unsafe():
    field = open_field()
    field[4, 3] = -1

    features = state_to_features(make_state(field=field))

    assert features[12] == 1.0