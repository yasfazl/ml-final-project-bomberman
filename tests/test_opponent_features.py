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


def make_state(
    position=(2, 4),
    opponents=(),
    coins=(),
):
    field = open_field()

    others = [
        (f"opponent_{index}", 0, True, opponent_position)
        for index, opponent_position in enumerate(opponents)
    ]

    return {
        "field": field,
        "self": ("q_learning_agent", 0, True, position),
        "others": others,
        "bombs": [],
        "coins": list(coins),
        "explosion_map": np.zeros_like(field),
        "round": 1,
        "step": 1,
    }


def test_no_opponent_features():
    features = state_to_features(make_state())

    assert FEATURE_DIM == 39
    assert features.shape == (39,)
    assert np.allclose(features[32:39], 0.0)


def test_nearby_opponent_features():
    features = state_to_features(
        make_state(opponents=[(6, 4)])
    )

    assert features[32] == 1.0
    assert features[33] == 1.0

    # RIGHT is direction index 1, therefore feature 35.
    assert features[35] == 1.0
    assert np.sum(features[34:38]) == 1.0

    # Three movements on a board with maximum distance 18.
    assert np.isclose(features[38], 3.0 / 18.0)


def test_coin_disables_opponent_pursuit():
    features = state_to_features(
        make_state(
            opponents=[(6, 4)],
            coins=[(2, 5)],
        )
    )

    # The opponent still exists, but coin collection has priority.
    assert features[32] == 1.0
    assert np.allclose(features[33:39], 0.0)