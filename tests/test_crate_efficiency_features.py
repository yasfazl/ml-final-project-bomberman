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


def make_state(field, coins=(), bombs=()):
    return {
        "field": field,
        "self": ("q_learning_agent", 0, True, (4, 4)),
        "others": [],
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": np.zeros_like(field),
        "round": 1,
        "step": 1,
    }


def efficient_crate_field():
    field = open_field()
    field[4, 2] = 1
    field[3, 2] = 1
    field[3, 6] = 1
    return field


def test_no_crates_produce_no_efficiency_features():
    features = state_to_features(make_state(open_field()))

    assert FEATURE_DIM == 39
    assert features.shape == (FEATURE_DIM,)
    assert np.allclose(features[32:39], 0.0)


def test_better_position_features_point_left():
    features = state_to_features(
        make_state(efficient_crate_field())
    )

    assert features[32] == 1.0

    # Feature 33=UP, 34=RIGHT, 35=DOWN, 36=LEFT.
    assert features[36] == 1.0
    assert np.sum(features[33:37]) == 1.0

    assert np.isclose(features[37], 1.0 / 18.0)
    assert np.isclose(features[38], 2.0 / 4.0)


def test_visible_coin_disables_efficiency_features():
    features = state_to_features(
        make_state(
            efficient_crate_field(),
            coins=[(4, 5)],
        )
    )

    assert np.allclose(features[32:39], 0.0)


def test_current_best_position_has_yield_without_move_direction():
    field = open_field()
    field[4, 2] = 1
    field[4, 6] = 1

    features = state_to_features(make_state(field))

    assert features[32] == 0.0
    assert np.allclose(features[33:38], 0.0)
    assert np.isclose(features[38], 2.0 / 4.0)