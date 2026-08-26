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


def make_state(field, coins=(), bombs=(), others=()):
    return {
        "field": field,
        "self": ("q_learning_agent", 0, True, (4, 4)),
        "others": list(others),
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

    assert FEATURE_DIM == 45
    assert features.shape == (FEATURE_DIM,)
    assert np.allclose(features[32:39], 0.0)
    assert np.allclose(features[39:45], 0.0)


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


def test_opponent_target_disables_efficiency_features():
    features = state_to_features(
        make_state(
            efficient_crate_field(),
            others=[("opponent", 0, True, (4, 3))],
        )
    )

    assert np.allclose(features[32:39], 0.0)


def test_distant_opponent_keeps_efficiency_features_active():
    features = state_to_features(
        make_state(
            efficient_crate_field(),
            others=[("opponent", 0, True, (1, 1))],
        )
    )

    assert features[32] == 1.0
    assert features[36] == 1.0
    assert np.sum(features[33:37]) == 1.0
    assert np.isclose(features[37], 1.0 / 18.0)
    assert np.isclose(features[38], 2.0 / 4.0)


def test_current_best_position_has_yield_without_move_direction():
    field = open_field()
    field[4, 2] = 1
    field[4, 6] = 1

    features = state_to_features(make_state(field))

    assert features[32] == 0.0
    assert np.allclose(features[33:38], 0.0)
    assert np.isclose(features[38], 2.0 / 4.0)


def test_endgame_opponent_features_activate_with_reachable_path():
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

    features = state_to_features(state)

    assert features[39] == 1.0
    assert features[41] == 1.0  # RIGHT (feature 40+1)
    assert np.sum(features[40:44]) == 1.0
    assert np.isclose(features[44], 3.0 / 22.0)


def test_endgame_opponent_features_disabled_by_gate_conditions():
    field = open_field(size=11)

    base_state = {
        "field": field,
        "self": ("q_learning_agent", 0, True, (5, 5)),
        "others": [("opponent", 0, True, (9, 5))],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "round": 1,
        "step": 1,
    }

    coin_state = dict(base_state)
    coin_state["coins"] = [(1, 1)]
    assert np.allclose(state_to_features(coin_state)[39:45], 0.0)

    crate_state = dict(base_state)
    crate_field = field.copy()
    crate_field[4, 4] = 1
    crate_state["field"] = crate_field
    assert np.allclose(state_to_features(crate_state)[39:45], 0.0)

    bomb_state = dict(base_state)
    bomb_state["bombs"] = [((6, 5), 3)]
    assert np.allclose(state_to_features(bomb_state)[39:45], 0.0)

    explosion_state = dict(base_state)
    explosion_map = np.zeros_like(field)
    explosion_map[6, 5] = 1
    explosion_state["explosion_map"] = explosion_map
    assert np.allclose(state_to_features(explosion_state)[39:45], 0.0)

    in_range_state = dict(base_state)
    in_range_state["others"] = [("opponent", 0, True, (8, 5))]
    assert np.allclose(state_to_features(in_range_state)[39:45], 0.0)

    no_opponent_state = dict(base_state)
    no_opponent_state["others"] = []
    assert np.allclose(state_to_features(no_opponent_state)[39:45], 0.0)