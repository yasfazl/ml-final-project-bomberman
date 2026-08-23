
import numpy as np

from agent_code.q_learning_agent.callbacks import (
    FEATURE_DIM,
    state_to_features,
)
from agent_code.q_learning_agent.game_utils import (
    nearest_crate_bombing_path,
)
from agent_code.q_learning_agent.train import (
    MOVED_AWAY_FROM_CRATE,
    MOVED_TOWARD_CRATE,
    add_crate_navigation_event,
)


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(field, position):
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test", 0, True, position),
        "bombs": [],
        "others": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_path_points_to_nearest_safe_crate_bombing_tile():
    field = open_field()
    field[8, 5] = 1

    direction, distance = nearest_crate_bombing_path(
        make_state(field, (2, 5))
    )

    assert direction == 1  # RIGHT
    assert distance == 3


def test_crate_navigation_features():
    field = open_field()
    field[8, 5] = 1

    features = state_to_features(make_state(field, (2, 5)))

    assert FEATURE_DIM == 39
    assert features.shape == (39,)
    assert features[26] == 1.0
    assert features[28] == 1.0  # RIGHT
    assert features[27] == 0.0
    assert features[29] == 0.0
    assert features[30] == 0.0
    assert np.isclose(features[31], 3.0 / 22.0)


def test_reward_for_moving_toward_crate_target():
    field = open_field()
    field[8, 5] = 1
    events = []

    add_crate_navigation_event(
        make_state(field, (2, 5)),
        make_state(field.copy(), (3, 5)),
        events,
    )

    assert MOVED_TOWARD_CRATE in events


def test_penalty_for_moving_away_from_crate_target():
    field = open_field()
    field[8, 5] = 1
    events = []

    add_crate_navigation_event(
        make_state(field, (3, 5)),
        make_state(field.copy(), (2, 5)),
        events,
    )

    assert MOVED_AWAY_FROM_CRATE in events


def test_visible_coin_disables_crate_navigation_features():
    field = open_field()
    field[8, 5] = 1
    state = make_state(field, (2, 5))
    state["coins"] = [(2, 7)]

    features = state_to_features(state)

    assert features[5] == 1.0
    assert np.allclose(features[26:32], 0.0)