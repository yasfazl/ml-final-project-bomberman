from types import SimpleNamespace
import pickle

import numpy as np

from agent_code.q_learning_agent import callbacks
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM,
    state_to_features,
    valid_action_indices,
)


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(
    field=None,
    position=(5, 5),
    bombs=None,
    bomb_available=True,
):
    if field is None:
        field = open_field()
    if bombs is None:
        bombs = []

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test", 0, bomb_available, position),
        "bombs": bombs,
        "others": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_bomb_requires_a_target():
    state = make_state()
    features = state_to_features(state)

    assert FEATURE_DIM == 32
    assert features[17] == 1.0
    assert features[18] == 1.0
    assert features[19] == 0.0
    assert ACTIONS.index("BOMB") not in valid_action_indices(state)


def test_targeted_bomb_is_enabled_when_escape_exists():
    field = open_field()
    field[5, 3] = 1
    state = make_state(field=field)
    features = state_to_features(state)

    assert features[19] == 0.25
    assert ACTIONS.index("BOMB") in valid_action_indices(state)


def test_bomb_is_disabled_without_escape():
    field = np.full((11, 11), -1, dtype=int)
    field[5, 5] = 0
    field[5, 4] = 1
    state = make_state(field=field)
    features = state_to_features(state)

    assert features[18] == 0.0
    assert ACTIONS.index("BOMB") not in valid_action_indices(state)


def test_escape_direction_features():
    state = make_state(
        bombs=[((5, 5), 3)],
        bomb_available=False,
    )
    features = state_to_features(state)

    assert features[21] == 1.0  # UP
    assert np.allclose(features[22:25], 0.0)
    assert features[25] == 0.5


def test_setup_migrates_an_11_feature_model(tmp_path, monkeypatch):
    model_path = tmp_path / "q_model.pkl"
    old_weights = np.ones((len(ACTIONS), 11), dtype=np.float64)

    with model_path.open("wb") as file:
        pickle.dump(
            {
                "weights": old_weights,
                "epsilon": 0.2,
                "episodes_trained": 100,
            },
            file,
        )

    monkeypatch.setattr(callbacks, "MODEL_PATH", model_path)

    class Logger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    fake_self = SimpleNamespace(logger=Logger())
    callbacks.setup(fake_self)

    assert fake_self.model.shape == (len(ACTIONS), FEATURE_DIM)
    assert np.allclose(fake_self.model[:, :11], old_weights)
    assert np.allclose(fake_self.model[:, 11:], 0.0)
    assert fake_self.epsilon == 0.2
    assert fake_self.episodes_trained == 100