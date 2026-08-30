from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM,
)
from agent_code.q_learning_agent.train import (
    BASE_FEATURE_DIM,
    BOMBED_BEFORE_BETTER_BOMB_SPOT,
    MOVED_AWAY_FROM_BETTER_BOMB_SPOT,
    MOVED_TOWARD_BETTER_BOMB_SPOT,
    add_bomb_placement_event,
    add_crate_efficiency_navigation_event,
    setup_training,
    update_q_learning,
)


class Logger:
    def info(self, *_args, **_kwargs):
        pass

    def debug(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def efficient_crate_field():
    """Current tile hits one crate; one step left hits two."""
    field = open_field()
    field[4, 2] = 1
    field[3, 2] = 1
    field[3, 6] = 1
    return field


def make_state(position=(4, 4), field=None):
    if field is None:
        field = efficient_crate_field()

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


def test_moving_toward_better_bomb_spot_adds_event():
    events = []

    add_crate_efficiency_navigation_event(
        make_state(position=(4, 4)),
        make_state(position=(3, 4)),
        events,
    )

    assert MOVED_TOWARD_BETTER_BOMB_SPOT in events
    assert MOVED_AWAY_FROM_BETTER_BOMB_SPOT not in events


def test_moving_away_from_better_bomb_spot_adds_event():
    events = []

    add_crate_efficiency_navigation_event(
        make_state(position=(4, 4)),
        make_state(position=(5, 4)),
        events,
    )

    assert MOVED_AWAY_FROM_BETTER_BOMB_SPOT in events
    assert MOVED_TOWARD_BETTER_BOMB_SPOT not in events


def test_bombing_early_is_penalized_when_better_spot_exists():
    events = [e.BOMB_DROPPED]

    add_bomb_placement_event(
        make_state(position=(4, 4)),
        "BOMB",
        events,
    )

    assert BOMBED_BEFORE_BETTER_BOMB_SPOT in events


def test_fine_tuning_preserves_original_feature_weights():
    fake_self = SimpleNamespace(
        logger=Logger(),
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        epsilon=0.2,
        episodes_trained=1000,
    )

    setup_training(fake_self)
    weights_before = fake_self.model.copy()

    update_q_learning(
        self=fake_self,
        old_game_state=make_state(position=(4, 4)),
        action="LEFT",
        new_game_state=None,
        reward=2.0,
    )

    action_index = ACTIONS.index("LEFT")

    assert np.array_equal(
        fake_self.model[:, :BASE_FEATURE_DIM],
        weights_before[:, :BASE_FEATURE_DIM],
    )
    assert np.any(
        fake_self.model[
            action_index,
            BASE_FEATURE_DIM:,
        ] != 0.0
    )