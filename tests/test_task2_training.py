from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.q_learning_agent.callbacks import ACTIONS, FEATURE_DIM
from agent_code.q_learning_agent.train import (
    BOMB_TARGETED_CRATE,
    MOVED_TOWARD_SAFETY,
    REACHED_SAFETY,
    add_bomb_placement_event,
    add_escape_events,
    reward_from_events,
    update_q_learning,
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
):
    if field is None:
        field = open_field()
    if bombs is None:
        bombs = []

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test", 0, True, position),
        "bombs": bombs,
        "others": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


class Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def test_targeted_crate_bomb_event():
    field = open_field()
    field[5, 3] = 1
    events = [e.BOMB_DROPPED]

    add_bomb_placement_event(
        make_state(field=field),
        "BOMB",
        events,
    )

    assert BOMB_TARGETED_CRATE in events


def test_moving_toward_safety_event():
    old_state = make_state(
        position=(5, 5),
        bombs=[((5, 5), 3)],
    )
    new_state = make_state(
        position=(5, 4),
        bombs=[((5, 5), 2)],
    )
    events = []

    add_escape_events(old_state, "UP", new_state, events)

    assert MOVED_TOWARD_SAFETY in events


def test_reaching_safety_event():
    old_state = make_state(
        position=(5, 4),
        bombs=[((5, 5), 2)],
    )
    new_state = make_state(
        position=(6, 4),
        bombs=[((5, 5), 1)],
    )
    events = []

    add_escape_events(old_state, "RIGHT", new_state, events)

    assert REACHED_SAFETY in events


def test_self_kill_reward_is_strongly_negative():
    fake_self = SimpleNamespace(logger=Logger())

    reward = reward_from_events(fake_self, [e.KILLED_SELF])

    assert reward < -20.0


def test_terminal_q_learning_update_changes_weights():
    fake_self = SimpleNamespace(
        logger=Logger(),
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
    )

    td_error = update_q_learning(
        self=fake_self,
        old_game_state=make_state(),
        action="UP",
        new_game_state=None,
        reward=2.0,
    )

    assert td_error == 2.0
    assert np.any(fake_self.model[ACTIONS.index("UP")] != 0.0)