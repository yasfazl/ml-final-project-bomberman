from types import SimpleNamespace

import numpy as np

from agent_code.q_learning_agent.train import (
    WAITED_WITH_GOAL,
    add_waiting_event,
    reward_from_events,
)


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(field, bombs=None):
    if bombs is None:
        bombs = []

    return {
        "field": field,
        "self": ("test", 0, True, (5, 5)),
        "bombs": bombs,
        "others": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


class Logger:
    def debug(self, *_args, **_kwargs):
        pass


def test_waiting_with_reachable_crate_goal_is_penalized():
    field = open_field()
    field[8, 5] = 1
    state = make_state(field)
    events = []

    add_waiting_event(state, "WAIT", events)

    assert WAITED_WITH_GOAL in events
    assert reward_from_events(SimpleNamespace(logger=Logger()), events) < -1.0


def test_waiting_during_active_bomb_has_no_extra_goal_penalty():
    field = open_field()
    field[8, 5] = 1
    state = make_state(field, bombs=[((5, 5), 3)])
    events = []

    add_waiting_event(state, "WAIT", events)

    assert WAITED_WITH_GOAL not in events


def test_waiting_during_endgame_pursuit_uses_goal_penalty():
    field = open_field(size=11)
    state = make_state(field)
    state["self"] = ("test", 0, True, (5, 5))
    state["others"] = [("opponent", 0, False, (9, 5))]
    state["coins"] = []
    state["explosion_map"] = np.zeros_like(field)
    events = []

    add_waiting_event(state, "WAIT", events)

    assert WAITED_WITH_GOAL in events
    assert reward_from_events(SimpleNamespace(logger=Logger()), events) < -1.0