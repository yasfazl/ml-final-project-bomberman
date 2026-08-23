import numpy as np

from agent_code.q_learning_agent.train import (
    MOVED_AWAY_FROM_OPPONENT,
    MOVED_TOWARD_OPPONENT,
    WAITED_WITH_GOAL,
    add_crate_navigation_event,
    add_opponent_navigation_event,
    add_waiting_event,
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
    opponent=(6, 4),
    coins=(),
    field=None,
):
    if field is None:
        field = open_field()

    others = []

    if opponent is not None:
        others.append(("opponent", 0, True, opponent))

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


def test_moving_toward_opponent_event():
    old_state = make_state(position=(2, 4))
    new_state = make_state(position=(3, 4))
    events = []

    add_opponent_navigation_event(
        old_state,
        new_state,
        events,
    )

    assert MOVED_TOWARD_OPPONENT in events
    assert MOVED_AWAY_FROM_OPPONENT not in events


def test_moving_away_from_opponent_event():
    old_state = make_state(position=(2, 4))
    new_state = make_state(position=(1, 4))
    events = []

    add_opponent_navigation_event(
        old_state,
        new_state,
        events,
    )

    assert MOVED_AWAY_FROM_OPPONENT in events
    assert MOVED_TOWARD_OPPONENT not in events


def test_coin_disables_opponent_navigation_reward():
    old_state = make_state(
        position=(2, 4),
        coins=[(2, 5)],
    )
    new_state = make_state(
        position=(3, 4),
        coins=[(2, 5)],
    )
    events = []

    add_opponent_navigation_event(
        old_state,
        new_state,
        events,
    )

    assert MOVED_TOWARD_OPPONENT not in events
    assert MOVED_AWAY_FROM_OPPONENT not in events


def test_waiting_with_opponent_goal_is_penalized():
    state = make_state(position=(2, 4))
    events = []

    add_waiting_event(state, "WAIT", events)

    assert WAITED_WITH_GOAL in events


def test_crate_reward_is_disabled_during_opponent_pursuit():
    field = open_field()
    field[2, 2] = 1

    old_state = make_state(
        position=(2, 4),
        field=field,
    )
    new_state = make_state(
        position=(3, 4),
        field=field,
    )
    events = []

    add_crate_navigation_event(
        old_state,
        new_state,
        events,
    )

    assert events == []