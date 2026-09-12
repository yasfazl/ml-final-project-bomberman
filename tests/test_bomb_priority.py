from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.q_learning_agent.train import (
    BOMB_TARGETED_CRATE,
    BOMB_WHILE_COIN_VISIBLE,
    add_bomb_placement_event,
    reward_from_events,
)


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(field, coins=None):
    if coins is None:
        coins = []

    return {
        "field": field,
        "self": ("test", 0, True, (5, 5)),
        "bombs": [],
        "others": [],
        "coins": coins,
        "explosion_map": np.zeros_like(field),
    }


class Logger:
    def debug(self, *_args, **_kwargs):
        pass


def test_bomb_while_coin_visible_gets_penalty_event():
    field = open_field()
    field[5, 3] = 1
    events = [e.BOMB_DROPPED]

    add_bomb_placement_event(
        make_state(field, coins=[(7, 5)]),
        "BOMB",
        events,
    )

    assert BOMB_TARGETED_CRATE in events
    assert BOMB_WHILE_COIN_VISIBLE in events


def test_multiple_crates_receive_proportional_credit():
    field = open_field()
    field[5, 3] = 1
    field[7, 5] = 1
    events = [e.BOMB_DROPPED]

    add_bomb_placement_event(
        make_state(field),
        "BOMB",
        events,
    )

    assert events.count(BOMB_TARGETED_CRATE) == 2


def test_bombing_while_coin_visible_is_immediately_negative():
    fake_self = SimpleNamespace(logger=Logger())

    reward = reward_from_events(
        fake_self,
        [BOMB_TARGETED_CRATE, BOMB_WHILE_COIN_VISIBLE],
    )

    assert reward < 0.0