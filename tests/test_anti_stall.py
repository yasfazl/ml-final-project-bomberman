from types import SimpleNamespace

import numpy as np

from agent_code.q_learning_agent import callbacks
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    ANTI_STALL_WAIT_LIMIT,
    FEATURE_DIM,
    _anti_stall_candidate_indices,
    _record_anti_stall_choice,
    act,
)


class Logger:
    def debug(self, *_args, **_kwargs):
        pass


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(
    *,
    field=None,
    position=(5, 5),
    coins=None,
    bombs=None,
    explosion_map=None,
    round_number=1,
):
    if field is None:
        field = open_field()
    if coins is None:
        coins = []
    if bombs is None:
        bombs = []
    if explosion_map is None:
        explosion_map = np.zeros_like(field)

    return {
        "round": round_number,
        "step": 1,
        "field": field,
        "self": ("test", 0, False, position),
        "others": [],
        "bombs": bombs,
        "coins": coins,
        "explosion_map": explosion_map,
    }


def fake_agent():
    return SimpleNamespace(
        train=False,
        epsilon=0.0,
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        logger=Logger(),
        last_own_bomb_position=None,
        last_seen_round=None,
        consecutive_safe_waits=0,
        last_wait_position=None,
    )


def stalled_agent(position=(5, 5)):
    agent = fake_agent()
    agent.consecutive_safe_waits = ANTI_STALL_WAIT_LIMIT
    agent.last_wait_position = position
    return agent


def test_guard_is_inactive_before_wait_limit():
    state = make_state(coins=[(5, 3)])
    agent = stalled_agent()
    agent.consecutive_safe_waits = ANTI_STALL_WAIT_LIMIT - 1
    candidates = [ACTIONS.index("UP"), ACTIONS.index("WAIT")]

    assert (
        _anti_stall_candidate_indices(agent, state, candidates)
        == candidates
    )


def test_guard_removes_wait_for_approved_coin_progress():
    state = make_state(coins=[(5, 3)])
    agent = stalled_agent()
    candidates = [ACTIONS.index("UP"), ACTIONS.index("WAIT")]

    assert _anti_stall_candidate_indices(
        agent,
        state,
        candidates,
    ) == [ACTIONS.index("UP")]


def test_guard_removes_wait_for_approved_crate_progress():
    field = open_field()
    # The crate is in range from one tile UP, but not from the current tile.
    field[8, 4] = 1
    state = make_state(field=field)
    agent = stalled_agent()
    candidates = [ACTIONS.index("UP"), ACTIONS.index("WAIT")]

    assert _anti_stall_candidate_indices(
        agent,
        state,
        candidates,
    ) == [ACTIONS.index("UP")]


def test_unreachable_coin_falls_back_to_crate_progress(monkeypatch):
    field = open_field()
    field[8, 4] = 1
    state = make_state(
        field=field,
        coins=[(8, 8)],
    )
    agent = stalled_agent()
    candidates = [ACTIONS.index("UP"), ACTIONS.index("WAIT")]
    monkeypatch.setattr(
        callbacks,
        "nearest_coin_path",
        lambda _state: (None, None),
    )

    assert _anti_stall_candidate_indices(
        agent,
        state,
        candidates,
    ) == [ACTIONS.index("UP")]


def test_guard_does_not_override_known_danger():
    state = make_state(
        coins=[(5, 3)],
        bombs=[((5, 2), 2)],
    )
    agent = stalled_agent()
    candidates = [ACTIONS.index("UP"), ACTIONS.index("WAIT")]

    assert (
        _anti_stall_candidate_indices(agent, state, candidates)
        == candidates
    )


def test_guard_requires_safety_approved_progress_direction():
    state = make_state(coins=[(5, 3)])
    agent = stalled_agent()
    candidates = [ACTIONS.index("RIGHT"), ACTIONS.index("WAIT")]

    assert (
        _anti_stall_candidate_indices(agent, state, candidates)
        == candidates
    )


def test_guard_keeps_wait_without_coin_or_crate_objective():
    state = make_state()
    agent = stalled_agent()
    candidates = [ACTIONS.index("UP"), ACTIONS.index("WAIT")]

    assert (
        _anti_stall_candidate_indices(agent, state, candidates)
        == candidates
    )


def test_recording_resets_after_non_wait_action():
    state = make_state()
    agent = stalled_agent()

    _record_anti_stall_choice(agent, "UP", state)

    assert agent.consecutive_safe_waits == 0
    assert agent.last_wait_position is None


def test_act_breaks_safe_wait_loop_on_fourth_decision():
    field = np.full((11, 11), -1, dtype=int)
    field[5, 5] = 0
    field[5, 4] = 0
    field[5, 3] = 0
    state = make_state(
        field=field,
        coins=[(5, 4)],
    )
    agent = fake_agent()
    agent.model[ACTIONS.index("WAIT"), 0] = 100.0
    agent.model[ACTIONS.index("UP"), 0] = 10.0

    assert [act(agent, state) for _ in range(3)] == [
        "WAIT",
        "WAIT",
        "WAIT",
    ]
    assert act(agent, state) == "UP"
    assert agent.consecutive_safe_waits == 0
