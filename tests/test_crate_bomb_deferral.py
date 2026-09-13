from types import SimpleNamespace

import numpy as np

from agent_code.q_learning_agent import callbacks
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM,
    _crate_bomb_deferral_candidate_indices,
    act,
    state_to_features,
    valid_action_indices,
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


def efficient_step_field():
    """Current bomb hits one crate; one step UP hits three."""
    field = open_field()
    field[5, 2] = 1
    field[2, 4] = 1
    field[8, 4] = 1
    return field


def make_state(
    *,
    field=None,
    position=(5, 5),
    coins=None,
    bombs=None,
    others=None,
    explosion_map=None,
):
    if field is None:
        field = efficient_step_field()
    if coins is None:
        coins = []
    if bombs is None:
        bombs = []
    if others is None:
        others = []
    if explosion_map is None:
        explosion_map = np.zeros_like(field)

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test", 0, True, position),
        "others": others,
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


def test_defers_one_crate_bomb_for_safe_three_crate_position():
    state = make_state()
    candidates = valid_action_indices(state)

    assert ACTIONS.index("BOMB") in candidates
    assert _crate_bomb_deferral_candidate_indices(
        state,
        candidates,
    ) == [ACTIONS.index("UP")]


def test_act_moves_even_when_model_strongly_prefers_immediate_bomb():
    state = make_state()
    agent = fake_agent()
    agent.model[ACTIONS.index("BOMB"), 0] = 100.0
    agent.model[ACTIONS.index("UP"), 0] = 1.0

    assert act(agent, state) == "UP"
    assert agent.last_own_bomb_position is None


def test_equal_crate_yield_keeps_original_candidates():
    field = open_field()
    field[5, 2] = 1
    state = make_state(field=field)
    candidates = valid_action_indices(state)

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )


def test_one_additional_crate_is_not_enough_to_defer():
    field = open_field()
    field[5, 2] = 1
    field[8, 4] = 1
    state = make_state(field=field)
    candidates = valid_action_indices(state)

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )


def test_opponent_target_keeps_immediate_bomb_available():
    state = make_state(
        others=[("enemy", 0, True, (8, 5))],
    )
    candidates = valid_action_indices(state)

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )
    assert ACTIONS.index("BOMB") in candidates


def test_reachable_coin_keeps_original_candidates():
    state = make_state(coins=[(6, 5)])
    candidates = valid_action_indices(state)

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )


def test_distant_active_bomb_does_not_disable_deferral():
    state = make_state(bombs=[((9, 9), 3)])
    candidates = valid_action_indices(state)

    assert _crate_bomb_deferral_candidate_indices(
        state,
        candidates,
    ) == [ACTIONS.index("UP")]


def test_locally_dangerous_bomb_keeps_original_candidates():
    state = make_state(bombs=[((5, 8), 3)])
    candidates = valid_action_indices(state)

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )


def test_distant_active_explosion_does_not_disable_deferral():
    explosion_map = np.zeros((11, 11), dtype=int)
    explosion_map[9, 9] = 1
    state = make_state(explosion_map=explosion_map)
    candidates = valid_action_indices(state)

    assert _crate_bomb_deferral_candidate_indices(
        state,
        candidates,
    ) == [ACTIONS.index("UP")]


def test_explosion_on_current_tile_keeps_original_candidates():
    explosion_map = np.zeros((11, 11), dtype=int)
    explosion_map[5, 5] = 1
    state = make_state(explosion_map=explosion_map)
    candidates = valid_action_indices(state)

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )


def test_safety_rejected_better_direction_is_not_restored():
    state = make_state()
    candidates = [
        index
        for index in valid_action_indices(state)
        if ACTIONS[index] != "UP"
    ]

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )


def test_fragile_future_bomb_does_not_force_the_better_move(
    monkeypatch,
):
    state = make_state()
    candidates = valid_action_indices(state)
    monkeypatch.setattr(
        callbacks,
        "bomb_has_robust_escape_route",
        lambda _state: False,
    )

    assert (
        _crate_bomb_deferral_candidate_indices(state, candidates)
        == candidates
    )


def test_filter_does_not_change_features_or_dimension():
    state = make_state()
    before = state_to_features(state)

    _crate_bomb_deferral_candidate_indices(
        state,
        valid_action_indices(state),
    )
    after = state_to_features(state)

    assert FEATURE_DIM == 39
    assert before.shape == (FEATURE_DIM,)
    assert np.array_equal(before, after)
