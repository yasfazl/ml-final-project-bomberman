from types import SimpleNamespace

import numpy as np

from agent_code.q_learning_agent import callbacks
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM,
    _clear_inactive_own_bomb_memory,
    _post_bomb_candidate_indices,
    _remember_own_bomb_if_selected,
    act,
    state_to_features,
    valid_action_indices,
)
from agent_code.q_learning_agent.game_utils import (
    earliest_danger_times,
    tile_is_safe_at_time,
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
    field=None,
    position=(5, 5),
    bombs=None,
    coins=None,
    bomb_available=False,
    round_number=1,
):
    if field is None:
        field = open_field()
    if bombs is None:
        bombs = []
    if coins is None:
        coins = []

    return {
        "round": round_number,
        "step": 1,
        "field": field,
        "self": ("test", 0, bomb_available, position),
        "others": [],
        "bombs": bombs,
        "coins": coins,
        "explosion_map": np.zeros_like(field),
    }


def fake_agent(*, train=False, epsilon=0.0):
    return SimpleNamespace(
        train=train,
        epsilon=epsilon,
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        logger=Logger(),
        last_own_bomb_position=None,
        last_seen_round=None,
    )


def test_remember_selected_bomb_only():
    agent = fake_agent()
    state = make_state(position=(4, 6))

    _remember_own_bomb_if_selected(agent, "WAIT", state)
    assert agent.last_own_bomb_position is None

    _remember_own_bomb_if_selected(agent, "BOMB", state)
    assert agent.last_own_bomb_position == (4, 6)


def test_exploitation_records_selected_bomb():
    field = open_field()
    field[5, 3] = 1
    state = make_state(field=field, bomb_available=True)
    agent = fake_agent()
    agent.model[ACTIONS.index("BOMB"), 0] = 10.0

    assert act(agent, state) == "BOMB"
    assert agent.last_own_bomb_position == (5, 5)


def test_exploration_records_selected_bomb(monkeypatch):
    field = open_field()
    field[5, 3] = 1
    state = make_state(field=field, bomb_available=True)
    agent = fake_agent(train=True, epsilon=1.0)

    monkeypatch.setattr(callbacks.random, "random", lambda: 0.0)
    monkeypatch.setattr(
        callbacks.random,
        "choice",
        lambda _indices: ACTIONS.index("BOMB"),
    )

    assert act(agent, state) == "BOMB"
    assert agent.last_own_bomb_position == (5, 5)


def test_enemy_bomb_alone_does_not_activate_commitment():
    state = make_state(bombs=[((5, 2), 1)])
    agent = fake_agent()
    valid = valid_action_indices(state)

    assert _post_bomb_candidate_indices(agent, state, valid) == valid


def test_inside_own_blast_suppresses_wait_and_keeps_escape():
    state = make_state(bombs=[((5, 5), 3)])
    agent = fake_agent()
    agent.last_own_bomb_position = (5, 5)
    valid = valid_action_indices(state)

    filtered = _post_bomb_candidate_indices(agent, state, valid)

    assert filtered == [ACTIONS.index("UP")]
    assert ACTIONS.index("WAIT") not in filtered


def test_immediately_safe_alternative_movements_are_kept():
    state = make_state(
        position=(5, 7),
        bombs=[((5, 5), 3)],
    )
    agent = fake_agent()
    agent.last_own_bomb_position = (5, 5)

    filtered = _post_bomb_candidate_indices(
        agent,
        state,
        valid_action_indices(state),
    )

    assert filtered == [ACTIONS.index("RIGHT"), ACTIONS.index("LEFT")]


def test_outside_own_blast_blocks_imminent_reentry_and_keeps_wait():
    state = make_state(
        position=(6, 6),
        bombs=[((5, 5), 0)],
    )
    agent = fake_agent()
    agent.last_own_bomb_position = (5, 5)

    filtered = _post_bomb_candidate_indices(
        agent,
        state,
        valid_action_indices(state),
    )

    assert ACTIONS.index("UP") not in filtered
    assert ACTIONS.index("LEFT") not in filtered
    assert ACTIONS.index("WAIT") in filtered


def test_outside_own_blast_keeps_wait_when_future_enemy_blast_is_not_immediate():
    state = make_state(
        position=(6, 6),
        bombs=[
            ((5, 5), 3),
            ((6, 3), 3),
        ],
    )
    agent = fake_agent()
    agent.last_own_bomb_position = (5, 5)
    danger = earliest_danger_times(state)[6, 6]

    assert tile_is_safe_at_time(danger, arrival_time=1)

    filtered = _post_bomb_candidate_indices(
        agent,
        state,
        valid_action_indices(state),
    )

    assert ACTIONS.index("WAIT") in filtered


def test_missing_remembered_bomb_clears_memory_and_restores_actions():
    state = make_state()
    agent = fake_agent()
    agent.last_own_bomb_position = (5, 5)
    valid = valid_action_indices(state)

    _clear_inactive_own_bomb_memory(agent, state)

    assert agent.last_own_bomb_position is None
    assert _post_bomb_candidate_indices(agent, state, valid) == valid


def test_no_escape_candidate_uses_original_nonempty_fallback():
    field = np.full((11, 11), -1, dtype=int)
    field[5, 5] = 0
    state = make_state(
        field=field,
        position=(5, 5),
        bombs=[((5, 5), 3)],
    )
    agent = fake_agent()
    agent.last_own_bomb_position = (5, 5)
    valid = valid_action_indices(state)

    assert valid == [ACTIONS.index("WAIT")]
    assert _post_bomb_candidate_indices(agent, state, valid) == valid


def test_safe_feature_shape_and_values_are_stable_across_memory_state():
    state = make_state(coins=[(2, 2)])
    features_without_memory = state_to_features(state)
    agent = fake_agent()
    agent.last_own_bomb_position = (5, 5)
    features_with_memory = state_to_features(state)

    assert FEATURE_DIM == 39
    assert features_without_memory.shape == (FEATURE_DIM,)
    assert np.array_equal(features_with_memory, features_without_memory)
