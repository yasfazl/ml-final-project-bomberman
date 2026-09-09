import numpy as np

from agent_code.dqn_agent import callbacks
from agent_code.q_learning_agent.callbacks import ACTIONS


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def game_state(*, coins=None, bombs=None):
    field = open_field()
    return {
        "round": 1,
        "step": 50,
        "field": field,
        "self": ("dqn_agent", 0, True, (3, 3)),
        # RIGHT -> (4, 3) and UP -> (3, 2) can both be entered by
        # this opponent from (4, 2).
        "others": [("enemy", 0, True, (4, 2))],
        "bombs": [] if bombs is None else bombs,
        "coins": [] if coins is None else coins,
        "explosion_map": np.zeros_like(field),
    }


def test_identifies_only_opponent_reachable_movement_destinations():
    candidates = [
        ACTIONS.index("UP"),
        ACTIONS.index("RIGHT"),
        ACTIONS.index("DOWN"),
        ACTIONS.index("WAIT"),
        ACTIONS.index("BOMB"),
    ]

    contested = callbacks._contested_movement_indices(
        game_state(),
        candidates,
    )

    assert contested == {
        ACTIONS.index("UP"),
        ACTIONS.index("RIGHT"),
    }


def test_danger_guard_removes_contested_moves_when_safe_move_exists():
    state = game_state(bombs=[((3, 5), 2)])
    candidates = [
        ACTIONS.index("RIGHT"),
        ACTIONS.index("DOWN"),
        ACTIONS.index("WAIT"),
    ]

    filtered = callbacks._danger_conflict_candidate_indices(
        state,
        candidates,
    )

    assert ACTIONS.index("RIGHT") not in filtered
    assert ACTIONS.index("DOWN") in filtered
    assert ACTIONS.index("WAIT") in filtered


def test_danger_guard_keeps_only_possible_escape_move():
    state = game_state(bombs=[((3, 5), 2)])
    candidates = [ACTIONS.index("RIGHT"), ACTIONS.index("WAIT")]

    assert callbacks._danger_conflict_candidate_indices(
        state,
        candidates,
    ) == candidates


def test_soft_endgame_guard_selects_near_optimal_uncontested_action():
    state = game_state()
    candidates = [
        ACTIONS.index("RIGHT"),
        ACTIONS.index("DOWN"),
        ACTIONS.index("WAIT"),
    ]
    q_values = np.array([0.0, 10.0, 9.6, 0.0, 0.0, 0.0])

    filtered = callbacks._soft_endgame_conflict_candidate_indices(
        state,
        candidates,
        q_values,
    )

    assert filtered == [ACTIONS.index("DOWN")]


def test_soft_endgame_guard_preserves_clear_learned_preference():
    state = game_state()
    candidates = [
        ACTIONS.index("RIGHT"),
        ACTIONS.index("DOWN"),
        ACTIONS.index("WAIT"),
    ]
    q_values = np.array([0.0, 10.0, 8.0, 0.0, 0.0, 0.0])

    assert callbacks._soft_endgame_conflict_candidate_indices(
        state,
        candidates,
        q_values,
    ) == candidates


def test_soft_guard_is_inactive_during_coin_play():
    state = game_state(coins=[(6, 3)])
    candidates = [
        ACTIONS.index("RIGHT"),
        ACTIONS.index("DOWN"),
        ACTIONS.index("WAIT"),
    ]
    q_values = np.array([0.0, 10.0, 9.9, 0.0, 0.0, 0.0])

    assert callbacks._soft_endgame_conflict_candidate_indices(
        state,
        candidates,
        q_values,
    ) == candidates
