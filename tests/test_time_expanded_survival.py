from types import SimpleNamespace

import numpy as np

from agent_code.q_learning_agent import callbacks
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM,
    _time_expanded_candidate_indices,
    act,
    state_to_features,
    valid_action_indices,
)
from agent_code.q_learning_agent.game_utils import (
    action_has_survival_route,
    blast_tiles,
    bomb_has_robust_escape_route,
    earliest_danger_times,
    safety_blast_tiles,
    tile_is_safe_at_time,
    time_indexed_danger_schedule,
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
    others=None,
    explosion_map=None,
    bomb_available=False,
):
    if field is None:
        field = open_field()
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
        "self": ("test", 0, bomb_available, position),
        "others": others,
        "bombs": bombs,
        "coins": [],
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
    )


def test_safety_blast_matches_engine_and_continues_through_crate():
    field = open_field()
    field[6, 5] = 1

    assert (7, 5) not in blast_tiles(field, (5, 5))
    assert (7, 5) in safety_blast_tiles(field, (5, 5))


def test_schedule_preserves_two_separate_explosion_windows():
    state = make_state(
        bombs=[
            ((3, 5), 0),
            ((5, 3), 3),
        ]
    )

    danger = time_indexed_danger_schedule(state)

    assert danger[1, 5, 5]
    assert danger[2, 5, 5]
    assert not danger[3, 5, 5]
    assert danger[4, 5, 5]
    assert danger[5, 5, 5]
    assert not danger[6, 5, 5]


def test_schedule_accounts_for_early_chain_reaction():
    state = make_state(
        bombs=[
            ((3, 5), 0),
            ((6, 5), 3),
        ]
    )

    danger = time_indexed_danger_schedule(state)

    assert danger[1, 8, 5]
    assert danger[2, 8, 5]
    assert not danger[3, 8, 5]


def test_existing_explosion_marks_only_its_remaining_danger():
    explosion_map = np.zeros((11, 11), dtype=int)
    explosion_map[5, 5] = 1
    state = make_state(explosion_map=explosion_map)

    danger = time_indexed_danger_schedule(state)

    assert danger[1, 5, 5]
    assert not danger[2, 5, 5]


def test_safe_board_keeps_exact_candidate_set():
    state = make_state()
    valid = valid_action_indices(state)

    assert _time_expanded_candidate_indices(state, valid) == valid


def test_immediately_exploding_tile_is_removed():
    state = make_state(
        position=(5, 5),
        bombs=[((5, 1), 0)],
    )
    valid = valid_action_indices(state)

    filtered = _time_expanded_candidate_indices(state, valid)

    assert ACTIONS.index("UP") not in filtered
    assert ACTIONS.index("WAIT") in filtered


def test_wait_is_kept_when_explosion_temporarily_blocks_only_exit():
    field = np.full((11, 11), -1, dtype=int)
    field[5, 5] = 0
    field[5, 4] = 0
    field[5, 3] = 0
    explosion_map = np.zeros_like(field)
    explosion_map[5, 4] = 1
    state = make_state(
        field=field,
        explosion_map=explosion_map,
    )
    valid = valid_action_indices(state)

    assert not action_has_survival_route(state, "UP")
    assert action_has_survival_route(state, "WAIT")
    assert _time_expanded_candidate_indices(state, valid) == [
        ACTIONS.index("WAIT")
    ]


def test_safe_next_step_into_dead_end_is_rejected():
    field = np.full((11, 11), -1, dtype=int)
    field[4, 5] = 0
    field[5, 5] = 0
    field[6, 5] = 0
    field[7, 5] = 0
    field[8, 5] = 0
    state = make_state(
        field=field,
        position=(5, 5),
        bombs=[((8, 5), 1)],
    )
    danger_time = earliest_danger_times(state)[6, 5]
    valid = valid_action_indices(state)

    assert tile_is_safe_at_time(danger_time, arrival_time=1)
    assert not action_has_survival_route(state, "RIGHT")
    assert action_has_survival_route(state, "LEFT")

    filtered = _time_expanded_candidate_indices(state, valid)

    assert ACTIONS.index("RIGHT") not in filtered
    assert ACTIONS.index("LEFT") in filtered


def test_crate_does_not_hide_engine_danger_from_filter():
    field = open_field()
    field[6, 5] = 1
    state = make_state(
        field=field,
        position=(7, 6),
        bombs=[((5, 5), 0)],
    )
    old_danger = earliest_danger_times(state)[7, 5]
    valid = valid_action_indices(state)

    assert np.isinf(old_danger)

    filtered = _time_expanded_candidate_indices(state, valid)

    assert ACTIONS.index("UP") not in filtered


def test_act_cannot_choose_engine_danger_hidden_behind_crate():
    field = open_field()
    field[6, 5] = 1
    state = make_state(
        field=field,
        position=(7, 6),
        bombs=[((5, 5), 0)],
    )
    agent = fake_agent()
    agent.model[ACTIONS.index("UP"), 0] = 100.0

    assert act(agent, state) != "UP"


def test_open_board_bomb_has_two_step_escape_margin():
    state = make_state(bomb_available=True)

    assert bomb_has_robust_escape_route(state)


def test_nearby_opponent_bomb_has_two_distinct_short_exits():
    state = make_state(
        bomb_available=True,
        others=[("enemy", 0, True, (7, 5))],
    )

    assert bomb_has_robust_escape_route(state)


def test_fragile_four_step_corridor_bomb_is_removed():
    field = np.full((11, 11), -1, dtype=int)
    field[4, 5] = 1
    field[5:10, 5] = 0
    field[8, 6] = 0
    state = make_state(
        field=field,
        position=(5, 5),
        others=[("enemy", 0, True, (8, 6))],
        bomb_available=True,
    )
    valid = valid_action_indices(state)

    assert ACTIONS.index("BOMB") in valid
    assert not bomb_has_robust_escape_route(state)

    filtered = _time_expanded_candidate_indices(state, valid)

    assert ACTIONS.index("BOMB") not in filtered
    assert filtered


def test_long_crate_escape_is_kept_when_opponents_are_far_away():
    field = np.full((11, 11), -1, dtype=int)
    field[4, 5] = 1
    field[5:10, 5] = 0
    state = make_state(
        field=field,
        position=(5, 5),
        bomb_available=True,
    )
    valid = valid_action_indices(state)

    assert ACTIONS.index("BOMB") in valid
    assert bomb_has_robust_escape_route(state)

    filtered = _time_expanded_candidate_indices(state, valid)

    assert ACTIONS.index("BOMB") in filtered


def test_no_survival_route_preserves_nonempty_fallback():
    field = np.full((11, 11), -1, dtype=int)
    field[5, 5] = 0
    state = make_state(
        field=field,
        bombs=[((5, 5), 0)],
    )
    valid = valid_action_indices(state)

    assert valid == [ACTIONS.index("WAIT")]
    assert _time_expanded_candidate_indices(state, valid) == valid


def test_unsolved_danger_fallback_never_adds_another_bomb(monkeypatch):
    field = open_field()
    field[5, 3] = 1
    state = make_state(
        field=field,
        bomb_available=True,
    )
    valid = valid_action_indices(state)

    assert ACTIONS.index("BOMB") in valid

    monkeypatch.setattr(
        callbacks,
        "action_has_survival_route",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        callbacks,
        "bomb_has_robust_escape_route",
        lambda *_args, **_kwargs: False,
    )

    filtered = _time_expanded_candidate_indices(state, valid)

    assert ACTIONS.index("BOMB") not in filtered
    assert filtered


def test_shield_does_not_change_feature_values_or_dimension():
    state = make_state()
    before = state_to_features(state)

    _time_expanded_candidate_indices(
        state,
        valid_action_indices(state),
    )
    after = state_to_features(state)

    assert FEATURE_DIM == 39
    assert before.shape == (FEATURE_DIM,)
    assert np.array_equal(before, after)
