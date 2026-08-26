from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.q_learning_agent import train as train_module
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM,
)
from agent_code.q_learning_agent.train import (
    BASE_FEATURE_DIM,
    BOMBED_BEFORE_BETTER_BOMB_SPOT,
    MOVED_AWAY_FROM_BETTER_BOMB_SPOT,
    MOVED_AWAY_FROM_OPPONENT,
    MOVED_TOWARD_BETTER_BOMB_SPOT,
    MOVED_TOWARD_OPPONENT,
    add_bomb_placement_event,
    add_crate_efficiency_navigation_event,
    add_endgame_opponent_navigation_event,
    reward_from_events,
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


def make_transition(
    step,
    reward=1.0,
    action="LEFT",
    terminal=False,
    field=None,
    position=(4, 4),
    next_position=(4, 4),
):
    old_game_state = make_state(
        position=position,
        field=field,
    )
    old_game_state["step"] = step

    new_game_state = None

    if not terminal:
        new_game_state = make_state(
            position=next_position,
            field=field,
        )
        new_game_state["step"] = step + 1

    return {
        "old_game_state": old_game_state,
        "action": action,
        "reward": reward,
        "new_game_state": new_game_state,
        "terminal": terminal,
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


def test_opponent_targets_block_crate_efficiency_navigation_events():
    field = efficient_crate_field()

    old_game_state = make_state(position=(4, 4), field=field)
    new_game_state = make_state(position=(3, 4), field=field)
    old_game_state["others"] = [("opponent", 0, False, (4, 3))]
    events = []
    add_crate_efficiency_navigation_event(
        old_game_state,
        new_game_state,
        events,
    )
    assert not events

    old_game_state = make_state(position=(4, 4), field=field)
    new_game_state = make_state(position=(3, 4), field=field)
    new_game_state["others"] = [("opponent", 0, False, (3, 3))]
    events = []
    add_crate_efficiency_navigation_event(
        old_game_state,
        new_game_state,
        events,
    )
    assert not events


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


def test_endgame_training_freezes_original_feature_weights():
    fake_self = SimpleNamespace(
        logger=Logger(),
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        epsilon=0.2,
        episodes_trained=1000,
    )

    setup_training(fake_self)
    weights_before = fake_self.model.copy()

    field = open_field(size=11)
    old_state = make_state(position=(5, 5), field=field)
    old_state["others"] = [("opponent", 0, False, (9, 5))]
    new_state = make_state(position=(6, 5), field=field)
    new_state["others"] = [("opponent", 0, False, (9, 5))]
    transitions = [
        {
            "old_game_state": old_state,
            "action": "DOWN",
            "reward": 1.0,
            "new_game_state": new_state,
            "terminal": False,
        }
    ]

    update_q_learning(
        self=fake_self,
        transitions=transitions,
    )

    action_index = ACTIONS.index("DOWN")

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


def test_endgame_opponent_navigation_reward_is_added_correctly():
    field = open_field(size=11)
    old_state = make_state(position=(5, 5), field=field)
    old_state["others"] = [("opponent", 0, False, (10, 5))]
    new_state = make_state(position=(6, 5), field=field)
    new_state["others"] = [("opponent", 0, False, (10, 5))]
    events = []

    add_endgame_opponent_navigation_event(old_state, new_state, events)

    assert MOVED_TOWARD_OPPONENT in events
    assert MOVED_AWAY_FROM_OPPONENT not in events
    assert reward_from_events(SimpleNamespace(logger=Logger()), events) == 0.45

    away_events = []
    add_endgame_opponent_navigation_event(
        new_state,
        old_state,
        away_events,
    )
    assert MOVED_AWAY_FROM_OPPONENT in away_events


def test_reaching_immediate_opponent_bomb_range_is_toward():
    field = open_field(size=11)
    old_state = make_state(position=(5, 5), field=field)
    old_state["others"] = [("opponent", 0, False, (10, 5))]

    # From (7, 5), bomb power can immediately reach opponent at (10, 5).
    new_state = make_state(position=(7, 5), field=field)
    new_state["others"] = [("opponent", 0, False, (10, 5))]

    events = []
    add_endgame_opponent_navigation_event(old_state, new_state, events)

    assert MOVED_TOWARD_OPPONENT in events
    assert MOVED_AWAY_FROM_OPPONENT not in events


def test_inactive_new_endgame_state_does_not_add_away_event():
    field = open_field(size=11)
    old_state = make_state(position=(5, 5), field=field)
    old_state["others"] = [("opponent", 0, False, (10, 5))]

    bomb_new_state = make_state(position=(6, 5), field=field)
    bomb_new_state["others"] = [("opponent", 0, False, (10, 5))]
    bomb_new_state["bombs"] = [((8, 5), 3)]

    bomb_events = []
    add_endgame_opponent_navigation_event(
        old_state,
        bomb_new_state,
        bomb_events,
    )

    assert MOVED_AWAY_FROM_OPPONENT not in bomb_events

    disappeared_new_state = make_state(position=(6, 5), field=field)
    disappeared_new_state["others"] = []

    disappeared_events = []
    add_endgame_opponent_navigation_event(
        old_state,
        disappeared_new_state,
        disappeared_events,
    )

    assert MOVED_AWAY_FROM_OPPONENT not in disappeared_events


def test_exact_five_step_return_uses_bootstrap():
    fake_self = SimpleNamespace(
        logger=Logger(),
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        epsilon=0.2,
        episodes_trained=1000,
    )

    fake_self.model[ACTIONS.index("WAIT"), 0] = 10.0

    transitions = [
        make_transition(step=index + 1, reward=1.0)
        for index in range(5)
    ]

    td_error = update_q_learning(
        self=fake_self,
        transitions=transitions,
    )

    expected = sum(0.9**index for index in range(5)) + (0.9**5) * 10.0

    assert np.isclose(td_error, expected)


def test_terminal_truncated_return_skips_bootstrap():
    fake_self = SimpleNamespace(
        logger=Logger(),
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        epsilon=0.2,
        episodes_trained=1000,
    )

    fake_self.model[ACTIONS.index("WAIT"), 0] = 10.0

    transitions = [
        make_transition(step=1, reward=1.0),
        make_transition(step=2, reward=2.0),
        make_transition(step=3, reward=3.0, terminal=True),
    ]

    td_error = update_q_learning(
        self=fake_self,
        transitions=transitions,
    )

    expected = 1.0 + (0.9 * 2.0) + ((0.9**2) * 3.0)

    assert np.isclose(td_error, expected)


def test_short_episode_buffer_flushes_every_transition(monkeypatch):
    processed_steps = []

    def fake_update(self, transitions):
        processed_steps.append(
            transitions[0]["old_game_state"]["step"]
        )
        return 0.0

    monkeypatch.setattr(train_module, "update_q_learning", fake_update)

    fake_self = SimpleNamespace(
        logger=Logger(),
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        epsilon=0.2,
        episodes_trained=1000,
    )

    setup_training(fake_self)

    states = [make_state() for _ in range(4)]

    for index in range(3):
        states[index]["step"] = index + 1
        states[index + 1]["step"] = index + 2
        train_module.game_events_occurred(
            fake_self,
            states[index],
            "WAIT",
            states[index + 1],
            [],
        )

    train_module.end_of_round(
        fake_self,
        states[3],
        "WAIT",
        [],
    )

    assert processed_steps == [1, 2, 3, 4]
    assert len(fake_self.transition_buffer) == 0


def test_each_transition_is_updated_once(monkeypatch):
    processed_steps = []

    def fake_update(self, transitions):
        processed_steps.append(
            transitions[0]["old_game_state"]["step"]
        )
        return 0.0

    monkeypatch.setattr(train_module, "update_q_learning", fake_update)

    fake_self = SimpleNamespace(
        logger=Logger(),
        model=np.zeros((len(ACTIONS), FEATURE_DIM)),
        epsilon=0.2,
        episodes_trained=1000,
    )

    setup_training(fake_self)

    states = [make_state() for _ in range(8)]

    for index in range(6):
        states[index]["step"] = index + 1
        states[index + 1]["step"] = index + 2
        train_module.game_events_occurred(
            fake_self,
            states[index],
            "WAIT",
            states[index + 1],
            [],
        )

    states[6]["step"] = 7
    train_module.end_of_round(
        fake_self,
        states[6],
        "WAIT",
        [],
    )

    assert processed_steps == [1, 2, 3, 4, 5, 6, 7]
    assert len(fake_self.transition_buffer) == 0