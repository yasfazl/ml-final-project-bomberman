import numpy as np

from agent_code.dqn_agent import callbacks as dqn_callbacks
from agent_code.dqn_agent import train as dqn_train


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def attack_state(
    *,
    position=(5, 5),
    opponent_position=(5, 2),
    bomb_available=True,
    coins=None,
    bombs=None,
    field=None,
    explosion_map=None,
):
    field = open_field() if field is None else field
    explosion_map = (
        np.zeros_like(field)
        if explosion_map is None
        else explosion_map
    )
    return {
        "round": 1,
        "step": 200,
        "field": field,
        "self": ("test", 0, bomb_available, position),
        "others": [("enemy", 0, True, opponent_position)],
        "bombs": [] if bombs is None else bombs,
        "coins": [] if coins is None else coins,
        "explosion_map": explosion_map,
    }


def test_safe_attack_path_reports_safe_bomb_now(monkeypatch):
    monkeypatch.setattr(
        dqn_callbacks,
        "bomb_has_robust_escape_route",
        lambda _state: True,
    )
    assert dqn_callbacks.safe_opponent_bombing_path(
        attack_state()
    ) == (None, 0, True)


def test_safe_attack_path_finds_nearest_robust_target_tile(monkeypatch):
    monkeypatch.setattr(
        dqn_callbacks,
        "bomb_has_robust_escape_route",
        lambda state: tuple(state["self"][3]) == (5, 4),
    )
    assert dqn_callbacks.safe_opponent_bombing_path(
        attack_state()
    ) == (0, 1, False)


def test_safe_attack_path_is_inactive_without_a_target(monkeypatch):
    monkeypatch.setattr(
        dqn_callbacks,
        "bomb_has_robust_escape_route",
        lambda _state: True,
    )
    assert dqn_callbacks.safe_opponent_bombing_path(
        attack_state(opponent_position=(8, 8))
    ) == (None, None, False)


def test_safe_attack_path_returns_inactive_when_no_robust_tile(
    monkeypatch,
):
    monkeypatch.setattr(
        dqn_callbacks,
        "bomb_has_robust_escape_route",
        lambda _state: False,
    )
    assert dqn_callbacks.safe_opponent_bombing_path(
        attack_state()
    ) == (None, None, False)


def test_safe_attack_features_have_stable_indices(monkeypatch):
    monkeypatch.setattr(
        dqn_callbacks,
        "bomb_has_robust_escape_route",
        lambda state: tuple(state["self"][3]) == (5, 4),
    )
    features = dqn_callbacks.state_to_features(attack_state())
    assert features.shape == (52,)
    assert not features[39:45].any()
    assert features[45] == 1.0
    assert features[46] == 1.0
    assert not features[47:50].any()
    assert features[50] == 1 / 22
    assert features[51] == 0.0


def test_safe_attack_features_are_zero_in_every_other_mode(monkeypatch):
    def fail_if_called(_state):
        raise AssertionError("expensive attack search should be gated")

    monkeypatch.setattr(
        dqn_callbacks,
        "bomb_has_robust_escape_route",
        fail_if_called,
    )
    crate_field = open_field()
    crate_field[7, 7] = 1
    explosion_map = np.zeros((11, 11), dtype=int)
    explosion_map[5, 5] = 1
    states = [
        attack_state(coins=[(7, 7)]),
        attack_state(field=crate_field),
        attack_state(bombs=[((7, 7), 3)]),
        attack_state(explosion_map=explosion_map),
        attack_state(bomb_available=False),
        attack_state(opponent_position=(8, 8)),
    ]
    for state in states:
        features = dqn_callbacks.state_to_features(state)
        assert not features[45:52].any()


def test_safe_attack_navigation_shaping_uses_our_move(monkeypatch):
    def fake_path(state):
        position = tuple(state["self"][3])
        mapping = {
            (5, 5): (0, 2, False),
            (5, 4): (0, 1, False),
            (6, 5): (None, None, False),
        }
        return mapping[position]

    monkeypatch.setattr(
        dqn_train,
        "safe_opponent_bombing_path",
        fake_path,
    )
    toward_events = []
    away_events = []
    dqn_train.add_safe_attack_navigation_event(
        attack_state(position=(5, 5)),
        attack_state(position=(5, 4)),
        toward_events,
    )
    dqn_train.add_safe_attack_navigation_event(
        attack_state(position=(5, 5)),
        attack_state(position=(6, 5)),
        away_events,
    )
    assert toward_events == [
        dqn_train.MOVED_TOWARD_SAFE_ATTACK_POSITION
    ]
    assert away_events == [
        dqn_train.MOVED_AWAY_FROM_SAFE_ATTACK_POSITION
    ]


def test_safe_attack_wait_penalty_is_strictly_gated(monkeypatch):
    monkeypatch.setattr(
        dqn_train,
        "endgame_safe_attack_active",
        lambda state: bool(state.get("attack_mode")),
    )
    active_state = {"attack_mode": True}
    inactive_state = {"attack_mode": False}
    events = []
    dqn_train.add_safe_attack_waiting_event(
        active_state,
        "WAIT",
        events,
    )
    dqn_train.add_safe_attack_waiting_event(
        inactive_state,
        "WAIT",
        events,
    )
    dqn_train.add_safe_attack_waiting_event(
        active_state,
        "UP",
        events,
    )
    assert events == [dqn_train.WAITED_DURING_SAFE_ATTACK]
