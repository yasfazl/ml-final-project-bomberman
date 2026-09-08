from types import SimpleNamespace
import pickle

import numpy as np
import torch
from torch import nn

from agent_code.dqn_agent import callbacks as dqn_callbacks
from agent_code.dqn_agent import train as dqn_train
from agent_code.dqn_agent.model import DQN, initialize_from_linear_q
from agent_code.dqn_agent.replay_buffer import ReplayBuffer
from agent_code.q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM as BASE_FEATURE_DIM,
    state_to_features as base_state_to_features,
)


DQN_FEATURE_DIM = dqn_callbacks.FEATURE_DIM
PREVIOUS_FEATURE_DIM = dqn_callbacks.PREVIOUS_FEATURE_DIM


class Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def open_field(size=11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def endgame_state(
    *,
    position=(5, 5),
    opponent_position=(5, 1),
    coins=None,
    bombs=None,
    field=None,
    explosion_map=None,
):
    field = open_field() if field is None else field
    explosion_map = (
        np.zeros_like(field) if explosion_map is None else explosion_map
    )
    return {
        "round": 1,
        "step": 300,
        "field": field,
        "self": ("test", 0, True, position),
        "others": [("enemy", 0, True, opponent_position)],
        "bombs": [] if bombs is None else bombs,
        "coins": [] if coins is None else coins,
        "explosion_map": explosion_map,
    }


def efficient_step_state():
    field = open_field()
    field[5, 2] = 1
    field[2, 4] = 1
    field[8, 4] = 1
    state = endgame_state(field=field)
    state["others"] = []
    state["step"] = 1
    return state


def bare_agent():
    return SimpleNamespace(
        logger=Logger(),
        train=False,
        last_own_bomb_position=None,
        last_seen_round=None,
        consecutive_safe_waits=0,
        last_wait_position=None,
        recent_positions=[],
    )


def test_network_output_shape():
    model = DQN(DQN_FEATURE_DIM, len(ACTIONS))
    assert model(torch.zeros(7, DQN_FEATURE_DIM)).shape == (7, len(ACTIONS))


def test_linear_warm_start_is_exact():
    random = np.random.default_rng(42)
    weights = random.normal(size=(len(ACTIONS), BASE_FEATURE_DIM)).astype(
        np.float32
    )
    features = random.normal(size=(32, BASE_FEATURE_DIM)).astype(np.float32)
    model = DQN(BASE_FEATURE_DIM, len(ACTIONS))
    initialize_from_linear_q(model, weights)
    with torch.no_grad():
        actual = model(torch.from_numpy(features)).numpy()
    np.testing.assert_allclose(
        actual,
        features @ weights.T,
        rtol=2e-6,
        atol=2e-5,
    )


def test_setup_warm_start_does_not_modify_teacher(tmp_path, monkeypatch):
    random = np.random.default_rng(7)
    weights = random.normal(size=(len(ACTIONS), BASE_FEATURE_DIM))
    teacher_path = tmp_path / "q_model.pkl"
    teacher_path.write_bytes(pickle.dumps({"weights": weights}))
    original = teacher_path.read_bytes()
    monkeypatch.setattr(
        dqn_callbacks,
        "DQN_MODEL_PATH",
        tmp_path / "dqn_model.pt",
    )
    monkeypatch.setattr(
        dqn_callbacks,
        "LINEAR_TEACHER_PATH",
        teacher_path,
    )
    agent = bare_agent()
    dqn_callbacks.setup(agent)

    features = random.normal(size=(5, DQN_FEATURE_DIM)).astype(np.float32)
    with torch.no_grad():
        actual = agent.policy_net(torch.from_numpy(features)).numpy()
    expected = features[:, :BASE_FEATURE_DIM] @ weights.astype(np.float32).T
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
    assert teacher_path.read_bytes() == original
    assert agent.warm_started


def test_dqn_uses_v22_crate_deferral_filter():
    candidates = dqn_callbacks.candidate_action_indices(
        bare_agent(),
        efficient_step_state(),
    )
    assert candidates == [ACTIONS.index("UP")]


def test_endgame_features_preserve_base_vector():
    state = endgame_state()
    features = dqn_callbacks.state_to_features(state)
    np.testing.assert_array_equal(
        features[:BASE_FEATURE_DIM],
        base_state_to_features(state),
    )
    assert features.shape == (DQN_FEATURE_DIM,)
    assert features[39] == 1.0
    assert features[40] == 1.0
    assert features[41:44].sum() == 0.0
    assert features[44] == 3 / 22
    assert not features[PREVIOUS_FEATURE_DIM:].any()


def test_endgame_features_are_strictly_gated():
    field_with_crate = open_field()
    field_with_crate[7, 7] = 1
    explosion_map = np.zeros((11, 11), dtype=int)
    explosion_map[7, 7] = 1
    states = [
        endgame_state(coins=[(7, 7)]),
        endgame_state(field=field_with_crate),
        endgame_state(bombs=[((7, 7), 3)]),
        endgame_state(explosion_map=explosion_map),
        endgame_state(opponent_position=(5, 2)),
    ]
    for state in states:
        assert not dqn_callbacks.state_to_features(state)[39:45].any()


def test_safe_attack_features_do_not_change_old_endgame_features():
    state = endgame_state(opponent_position=(5, 2))
    features = dqn_callbacks.state_to_features(state)
    assert not features[39:45].any()
    assert features[45] == 1.0
    assert features[51] == 1.0


def test_endgame_cycle_guard_breaks_repeated_backtrack():
    agent = bare_agent()
    agent.recent_positions = [(5, 5), (6, 5)]
    filtered = dqn_callbacks._endgame_cycle_candidate_indices(
        agent,
        endgame_state(),
        list(range(len(ACTIONS))),
    )
    assert ACTIONS.index("UP") in filtered
    assert ACTIONS.index("RIGHT") not in filtered


def test_endgame_wait_guard_requires_repeated_waits():
    agent = bare_agent()
    agent.consecutive_safe_waits = 2
    filtered = dqn_callbacks._endgame_cycle_candidate_indices(
        agent,
        endgame_state(),
        list(range(len(ACTIONS))),
    )
    assert ACTIONS.index("WAIT") not in filtered
    assert ACTIONS.index("UP") in filtered


def test_endgame_navigation_events():
    old = endgame_state(position=(5, 5))
    toward_events = []
    away_events = []
    dqn_train.add_endgame_opponent_navigation_event(
        old,
        endgame_state(position=(5, 4)),
        toward_events,
    )
    dqn_train.add_endgame_opponent_navigation_event(
        old,
        endgame_state(position=(6, 5)),
        away_events,
    )
    assert toward_events == [dqn_train.MOVED_TOWARD_OPPONENT]
    assert away_events == [dqn_train.MOVED_AWAY_FROM_OPPONENT]


def test_endgame_wait_event_is_mode_specific():
    events = []
    dqn_train.add_endgame_waiting_event(endgame_state(), "WAIT", events)
    assert events == [dqn_train.WAITED_DURING_ENDGAME]

    crate_field = open_field()
    crate_field[7, 7] = 1
    events = []
    dqn_train.add_endgame_waiting_event(
        endgame_state(field=crate_field),
        "WAIT",
        events,
    )
    assert events == []


def test_replay_buffer_makes_defensive_copies():
    replay = ReplayBuffer(3, seed=1)
    state = np.arange(DQN_FEATURE_DIM, dtype=np.float32)
    next_state = state + 1
    mask = np.ones(len(ACTIONS), dtype=bool)
    replay.add(state, 2, 3.5, next_state, mask, False)
    state[:] = -100
    next_state[:] = -200
    mask[:] = False
    transition = replay.sample(1)[0]
    assert transition.state[0] == 0
    assert transition.next_state[0] == 1
    assert transition.next_action_mask.all()


class FixedValues(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer(
            "values",
            torch.tensor(values, dtype=torch.float32),
        )

    def forward(self, features):
        return self.values.unsqueeze(0).expand(features.shape[0], -1)


def test_double_dqn_target_ignores_invalid_high_value_action():
    result = dqn_train.double_dqn_targets(
        FixedValues([100.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        FixedValues([500.0, 7.0, 6.0, 5.0, 4.0, 3.0]),
        rewards=torch.tensor([2.0]),
        next_states=torch.zeros(1, DQN_FEATURE_DIM),
        next_action_masks=torch.tensor(
            [[False, True, True, True, True, True]]
        ),
        dones=torch.tensor([False]),
        gamma=0.9,
    )
    torch.testing.assert_close(result, torch.tensor([8.3]))


def test_adapter_update_cannot_change_normal_mode_outputs():
    torch.manual_seed(12)
    policy = DQN(DQN_FEATURE_DIM, len(ACTIONS))
    target = DQN(DQN_FEATURE_DIM, len(ACTIONS))
    target.load_state_dict(policy.state_dict())
    agent = SimpleNamespace(
        policy_net=policy,
        target_net=target,
        pending_optimizer_state=None,
        logger=Logger(),
        episodes_trained=0,
        epsilon=0.05,
        endgame_epsilon=0.20,
        optimizer_steps=0,
        device=torch.device("cpu"),
    )
    dqn_train.setup_training(agent)
    agent.replay_warmup = 1
    agent.batch_size = 4
    agent.target_sync_interval = 100

    normal_states = torch.randn(5, DQN_FEATURE_DIM)
    normal_states[:, PREVIOUS_FEATURE_DIM:] = 0.0
    with torch.no_grad():
        before = policy(normal_states).clone()
    protected_before = policy.network[0].weight[
        :, :PREVIOUS_FEATURE_DIM
    ].detach().clone()
    adapter_before = policy.network[0].weight[
        :, PREVIOUS_FEATURE_DIM:
    ].detach().clone()
    for index in range(4):
        state = np.zeros(DQN_FEATURE_DIM, dtype=np.float32)
        state[0] = 1.0
        state[45] = 1.0
        state[46 + index] = 1.0
        state[50] = 0.25
        agent.replay_buffer.add(
            state,
            index,
            5.0,
            np.zeros(DQN_FEATURE_DIM, dtype=np.float32),
            np.zeros(len(ACTIONS), dtype=bool),
            True,
        )
    assert dqn_train.optimize_model(agent) is not None
    with torch.no_grad():
        after = policy(normal_states)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(
        policy.network[0].weight[:, :PREVIOUS_FEATURE_DIM],
        protected_before,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        policy.network[0].weight[:, PREVIOUS_FEATURE_DIM:],
        adapter_before,
    )


def test_v1_checkpoint_migration_preserves_old_q_values(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(9)
    old_policy = DQN(BASE_FEATURE_DIM, len(ACTIONS))
    old_target = DQN(BASE_FEATURE_DIM, len(ACTIONS))
    old_target.load_state_dict(old_policy.state_dict())
    checkpoint_path = tmp_path / "dqn_model.pt"
    torch.save(
        {
            "checkpoint_version": 1,
            "feature_dim": BASE_FEATURE_DIM,
            "actions": ACTIONS,
            "policy_state": old_policy.state_dict(),
            "target_state": old_target.state_dict(),
            "optimizer_state": {"old": "incompatible"},
            "epsilon": 0.05,
            "episodes_trained": 500,
            "environment_steps": 1000,
            "optimizer_steps": 500,
            "warm_started": True,
        },
        checkpoint_path,
    )
    monkeypatch.setattr(dqn_callbacks, "DQN_MODEL_PATH", checkpoint_path)
    monkeypatch.setattr(
        dqn_callbacks,
        "LINEAR_TEACHER_PATH",
        tmp_path / "missing.pkl",
    )
    features = torch.randn(12, BASE_FEATURE_DIM)
    expanded = torch.cat(
        [
            features,
            torch.randn(12, DQN_FEATURE_DIM - BASE_FEATURE_DIM),
        ],
        dim=1,
    )
    with torch.no_grad():
        expected = old_policy(features)
    agent = bare_agent()
    dqn_callbacks.setup(agent)
    with torch.no_grad():
        actual = agent.policy_net(expanded)
    torch.testing.assert_close(actual, expected)
    assert agent.feature_migrated
    assert agent.pending_optimizer_state is None


def test_v2_checkpoint_migration_preserves_all_old_q_values(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(19)
    old_policy = DQN(PREVIOUS_FEATURE_DIM, len(ACTIONS))
    old_target = DQN(PREVIOUS_FEATURE_DIM, len(ACTIONS))
    old_target.load_state_dict(old_policy.state_dict())
    checkpoint_path = tmp_path / "dqn_model.pt"
    torch.save(
        {
            "checkpoint_version": 2,
            "feature_dim": PREVIOUS_FEATURE_DIM,
            "actions": ACTIONS,
            "policy_state": old_policy.state_dict(),
            "target_state": old_target.state_dict(),
            "optimizer_state": {"old": "incompatible"},
            "epsilon": 0.05,
            "episodes_trained": 300,
            "environment_steps": 5000,
            "optimizer_steps": 4000,
            "warm_started": True,
            "endgame_epsilon": 0.05,
        },
        checkpoint_path,
    )
    monkeypatch.setattr(dqn_callbacks, "DQN_MODEL_PATH", checkpoint_path)
    monkeypatch.setattr(
        dqn_callbacks,
        "LINEAR_TEACHER_PATH",
        tmp_path / "missing.pkl",
    )
    old_features = torch.randn(12, PREVIOUS_FEATURE_DIM)
    expanded_features = torch.cat(
        [
            old_features,
            torch.randn(
                12,
                DQN_FEATURE_DIM - PREVIOUS_FEATURE_DIM,
            ),
        ],
        dim=1,
    )
    with torch.no_grad():
        expected = old_policy(old_features)

    agent = bare_agent()
    dqn_callbacks.setup(agent)

    with torch.no_grad():
        actual = agent.policy_net(expanded_features)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        agent.policy_net.network[0].weight[
            :, :PREVIOUS_FEATURE_DIM
        ],
        old_policy.network[0].weight,
    )
    assert not agent.policy_net.network[0].weight[
        :, PREVIOUS_FEATURE_DIM:
    ].count_nonzero()
    assert agent.feature_migrated
    assert agent.pending_optimizer_state is None


def test_checkpoint_round_trip(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "dqn_model.pt"
    teacher_path = tmp_path / "q_model.pkl"
    teacher_path.write_bytes(
        pickle.dumps(
            {"weights": np.zeros((len(ACTIONS), BASE_FEATURE_DIM))}
        )
    )
    monkeypatch.setattr(dqn_callbacks, "DQN_MODEL_PATH", checkpoint_path)
    monkeypatch.setattr(
        dqn_callbacks,
        "LINEAR_TEACHER_PATH",
        teacher_path,
    )
    first = bare_agent()
    dqn_callbacks.setup(first)
    dqn_train.setup_training(first)
    first.episodes_trained = 23
    first.environment_steps = 456
    first.optimizer_steps = 78
    first.endgame_epsilon = 0.12
    dqn_train.save_checkpoint(first)

    saved = dqn_callbacks._load_torch_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )
    assert saved["algorithm"] == dqn_train.ALGORITHM
    assert saved["training_scope"] == dqn_train.TRAINING_SCOPE
    assert saved["n_step_return"] == 5

    second = bare_agent()
    dqn_callbacks.setup(second)
    assert second.episodes_trained == 23
    assert second.environment_steps == 456
    assert second.optimizer_steps == 78
    assert second.endgame_epsilon == 0.12
    assert second.pending_optimizer_state is not None
    assert second.loaded_algorithm == dqn_train.ALGORITHM
    assert second.loaded_n_step_return == 5
