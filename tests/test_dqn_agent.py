from types import SimpleNamespace
import pickle

import numpy as np
import torch
from torch import nn

from agent_code.dqn_agent import callbacks as dqn_callbacks
from agent_code.dqn_agent import train as dqn_train
from agent_code.dqn_agent.model import DQN, initialize_from_linear_q
from agent_code.dqn_agent.replay_buffer import ReplayBuffer
from agent_code.q_learning_agent.callbacks import ACTIONS, FEATURE_DIM


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


def efficient_step_state():
    field = open_field()
    # Bombing at (5, 5) hits one crate; moving UP enables three.
    field[5, 2] = 1
    field[2, 4] = 1
    field[8, 4] = 1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test", 0, True, (5, 5)),
        "others": [],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def bare_agent():
    return SimpleNamespace(
        logger=Logger(),
        train=False,
        last_own_bomb_position=None,
        last_seen_round=None,
        consecutive_safe_waits=0,
        last_wait_position=None,
    )


def test_network_output_shape():
    model = DQN(FEATURE_DIM, len(ACTIONS))
    output = model(torch.zeros(7, FEATURE_DIM))
    assert output.shape == (7, len(ACTIONS))


def test_linear_warm_start_is_exact():
    random = np.random.default_rng(42)
    weights = random.normal(size=(len(ACTIONS), FEATURE_DIM)).astype(
        np.float32
    )
    features = random.normal(size=(32, FEATURE_DIM)).astype(np.float32)
    model = DQN(FEATURE_DIM, len(ACTIONS))
    initialize_from_linear_q(model, weights)

    with torch.no_grad():
        actual = model(torch.from_numpy(features)).numpy()
    expected = features @ weights.T

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_setup_warm_starts_from_teacher_without_modifying_it(
    tmp_path,
    monkeypatch,
):
    random = np.random.default_rng(7)
    weights = random.normal(size=(len(ACTIONS), FEATURE_DIM)).astype(
        np.float64
    )
    teacher_path = tmp_path / "q_model.pkl"
    teacher_path.write_bytes(
        pickle.dumps(
            {
                "weights": weights,
                "epsilon": 0.05,
                "episodes_trained": 3701,
            }
        )
    )
    original_bytes = teacher_path.read_bytes()
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

    features = random.normal(size=(5, FEATURE_DIM)).astype(np.float32)
    with torch.no_grad():
        actual = agent.policy_net(torch.from_numpy(features)).numpy()
    np.testing.assert_allclose(
        actual,
        features @ weights.astype(np.float32).T,
        rtol=2e-6,
        atol=2e-5,
    )
    assert agent.warm_started
    assert teacher_path.read_bytes() == original_bytes
    assert not (tmp_path / "dqn_model.pt").exists()


def test_dqn_uses_exact_v22_candidate_filters():
    agent = bare_agent()
    candidates = dqn_callbacks.candidate_action_indices(
        agent,
        efficient_step_state(),
    )
    assert candidates == [ACTIONS.index("UP")]


def test_replay_buffer_makes_defensive_copies():
    replay = ReplayBuffer(capacity=3, seed=1)
    state = np.arange(FEATURE_DIM, dtype=np.float32)
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
    assert transition.action == 2
    assert transition.reward == 3.5


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
    policy = FixedValues([100.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    target = FixedValues([500.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    mask = torch.tensor(
        [[False, True, True, True, True, True]],
        dtype=torch.bool,
    )
    result = dqn_train.double_dqn_targets(
        policy,
        target,
        rewards=torch.tensor([2.0]),
        next_states=torch.zeros(1, FEATURE_DIM),
        next_action_masks=mask,
        dones=torch.tensor([False]),
        gamma=0.9,
    )
    torch.testing.assert_close(result, torch.tensor([2.0 + 0.9 * 7.0]))


def test_one_replay_update_changes_policy_but_not_unsynced_target():
    policy = DQN(FEATURE_DIM, len(ACTIONS))
    target = DQN(FEATURE_DIM, len(ACTIONS))
    target.load_state_dict(policy.state_dict())
    replay = ReplayBuffer(capacity=10, seed=4)
    for index in range(4):
        replay.add(
            np.full(FEATURE_DIM, index + 1, dtype=np.float32),
            index % len(ACTIONS),
            10.0,
            np.zeros(FEATURE_DIM, dtype=np.float32),
            np.zeros(len(ACTIONS), dtype=bool),
            True,
        )
    agent = SimpleNamespace(
        policy_net=policy,
        target_net=target,
        replay_buffer=replay,
        optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3),
        optimizer_steps=0,
        device=torch.device("cpu"),
        replay_warmup=1,
        batch_size=4,
        target_sync_interval=100,
    )
    policy_before = [item.detach().clone() for item in policy.parameters()]
    target_before = [item.detach().clone() for item in target.parameters()]

    loss = dqn_train.optimize_model(agent)

    assert loss is not None and np.isfinite(loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(policy_before, policy.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, target.parameters())
    )


def test_checkpoint_round_trip_preserves_outputs_and_counters(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "dqn_model.pt"
    teacher_path = tmp_path / "q_model.pkl"
    teacher_path.write_bytes(
        pickle.dumps(
            {"weights": np.zeros((len(ACTIONS), FEATURE_DIM))}
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
    with torch.no_grad():
        first.policy_net.network[4].bias.add_(0.75)
    first.target_net.load_state_dict(first.policy_net.state_dict())
    first.epsilon = 0.12
    first.episodes_trained = 23
    first.environment_steps = 456
    first.optimizer_steps = 78
    dqn_train.save_checkpoint(first)

    features = torch.randn(3, FEATURE_DIM)
    with torch.no_grad():
        expected = first.policy_net(features).clone()

    second = bare_agent()
    dqn_callbacks.setup(second)
    with torch.no_grad():
        actual = second.policy_net(features)

    torch.testing.assert_close(actual, expected)
    assert second.epsilon == 0.12
    assert second.episodes_trained == 23
    assert second.environment_steps == 456
    assert second.optimizer_steps == 78
    assert second.pending_optimizer_state is not None

