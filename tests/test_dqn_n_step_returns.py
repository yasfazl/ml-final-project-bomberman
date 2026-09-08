from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from agent_code.dqn_agent import train as dqn_train
from agent_code.dqn_agent.model import DQN
from agent_code.dqn_agent.replay_buffer import NStepAccumulator
from agent_code.q_learning_agent.callbacks import ACTIONS


class FixedValues(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer(
            "values",
            torch.tensor(values, dtype=torch.float32),
        )

    def forward(self, features):
        return self.values.unsqueeze(0).expand(features.shape[0], -1)


class Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def step_arrays(value):
    state = np.full(52, value, dtype=np.float32)
    next_state = np.full(52, value + 1, dtype=np.float32)
    mask = np.array([True, False, True, False, True, False])
    return state, next_state, mask


def append_step(accumulator, value, reward, done=False):
    state, next_state, mask = step_arrays(value)
    return accumulator.append(
        state=state,
        action=int(value) % len(ACTIONS),
        reward=reward,
        next_state=next_state,
        next_action_mask=mask,
        done=done,
    )


def test_n_step_accumulator_emits_discounted_five_step_return():
    accumulator = NStepAccumulator(n_steps=5, gamma=0.9)
    for offset, reward in enumerate([1.0, 2.0, 3.0, 4.0]):
        assert append_step(accumulator, offset, reward) == []

    ready = append_step(accumulator, 4, 5.0)

    assert len(ready) == 1
    transition = ready[0]
    assert transition.reward == pytest.approx(
        1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 4.0
        + 0.9**4 * 5.0
    )
    assert transition.n_steps == 5
    assert transition.action == 0
    assert not transition.done
    np.testing.assert_array_equal(transition.state, np.zeros(52))
    np.testing.assert_array_equal(
        transition.next_state,
        np.full(52, 5.0),
    )
    assert len(accumulator) == 4


def test_terminal_step_flushes_every_remaining_prefix():
    accumulator = NStepAccumulator(n_steps=5, gamma=0.9)
    assert append_step(accumulator, 0, 2.0) == []

    ready = append_step(accumulator, 1, 4.0, done=True)

    assert len(ready) == 2
    assert ready[0].reward == pytest.approx(2.0 + 0.9 * 4.0)
    assert ready[0].n_steps == 2
    assert ready[0].done
    assert ready[1].reward == pytest.approx(4.0)
    assert ready[1].n_steps == 1
    assert ready[1].done
    assert len(accumulator) == 0


def test_one_step_accumulator_matches_original_replay_semantics():
    accumulator = NStepAccumulator(n_steps=1, gamma=0.9)

    ready = append_step(accumulator, 2, -3.5)

    assert len(ready) == 1
    assert ready[0].reward == pytest.approx(-3.5)
    assert ready[0].n_steps == 1
    assert not ready[0].done
    assert len(accumulator) == 0


def test_double_dqn_target_uses_each_transition_horizon():
    result = dqn_train.double_dqn_targets(
        FixedValues([100.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        FixedValues([500.0, 7.0, 6.0, 5.0, 4.0, 3.0]),
        rewards=torch.tensor([2.0, 2.0]),
        next_states=torch.zeros(2, 52),
        next_action_masks=torch.tensor(
            [
                [False, True, True, True, True, True],
                [False, True, True, True, True, True],
            ]
        ),
        dones=torch.tensor([False, True]),
        n_steps=torch.tensor([3, 3]),
        gamma=0.9,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([2.0 + 0.9**3 * 7.0, 2.0]),
    )


def test_setup_resets_v3_optimizer_but_resumes_v4_optimizer():
    def make_agent(algorithm, horizon):
        policy = DQN(52, len(ACTIONS))
        target = DQN(52, len(ACTIONS))
        target.load_state_dict(policy.state_dict())
        return SimpleNamespace(
            policy_net=policy,
            target_net=target,
            pending_optimizer_state={"invalid": "old-v3-state"},
            loaded_algorithm=algorithm,
            loaded_n_step_return=horizon,
            logger=Logger(),
            episodes_trained=500,
            epsilon=0.05,
            endgame_epsilon=0.05,
            optimizer_steps=100,
            device=torch.device("cpu"),
        )

    v3_agent = make_agent("safe_attack_staging_masked_double_dqn_v3", 1)
    dqn_train.setup_training(v3_agent)
    assert not v3_agent.optimizer.state

    v4_agent = make_agent(dqn_train.ALGORITHM, dqn_train.N_STEP_RETURN)
    clean_optimizer = torch.optim.Adam(
        [v4_agent.policy_net.network[0].weight],
        lr=dqn_train.LEARNING_RATE,
    )
    v4_agent.pending_optimizer_state = clean_optimizer.state_dict()
    dqn_train.setup_training(v4_agent)
    assert v4_agent.optimizer.param_groups[0]["lr"] == pytest.approx(
        dqn_train.LEARNING_RATE
    )

