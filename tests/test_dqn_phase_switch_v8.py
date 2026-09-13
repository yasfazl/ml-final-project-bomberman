"""Tests for the crate-aware V3-75/Trial-8 policy switch."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from agent_code.dqn_agent import callbacks
from agent_code.dqn_agent.model import DQN


class Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


class FixedValues(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer(
            "values",
            torch.tensor(values, dtype=torch.float32),
        )

    def forward(self, features):
        return self.values.unsqueeze(0).expand(features.shape[0], -1)


def game_state(*, crates):
    field = np.zeros((11, 11), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    if crates:
        field[3, 3] = 1
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


def agent(*, training=False, endgame_policy=True):
    return SimpleNamespace(
        train=training,
        logger=Logger(),
        device=torch.device("cpu"),
        epsilon=0.0,
        endgame_epsilon=0.0,
        policy_net=FixedValues([10, 0, 0, 0, 0, 0]),
        endgame_policy_net=(
            FixedValues([0, 10, 0, 0, 0, 0])
            if endgame_policy
            else None
        ),
        last_own_bomb_position=None,
        last_seen_round=None,
        consecutive_safe_waits=0,
        last_wait_position=None,
        recent_positions=[],
    )


def test_phase_switch_uses_v3_while_a_crate_remains():
    test_agent = agent()

    selected, name = callbacks._policy_net_for_state(
        test_agent,
        game_state(crates=True),
    )

    assert selected is test_agent.policy_net
    assert name == "v3_75"


def test_phase_switch_uses_trial8_when_no_crates_remain():
    test_agent = agent()

    selected, name = callbacks._policy_net_for_state(
        test_agent,
        game_state(crates=False),
    )

    assert selected is test_agent.endgame_policy_net
    assert name == "trial8_endgame"


def test_phase_switch_uses_v3_during_training():
    test_agent = agent(training=True)

    selected, name = callbacks._policy_net_for_state(
        test_agent,
        game_state(crates=False),
    )

    assert selected is test_agent.policy_net
    assert name == "v3_75"


def test_phase_switch_falls_back_when_trial8_is_unavailable():
    test_agent = agent(endgame_policy=False)

    selected, name = callbacks._policy_net_for_state(
        test_agent,
        game_state(crates=False),
    )

    assert selected is test_agent.policy_net
    assert name == "v3_75"


def test_act_uses_the_selected_network(monkeypatch):
    monkeypatch.setattr(
        callbacks,
        "state_to_features",
        lambda _state: np.zeros(callbacks.FEATURE_DIM, dtype=np.float32),
    )
    monkeypatch.setattr(
        callbacks,
        "candidate_action_indices",
        lambda _agent, _state: list(range(len(callbacks.ACTIONS))),
    )
    monkeypatch.setattr(
        callbacks,
        "endgame_opponent_pursuit_active",
        lambda _state: False,
    )
    monkeypatch.setattr(
        callbacks,
        "endgame_safe_attack_active",
        lambda _state: False,
    )
    monkeypatch.setattr(
        callbacks,
        "_remember_own_bomb_if_selected",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        callbacks,
        "_record_anti_stall_choice",
        lambda *_args: None,
    )

    test_agent = agent()

    assert callbacks.act(test_agent, game_state(crates=True)) == "UP"
    assert callbacks.act(test_agent, game_state(crates=False)) == "RIGHT"


def test_trial8_loader_is_read_only(tmp_path, monkeypatch):
    torch.manual_seed(17)
    trial8 = DQN(callbacks.FEATURE_DIM, len(callbacks.ACTIONS))
    checkpoint_path = tmp_path / "dqn_model_trial8.pt"
    torch.save(
        {
            "checkpoint_version": callbacks.CHECKPOINT_VERSION,
            "feature_dim": callbacks.FEATURE_DIM,
            "actions": callbacks.ACTIONS,
            "policy_state": trial8.state_dict(),
        },
        checkpoint_path,
    )
    original = checkpoint_path.read_bytes()
    monkeypatch.setattr(
        callbacks,
        "ENDGAME_DQN_MODEL_PATH",
        checkpoint_path,
    )
    test_agent = SimpleNamespace(
        device=torch.device("cpu"),
        logger=Logger(),
    )
    torch.manual_seed(1234)
    rng_state = torch.random.get_rng_state().clone()

    callbacks._load_endgame_policy(test_agent)

    assert test_agent.endgame_policy_net is not None
    assert not test_agent.endgame_policy_net.training
    assert all(
        not parameter.requires_grad
        for parameter in test_agent.endgame_policy_net.parameters()
    )
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
    assert checkpoint_path.read_bytes() == original
