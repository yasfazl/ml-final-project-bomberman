from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_code.dqn_agent import potential_shaping
from agent_code.dqn_agent import train as dqn_train
from agent_code.dqn_agent.model import DQN
from tools.train_dqn_potential_v7 import verify_adapter_only_update


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


def test_potential_is_zero_outside_relevant_endgame():
    crate_field = open_field()
    crate_field[7, 7] = 1
    without_opponent = attack_state()
    without_opponent["others"] = []
    explosion_map = np.zeros_like(crate_field)
    explosion_map[7, 7] = 1

    assert potential_shaping.attack_readiness_potential(
        attack_state(coins=[(7, 7)])
    ) == 0.0
    assert potential_shaping.attack_readiness_potential(
        attack_state(field=crate_field)
    ) == 0.0
    assert potential_shaping.attack_readiness_potential(
        without_opponent
    ) == 0.0
    assert potential_shaping.attack_readiness_potential(
        attack_state(bombs=[((7, 7), 3)])
    ) == 0.0
    assert potential_shaping.attack_readiness_potential(
        attack_state(explosion_map=explosion_map)
    ) == 0.0
    assert potential_shaping.attack_readiness_potential(None) == 0.0


def test_potential_is_bounded_and_orders_attack_readiness(monkeypatch):
    distances = {
        (5, 5): 3,
        (5, 4): 2,
        (6, 5): 4,
    }

    def fake_path(state):
        return 0, distances[tuple(state["self"][3])], False

    monkeypatch.setattr(
        potential_shaping.callbacks,
        "safe_opponent_bombing_path",
        fake_path,
    )
    old_state = attack_state(position=(5, 5))
    toward_state = attack_state(position=(5, 4))
    away_state = attack_state(position=(6, 5))

    old_value = potential_shaping.attack_readiness_potential(old_state)
    toward_value = potential_shaping.attack_readiness_potential(toward_state)
    away_value = potential_shaping.attack_readiness_potential(away_state)

    assert 0.0 <= away_value < old_value < toward_value <= 1.0

    toward_reward = potential_shaping.potential_difference(
        old_state,
        toward_state,
    )
    waiting_reward = potential_shaping.potential_difference(
        old_state,
        old_state,
    )
    away_reward = potential_shaping.potential_difference(
        old_state,
        away_state,
    )
    assert toward_reward > waiting_reward > away_reward


def test_safe_targeted_bomb_recreates_approximately_four_credit(
    monkeypatch,
):
    monkeypatch.setattr(
        potential_shaping.callbacks,
        "safe_opponent_bombing_path",
        lambda _state: (None, 0, True),
    )
    monkeypatch.setattr(
        potential_shaping.game_utils,
        "nearest_safe_path",
        lambda _state: (0, 1),
    )
    old_state = attack_state()
    post_bomb_state = attack_state(
        bomb_available=False,
        bombs=[((5, 5), 3)],
    )

    assert potential_shaping.attack_readiness_potential(
        old_state
    ) == potential_shaping.READY_TO_BOMB_POTENTIAL
    assert potential_shaping.attack_readiness_potential(
        post_bomb_state
    ) == potential_shaping.POST_BOMB_PRESSURE_POTENTIAL
    assert potential_shaping.potential_difference(
        old_state,
        post_bomb_state,
    ) == pytest.approx(4.0)


def test_terminal_potential_is_zero(monkeypatch):
    monkeypatch.setattr(
        potential_shaping.callbacks,
        "safe_opponent_bombing_path",
        lambda _state: (None, 0, True),
    )
    state = attack_state()
    assert potential_shaping.attack_readiness_potential(None) == 0.0
    assert potential_shaping.potential_difference(
        state,
        None,
    ) == pytest.approx(
        -potential_shaping.POTENTIAL_SCALE
        * potential_shaping.READY_TO_BOMB_POTENTIAL
    )


def test_discounted_potential_rewards_telescope_without_free_loop(
    monkeypatch,
):
    gamma = 0.9
    scale = 5.0
    states = [
        {"potential": 0.04},
        {"potential": 0.07},
        {"potential": 0.10},
        {"potential": 0.04},
    ]
    monkeypatch.setattr(
        potential_shaping,
        "attack_readiness_potential",
        lambda state: 0.0 if state is None else state["potential"],
    )

    shaping_rewards = [
        potential_shaping.potential_difference(
            old_state,
            new_state,
            gamma=gamma,
            scale=scale,
        )
        for old_state, new_state in zip(states, states[1:])
    ]
    discounted_total = sum(
        gamma**index * reward
        for index, reward in enumerate(shaping_rewards)
    )
    expected = scale * (
        -states[0]["potential"]
        + gamma ** (len(states) - 1) * states[-1]["potential"]
    )
    assert discounted_total == pytest.approx(expected)
    assert discounted_total < 0.0


def test_v7_replaces_immediate_target_bonus_with_state_potential(
    monkeypatch,
):
    monkeypatch.setattr(
        dqn_train,
        "base_reward_from_events",
        lambda _self, _events: 4.0,
    )
    monkeypatch.setattr(
        dqn_train,
        "endgame_safe_attack_active",
        lambda state: bool(state and state.get("attack_mode")),
    )
    monkeypatch.setattr(
        dqn_train,
        "potential_difference",
        lambda *_args, **_kwargs: 4.0,
    )
    reward = dqn_train.transition_reward(
        object(),
        {"attack_mode": True},
        {"post_bomb": True},
        [dqn_train.BOMB_TARGETED_OPPONENT],
    )
    assert reward == 4.0


def test_unrelated_mode_keeps_base_reward_unchanged(monkeypatch):
    monkeypatch.setattr(
        dqn_train,
        "base_reward_from_events",
        lambda _self, _events: 7.0,
    )
    monkeypatch.setattr(
        dqn_train,
        "endgame_safe_attack_active",
        lambda _state: False,
    )
    monkeypatch.setattr(
        dqn_train,
        "potential_difference",
        lambda *_args, **_kwargs: 0.0,
    )
    assert dqn_train.transition_reward(
        object(),
        {"coin_mode": True},
        {"coin_mode": True},
        [dqn_train.BOMB_TARGETED_OPPONENT],
    ) == 7.0


def _checkpoint_pair():
    policy = DQN(52, 6)
    target = DQN(52, 6)
    return {
        "policy_state": deepcopy(policy.state_dict()),
        "target_state": deepcopy(target.state_dict()),
    }


def test_checkpoint_verifier_accepts_only_adapter_change():
    baseline = _checkpoint_pair()
    candidate = deepcopy(baseline)
    candidate["policy_state"]["network.0.weight"][0, 45] += 1.0
    result = verify_adapter_only_update(baseline, candidate)
    assert result == {
        "features_0_44_unchanged": True,
        "later_layers_unchanged": True,
        "adapter_45_51_changed": True,
    }


class Logger:
    def info(self, *_args, **_kwargs):
        pass


def test_target_sync_changes_only_adapter_columns():
    torch.manual_seed(21)
    policy = DQN(52, 6)
    target = DQN(52, 6)
    target.load_state_dict(policy.state_dict())
    agent = SimpleNamespace(
        policy_net=policy,
        target_net=target,
        pending_optimizer_state=None,
        logger=Logger(),
        episodes_trained=0,
        epsilon=0.05,
        endgame_epsilon=0.05,
        optimizer_steps=0,
        device=torch.device("cpu"),
    )
    dqn_train.setup_training(agent)
    agent.replay_warmup = 1
    agent.batch_size = 1
    agent.target_sync_interval = 1

    target_protected = target.network[0].weight[:, :45].detach().clone()
    target_later = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if name != "network.0.weight"
    }
    state = np.zeros(52, dtype=np.float32)
    state[45] = 1.0
    agent.replay_buffer.add(
        state,
        0,
        5.0,
        np.zeros(52, dtype=np.float32),
        np.zeros(6, dtype=bool),
        True,
    )

    assert dqn_train.optimize_model(agent) is not None
    torch.testing.assert_close(
        target.network[0].weight[:, :45],
        target_protected,
        rtol=0,
        atol=0,
    )
    for name, value in target_later.items():
        torch.testing.assert_close(
            target.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        target.network[0].weight[:, 45:],
        policy.network[0].weight[:, 45:],
    )


@pytest.mark.parametrize(
    ("parameter_name", "index"),
    [
        ("network.0.weight", (0, 44)),
        ("network.2.weight", (0, 0)),
    ],
)
def test_checkpoint_verifier_rejects_protected_change(
    parameter_name,
    index,
):
    baseline = _checkpoint_pair()
    candidate = deepcopy(baseline)
    candidate["policy_state"][parameter_name][index] += 1.0
    with pytest.raises(RuntimeError, match="Protected"):
        verify_adapter_only_update(baseline, candidate)


def test_potential_parameters_are_valid():
    assert math.isclose(potential_shaping.DEFAULT_GAMMA, dqn_train.GAMMA)
    assert potential_shaping.POTENTIAL_SCALE > 0.0
    assert (
        0.0
        < potential_shaping.READY_TO_BOMB_POTENTIAL
        < potential_shaping.POST_BOMB_PRESSURE_POTENTIAL
        <= 1.0
    )
