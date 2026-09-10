import json

import pytest

from agent_code.dqn_agent import train as dqn_train
from agent_code.dqn_agent.reward_search_config import (
    FRESH_OPTIMIZER_ENV,
    PROGRESS_REWARD_ENV,
    REPLAY_SEED_ENV,
    RewardSearchConfig,
    TARGETED_BOMB_REWARD_ENV,
    WAIT_PENALTY_ENV,
    read_reward_search_config,
)
from tools.optimize_dqn_rewards import (
    _ensure_manifest,
    aggregate_runs,
    constraint_values,
    constraints_are_feasible,
)


def test_reward_search_defaults_reproduce_v3():
    config = read_reward_search_config({})
    assert config.progress_reward == 0.5
    assert config.wait_penalty == 1.0
    assert config.targeted_bomb_reward == 4.0
    assert config.fresh_optimizer is False
    assert config.replay_seed is None


def test_reward_search_reads_controlled_overrides():
    config = read_reward_search_config(
        {
            PROGRESS_REWARD_ENV: "0.35",
            WAIT_PENALTY_ENV: "1.25",
            TARGETED_BOMB_REWARD_ENV: "2.5",
            FRESH_OPTIMIZER_ENV: "true",
            REPLAY_SEED_ENV: "60013",
        }
    )
    assert config.progress_reward == 0.35
    assert config.wait_penalty == 1.25
    assert config.targeted_bomb_reward == 2.5
    assert config.fresh_optimizer is True
    assert config.replay_seed == 60013


def test_reward_search_rejects_out_of_range_value():
    with pytest.raises(ValueError, match=PROGRESS_REWARD_ENV):
        read_reward_search_config({PROGRESS_REWARD_ENV: "2.0"})


def test_default_reward_search_values_leave_v3_reward_unchanged(monkeypatch):
    monkeypatch.setattr(
        dqn_train,
        "REWARD_SEARCH_CONFIG",
        RewardSearchConfig(),
    )
    monkeypatch.setattr(
        dqn_train,
        "base_reward_from_events",
        lambda _self, _events: 4.0,
    )
    events = [
        dqn_train.BOMB_TARGETED_OPPONENT,
        dqn_train.MOVED_TOWARD_SAFE_ATTACK_POSITION,
    ]
    assert dqn_train.reward_from_events(object(), events) == 4.5


def test_reward_search_values_adjust_only_the_selected_terms(monkeypatch):
    monkeypatch.setattr(
        dqn_train,
        "REWARD_SEARCH_CONFIG",
        RewardSearchConfig(
            progress_reward=0.25,
            wait_penalty=1.5,
            targeted_bomb_reward=2.0,
        ),
    )
    monkeypatch.setattr(
        dqn_train,
        "base_reward_from_events",
        lambda _self, _events: 4.0,
    )
    events = [
        dqn_train.BOMB_TARGETED_OPPONENT,
        dqn_train.MOVED_TOWARD_SAFE_ATTACK_POSITION,
        dqn_train.WAITED_DURING_SAFE_ATTACK,
    ]
    assert dqn_train.reward_from_events(object(), events) == 0.75


def test_aggregate_runs_normalizes_to_one_hundred_rounds():
    runs = [
        {
            "score": 10,
            "kills": 2,
            "suicides": 1,
            "bombs": 20,
            "invalid": 4,
            "steps": 100,
        },
        {
            "score": 20,
            "kills": 4,
            "suicides": 0,
            "bombs": 30,
            "invalid": 6,
            "steps": 150,
        },
    ]
    metrics = aggregate_runs(runs, rounds_per_run=10)
    assert metrics["score_per_100_rounds"] == 150.0
    assert metrics["suicides_per_100_rounds"] == 5.0
    assert metrics["kills_per_100_bombs"] == 12.0
    assert metrics["invalid_per_1000_steps"] == 40.0
    assert metrics["total_rounds"] == 20


def test_candidate_is_feasible_only_when_all_guards_pass():
    baseline = {
        "score_per_100_rounds": 90.0,
        "suicides_per_100_rounds": 5.0,
        "kills_per_100_bombs": 1.30,
    }
    candidate = {
        "score_per_100_rounds": 92.0,
        "suicides_per_100_rounds": 5.4,
        "kills_per_100_bombs": 1.25,
    }
    values = constraint_values(
        candidate,
        baseline,
        suicide_tolerance=0.5,
        efficiency_retention=0.95,
    )
    assert constraints_are_feasible(values)


def test_candidate_with_lower_score_is_not_feasible():
    baseline = {
        "score_per_100_rounds": 90.0,
        "suicides_per_100_rounds": 5.0,
        "kills_per_100_bombs": 1.30,
    }
    candidate = {
        "score_per_100_rounds": 89.9,
        "suicides_per_100_rounds": 4.0,
        "kills_per_100_bombs": 2.0,
    }
    values = constraint_values(
        candidate,
        baseline,
        suicide_tolerance=0.5,
        efficiency_retention=0.95,
    )
    assert not constraints_are_feasible(values)
    assert values["score_retention"] > 0


def test_manifest_prevents_mixing_incompatible_studies(tmp_path):
    path = tmp_path / "manifest.json"
    expected = {"seeds": [1, 2, 3], "rounds": 30}
    _ensure_manifest(path, expected)
    assert json.loads(path.read_text()) == expected
    _ensure_manifest(path, expected)
    with pytest.raises(RuntimeError, match="differs"):
        _ensure_manifest(path, {"seeds": [4, 5, 6], "rounds": 30})
