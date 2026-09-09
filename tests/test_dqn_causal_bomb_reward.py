from types import SimpleNamespace

import numpy as np
import pytest

import events as e
from agent_code.dqn_agent import train as dqn_train
from agent_code.dqn_agent.replay_buffer import ReplayBuffer


class Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def state(step: int) -> dict:
    return {"round": 1, "step": step}


def causal_agent() -> SimpleNamespace:
    return SimpleNamespace(
        replay_buffer=ReplayBuffer(20, seed=1),
        environment_steps=0,
        pending_bomb_transition=None,
        bomb_outcome_stats={
            "resolved_bombs": 0,
            "opponents_killed": 0,
            "self_kills": 0,
            "misses": 0,
            "terminal_unresolved": 0,
        },
        logger=Logger(),
    )


@pytest.fixture(autouse=True)
def simple_transition_encoding(monkeypatch):
    monkeypatch.setattr(
        dqn_train,
        "_features_or_zeros",
        lambda game_state: np.full(
            dqn_train.FEATURE_DIM,
            0 if game_state is None else game_state.get("step", 0),
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        dqn_train,
        "candidate_action_mask",
        lambda _agent, _state: np.ones(
            len(dqn_train.ACTIONS),
            dtype=bool,
        ),
    )


def transitions(agent):
    return list(agent.replay_buffer._transitions)


def stage_bomb(agent, reward=-0.05, placed_step=10):
    dqn_train._stage_bomb_transition(
        agent,
        state(placed_step),
        "BOMB",
        reward,
        state(placed_step + 1),
    )


def test_successful_bomb_is_staged_and_flat_target_bonus_is_removed(
    monkeypatch,
):
    agent = causal_agent()
    monkeypatch.setattr(
        dqn_train,
        "reward_from_events",
        lambda _agent, _events: 3.95,
    )

    reward_delta = dqn_train._record_causal_transition(
        agent,
        state(10),
        "BOMB",
        state(11),
        [e.BOMB_DROPPED, dqn_train.BOMB_TARGETED_OPPONENT],
    )

    assert len(agent.replay_buffer) == 0
    assert agent.pending_bomb_transition is not None
    assert agent.pending_bomb_transition.reward == pytest.approx(-0.05)
    assert reward_delta == pytest.approx(-0.05)


def test_kill_reward_moves_to_originating_bomb_without_double_counting(
    monkeypatch,
):
    agent = causal_agent()
    stage_bomb(agent)
    monkeypatch.setattr(
        dqn_train,
        "reward_from_events",
        lambda _agent, _events: 19.95,
    )

    reward_delta = dqn_train._record_causal_transition(
        agent,
        state(14),
        "WAIT",
        state(15),
        [e.BOMB_EXPLODED, e.KILLED_OPPONENT],
    )

    stored = transitions(agent)
    redistributed = (dqn_train.GAMMA ** 4) * 20.0
    assert len(stored) == 2
    assert stored[0].action == dqn_train.ACTIONS.index("BOMB")
    assert stored[0].reward == pytest.approx(-0.05 + redistributed)
    assert stored[1].action == dqn_train.ACTIONS.index("WAIT")
    assert stored[1].reward == pytest.approx(-0.05)
    assert reward_delta == pytest.approx(-0.05 + redistributed)
    assert agent.pending_bomb_transition is None
    assert agent.bomb_outcome_stats == {
        "resolved_bombs": 1,
        "opponents_killed": 1,
        "self_kills": 0,
        "misses": 0,
        "terminal_unresolved": 0,
    }


def test_missed_bomb_gets_no_invented_outcome_reward(monkeypatch):
    agent = causal_agent()
    stage_bomb(agent, reward=-0.05)
    monkeypatch.setattr(
        dqn_train,
        "reward_from_events",
        lambda _agent, _events: -0.05,
    )

    dqn_train._record_causal_transition(
        agent,
        state(14),
        "UP",
        state(15),
        [e.BOMB_EXPLODED],
    )

    stored = transitions(agent)
    assert stored[0].reward == pytest.approx(-0.05)
    assert stored[1].reward == pytest.approx(-0.05)
    assert agent.bomb_outcome_stats["misses"] == 1


def test_self_kill_penalty_moves_to_bomb_but_death_penalty_stays(
    monkeypatch,
):
    agent = causal_agent()
    stage_bomb(agent)
    monkeypatch.setattr(
        dqn_train,
        "reward_from_events",
        lambda _agent, _events: -50.05,
    )

    reward_delta = dqn_train._record_causal_transition(
        agent,
        state(14),
        "LEFT",
        None,
        [e.BOMB_EXPLODED, e.KILLED_SELF, e.GOT_KILLED],
    )

    stored = transitions(agent)
    redistributed = (dqn_train.GAMMA ** 4) * -30.0
    assert stored[0].reward == pytest.approx(-0.05 + redistributed)
    assert stored[1].reward == pytest.approx(-20.05)
    assert reward_delta == pytest.approx(-20.05 + redistributed)
    assert agent.bomb_outcome_stats["self_kills"] == 1


def test_actual_early_explosion_delay_controls_discount(monkeypatch):
    agent = causal_agent()
    stage_bomb(agent, placed_step=10)
    monkeypatch.setattr(
        dqn_train,
        "reward_from_events",
        lambda _agent, _events: 19.95,
    )

    dqn_train._record_causal_transition(
        agent,
        state(12),
        "RIGHT",
        state(13),
        [e.BOMB_EXPLODED, e.KILLED_OPPONENT],
    )

    expected = -0.05 + (dqn_train.GAMMA ** 2) * 20.0
    assert transitions(agent)[0].reward == pytest.approx(expected)


def test_unresolved_bomb_is_flushed_at_terminal_without_fake_outcome():
    agent = causal_agent()
    stage_bomb(agent, reward=1.25)

    dqn_train._flush_pending_bomb(agent)

    assert len(agent.replay_buffer) == 1
    assert transitions(agent)[0].reward == pytest.approx(1.25)
    assert agent.pending_bomb_transition is None
    assert agent.bomb_outcome_stats["terminal_unresolved"] == 1


def test_non_explosion_transition_does_not_resolve_pending_bomb(
    monkeypatch,
):
    agent = causal_agent()
    stage_bomb(agent)
    monkeypatch.setattr(
        dqn_train,
        "reward_from_events",
        lambda _agent, _events: 0.45,
    )

    dqn_train._record_causal_transition(
        agent,
        state(11),
        "UP",
        state(12),
        [e.MOVED_UP],
    )

    assert agent.pending_bomb_transition is not None
    assert len(agent.replay_buffer) == 1
    assert transitions(agent)[0].action == dqn_train.ACTIONS.index("UP")
    assert transitions(agent)[0].reward == pytest.approx(0.45)
