import json
from types import SimpleNamespace

import numpy as np

from invalid_action_diagnostics import (
    classify_invalid_action,
    record_invalid_action,
)


def open_field(size=7):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_agent(name, position):
    return SimpleNamespace(
        name=name,
        code_name=name,
        x=position[0],
        y=position[1],
        bombs_left=True,
        last_game_state=None,
    )


def make_world():
    agent = make_agent("dqn_agent", (2, 2))
    opponent = make_agent("rule_based_agent_0", (4, 2))
    field = open_field()
    agent.last_game_state = {
        "round": 1,
        "step": 10,
        "field": field.copy(),
        "self": ("dqn_agent", 0, True, (2, 2)),
        "others": [
            ("rule_based_agent_0", 0, True, (4, 2)),
        ],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }
    world = SimpleNamespace(
        round=1,
        step=10,
        arena=field,
        bombs=[],
        active_agents=[agent, opponent],
        args=SimpleNamespace(seed=42, scenario="empty"),
    )
    return world, agent, opponent


def test_detects_simultaneous_agent_collision():
    world, agent, opponent = make_world()
    opponent.x, opponent.y = (3, 2)

    record = classify_invalid_action(world, agent, "RIGHT")

    assert record["category"] == "simultaneous_agent_collision"
    assert record["valid_at_snapshot"]
    assert record["destination"] == [3, 2]
    assert record["execution_occupants"] == ["rule_based_agent_0"]
    assert record["mode"] == "endgame"


def test_distinguishes_position_occupied_in_original_snapshot():
    world, agent, opponent = make_world()
    opponent.x, opponent.y = (3, 2)
    agent.last_game_state["others"][0] = (
        "rule_based_agent_0",
        0,
        True,
        (3, 2),
    )

    record = classify_invalid_action(world, agent, "RIGHT")

    assert record["category"] == "opponent_already_occupied"
    assert not record["valid_at_snapshot"]


def test_distinguishes_static_field_block():
    world, agent, _opponent = make_world()
    agent.last_game_state["field"][3, 2] = 1
    world.arena[3, 2] = 1

    record = classify_invalid_action(world, agent, "RIGHT")

    assert record["category"] == "static_field_block"
    assert not record["valid_at_snapshot"]


def test_distinguishes_unavailable_bomb():
    world, agent, _opponent = make_world()
    agent.last_game_state["self"] = (
        "dqn_agent",
        0,
        False,
        (2, 2),
    )
    agent.bombs_left = False

    record = classify_invalid_action(world, agent, "BOMB")

    assert record["category"] == "bomb_unavailable_at_snapshot"


def test_jsonl_recording_is_explicitly_gated(tmp_path, monkeypatch):
    world, agent, opponent = make_world()
    opponent.x, opponent.y = (3, 2)
    output = tmp_path / "invalid.jsonl"
    monkeypatch.setenv(
        "BOMBERMAN_INVALID_DIAGNOSTICS_PATH",
        str(output),
    )

    record_invalid_action(world, agent, "RIGHT")
    assert not output.exists()

    monkeypatch.setenv("BOMBERMAN_INVALID_DIAGNOSTICS", "1")
    monkeypatch.setenv(
        "BOMBERMAN_INVALID_DIAGNOSTICS_AGENT",
        "dqn_agent",
    )
    record_invalid_action(world, agent, "RIGHT")

    saved = json.loads(output.read_text())
    assert saved["category"] == "simultaneous_agent_collision"
    assert saved["seed"] == 42
