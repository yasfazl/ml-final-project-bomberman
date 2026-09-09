"""Optional ground-truth diagnostics for invalid Bomberman actions.

The game asks every agent to act from the same snapshot and then executes the
chosen actions in a randomized order.  This module compares that snapshot with
the world at execution time so a movement collision can be distinguished from
an action that was already illegal when it was selected.

Nothing is recorded unless ``BOMBERMAN_INVALID_DIAGNOSTICS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
DIAGNOSTICS_FLAG = "BOMBERMAN_INVALID_DIAGNOSTICS"
DIAGNOSTICS_PATH = "BOMBERMAN_INVALID_DIAGNOSTICS_PATH"
DIAGNOSTICS_AGENT = "BOMBERMAN_INVALID_DIAGNOSTICS_AGENT"


def _object_position(item) -> tuple[int, int]:
    if hasattr(item, "x") and hasattr(item, "y"):
        return int(item.x), int(item.y)
    return tuple(int(value) for value in item[0])


def _snapshot_mode(game_state: dict | None) -> str:
    if not game_state:
        return "unknown"
    explosion_map = np.asarray(game_state.get("explosion_map", []))
    if game_state.get("bombs", []) or (
        explosion_map.size and np.any(explosion_map > 0)
    ):
        return "danger"
    if game_state.get("coins", []):
        return "coin"
    if np.any(np.asarray(game_state["field"]) == 1):
        return "crate"
    if game_state.get("others", []):
        return "endgame"
    return "finished"


def diagnostics_enabled_for(agent) -> bool:
    """Return whether diagnostics are enabled for this particular agent."""
    if os.environ.get(DIAGNOSTICS_FLAG) != "1":
        return False
    selected = os.environ.get(DIAGNOSTICS_AGENT, "dqn_agent")
    return selected == "*" or selected in {
        getattr(agent, "name", None),
        getattr(agent, "code_name", None),
    }


def classify_invalid_action(world, agent, action: str) -> dict:
    """Describe why an action was invalid at its execution time."""
    snapshot = getattr(agent, "last_game_state", None)
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    snapshot_self = snapshot.get("self")
    if snapshot_self is not None:
        start = tuple(int(value) for value in snapshot_self[3])
        snapshot_bomb_available = bool(snapshot_self[2])
    else:
        start = (int(agent.x), int(agent.y))
        snapshot_bomb_available = bool(agent.bombs_left)

    record = {
        "round": int(getattr(world, "round", -1)),
        "step": int(getattr(world, "step", -1)),
        "seed": getattr(getattr(world, "args", None), "seed", None),
        "scenario": getattr(
            getattr(world, "args", None),
            "scenario",
            None,
        ),
        "agent_name": getattr(agent, "name", None),
        "agent_code": getattr(agent, "code_name", None),
        "action": str(action),
        "mode": _snapshot_mode(snapshot),
        "position": list(start),
        "snapshot_bomb_available": snapshot_bomb_available,
        "execution_bomb_available": bool(agent.bombs_left),
    }

    if action in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[action]
        destination = (start[0] + dx, start[1] + dy)
        record["destination"] = list(destination)

        snapshot_field = np.asarray(snapshot.get("field", world.arena))
        in_bounds = (
            0 <= destination[0] < snapshot_field.shape[0]
            and 0 <= destination[1] < snapshot_field.shape[1]
        )
        snapshot_field_free = bool(
            in_bounds and snapshot_field[destination] == 0
        )
        snapshot_bombs = {
            tuple(int(value) for value in bomb[0])
            for bomb in snapshot.get("bombs", [])
        }
        snapshot_opponents = {
            tuple(int(value) for value in opponent[3])
            for opponent in snapshot.get("others", [])
        }
        current_bombs = {
            _object_position(bomb)
            for bomb in getattr(world, "bombs", [])
        }
        current_occupants = [
            other
            for other in getattr(world, "active_agents", [])
            if other is not agent
            and (int(other.x), int(other.y)) == destination
        ]

        valid_at_snapshot = (
            snapshot_field_free
            and destination not in snapshot_bombs
            and destination not in snapshot_opponents
        )
        record.update(
            {
                "valid_at_snapshot": bool(valid_at_snapshot),
                "snapshot_field_free": snapshot_field_free,
                "snapshot_bomb_occupied": destination in snapshot_bombs,
                "snapshot_opponent_occupied": (
                    destination in snapshot_opponents
                ),
                "execution_bomb_occupied": destination in current_bombs,
                "execution_occupants": [
                    getattr(other, "name", None)
                    for other in current_occupants
                ],
            }
        )

        if not in_bounds or not snapshot_field_free:
            category = "static_field_block"
        elif destination in snapshot_bombs:
            category = "bomb_already_occupied"
        elif destination in snapshot_opponents:
            category = "opponent_already_occupied"
        elif current_occupants:
            category = "simultaneous_agent_collision"
        elif destination in current_bombs:
            category = "new_bomb_occupancy"
        else:
            category = "unexplained_movement_invalid"
    elif action == "BOMB":
        if not snapshot_bomb_available:
            category = "bomb_unavailable_at_snapshot"
        elif not bool(agent.bombs_left):
            category = "bomb_unavailable_at_execution"
        else:
            category = "unexplained_bomb_invalid"
    else:
        category = "unknown_action"

    record["category"] = category
    return record


def record_invalid_action(world, agent, action: str) -> None:
    """Append one JSONL record when diagnostics are explicitly enabled."""
    if not diagnostics_enabled_for(agent):
        return
    output_path = Path(
        os.environ.get(
            DIAGNOSTICS_PATH,
            "diagnostics/invalid_actions.jsonl",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = classify_invalid_action(world, agent, action)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")
