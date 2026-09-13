"""Opt-in, policy-neutral diagnostics for Bomberman deaths.

The game engine owns the ground-truth explosion owner, so diagnostics live
outside the agent callbacks. Nothing in this module changes an action, reward,
feature, network, checkpoint, or random-number generator.
"""

from __future__ import annotations

from collections import defaultdict, deque
import json
import os
from pathlib import Path
from typing import Any

from agent_code.q_learning_agent.game_utils import (
    action_has_survival_route,
    bomb_has_robust_escape_route,
    has_escape_route_after_bomb,
)


DIAGNOSTIC_ENV = "BOMBERMAN_SUICIDE_DIAGNOSTICS"
DIAGNOSTIC_PATH_ENV = "BOMBERMAN_SUICIDE_DIAGNOSTICS_PATH"
DEFAULT_DIAGNOSTIC_PATH = Path("diagnostics/dqn_deaths.jsonl")
HISTORY_LIMIT = 12
TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _position(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _inside(position: list[int] | None, tiles: set[tuple[int, int]]) -> bool:
    return position is not None and tuple(position) in tiles


def classify_death(
    victim_name: str,
    death_step: int,
    death_position: tuple[int, int],
    causes: list[dict],
    history: list[dict],
) -> dict:
    """Return a deterministic classification from ground-truth causes."""
    own_causes = [
        cause for cause in causes
        if cause.get("killer") == victim_name
    ]
    opponent_causes = [
        cause for cause in causes
        if cause.get("killer") != victim_name
    ]
    result = {
        "self_kill": bool(own_causes),
        "opponent_kill": bool(opponent_causes) and not own_causes,
        "overlapping_lethal_blasts": len(causes) > 1,
        "category": "unknown",
        "own_bomb_placed_step": None,
        "own_bomb_age": None,
        "left_own_blast": None,
        "reentered_own_blast": None,
        "unsafe_post_bomb_action_seen": None,
        "later_opponent_threat_seen": None,
        "placement_safety": None,
    }

    if not own_causes:
        if opponent_causes:
            result["category"] = "opponent_blast"
        return result

    own_cause = min(
        own_causes,
        key=lambda cause: (
            cause.get("placed_step") is None,
            cause.get("placed_step") or death_step,
        ),
    )
    placed_step = own_cause.get("placed_step")
    result["own_bomb_placed_step"] = placed_step
    if placed_step is not None:
        result["own_bomb_age"] = int(death_step - placed_step)

    relevant_history = [
        entry for entry in history
        if placed_step is None or entry.get("step", -1) >= placed_step
    ]
    blast_tiles = {
        tuple(position)
        for position in own_cause.get("blast_tiles", [])
    }
    trajectory = []
    if relevant_history:
        trajectory.append(relevant_history[0].get("previous_position"))
        trajectory.extend(
            entry.get("resulting_position")
            for entry in relevant_history
        )
    trajectory.append(_position(death_position))

    left_blast = any(
        position is not None and not _inside(position, blast_tiles)
        for position in trajectory
    )
    seen_outside = False
    reentered = False
    for position in trajectory:
        if position is None:
            continue
        if not _inside(position, blast_tiles):
            seen_outside = True
        elif seen_outside:
            reentered = True

    placement_entry = next(
        (
            entry for entry in relevant_history
            if entry.get("step") == placed_step
            and entry.get("outcome_event") == "BOMB_DROPPED"
        ),
        None,
    )
    placement_safety = (
        placement_entry.get("safety")
        if placement_entry is not None
        else None
    )
    unsafe_post_bomb_action = any(
        entry.get("safety", {}).get("selected_action_survivable") is False
        for entry in relevant_history
        if entry.get("safety") is not None
    )
    later_opponent_threat = any(
        entry.get("opponent_bombs_present", False)
        or entry.get("opponent_adjacent", False)
        for entry in relevant_history
        if placed_step is None or entry.get("step", -1) > placed_step
    )

    result.update(
        {
            "left_own_blast": left_blast,
            "reentered_own_blast": reentered,
            "unsafe_post_bomb_action_seen": unsafe_post_bomb_action,
            "later_opponent_threat_seen": later_opponent_threat,
            "placement_safety": placement_safety,
        }
    )

    robust_at_placement = (
        placement_safety.get("robust_bomb_escape")
        if placement_safety is not None
        else None
    )
    if own_causes and opponent_causes:
        result["category"] = "overlapping_own_and_opponent_blasts"
    elif robust_at_placement is False:
        result["category"] = "unsafe_own_bomb_placement"
    elif unsafe_post_bomb_action:
        result["category"] = "unsafe_post_bomb_action"
    elif reentered:
        result["category"] = "reentered_own_blast"
    elif not left_blast:
        result["category"] = "failed_to_clear_own_blast"
    elif later_opponent_threat:
        result["category"] = "own_blast_after_opponent_interference"
    else:
        result["category"] = "own_blast_after_escape_route_loss"
    return result


class DeathDiagnostics:
    """Collect a short DQN trajectory and write one JSON record per death."""

    def __init__(
        self,
        *,
        enabled: bool,
        path: Path,
        scenario: str,
        seed: int | None,
    ):
        self.enabled = bool(enabled)
        self.path = Path(path)
        self.scenario = scenario
        self.seed = seed
        self.histories = defaultdict(
            lambda: deque(maxlen=HISTORY_LIMIT)
        )
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "record_type": "run_start",
                            "scenario": scenario,
                            "seed": seed,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        except OSError:
            self.enabled = False

    @classmethod
    def from_args(cls, args):
        requested = os.environ.get(DIAGNOSTIC_ENV, "").lower()
        path = Path(
            os.environ.get(
                DIAGNOSTIC_PATH_ENV,
                str(DEFAULT_DIAGNOSTIC_PATH),
            )
        )
        return cls(
            enabled=requested in TRUTHY_VALUES,
            path=path,
            scenario=args.scenario,
            seed=args.seed,
        )

    def start_round(self, _round_number: int) -> None:
        if self.enabled:
            self.histories.clear()

    def _safety_snapshot(
        self,
        agent,
        requested_action: str,
        world_bombs: list,
    ) -> dict | None:
        state = agent.last_game_state
        owns_active_bomb = any(
            bomb.owner is agent
            for bomb in world_bombs
        )
        if state is None or (
            requested_action != "BOMB" and not owns_active_bomb
        ):
            return None
        try:
            snapshot = {
                "selected_action_survivable": bool(
                    action_has_survival_route(state, requested_action)
                )
            }
            if requested_action == "BOMB":
                snapshot.update(
                    {
                        "robust_bomb_escape": bool(
                            bomb_has_robust_escape_route(state)
                        ),
                        "legacy_bomb_escape": bool(
                            has_escape_route_after_bomb(state)
                        ),
                    }
                )
            return snapshot
        except (KeyError, TypeError, ValueError, IndexError):
            return {"analysis_error": True}

    def record_action(
        self,
        *,
        world,
        agent,
        requested_action: str,
        outcome_event: str | None,
        previous_position: tuple[int, int],
    ) -> None:
        if not self.enabled or agent.code_name != "dqn_agent":
            return

        safety = self._safety_snapshot(
            agent,
            requested_action,
            list(world.bombs),
        )
        opponents = [
            (other.x, other.y)
            for other in world.active_agents
            if other is not agent and not other.dead
        ]
        resulting_position = (agent.x, agent.y)
        opponent_adjacent = any(
            abs(resulting_position[0] - position[0])
            + abs(resulting_position[1] - position[1])
            == 1
            for position in opponents
        )
        opponent_bombs_present = any(
            bomb.owner is not agent
            for bomb in world.bombs
        )
        self.histories[agent.name].append(
            {
                "step": int(world.step),
                "requested_action": requested_action,
                "outcome_event": outcome_event,
                "previous_position": _position(previous_position),
                "resulting_position": _position(resulting_position),
                "opponents": [_position(position) for position in opponents],
                "opponent_adjacent": opponent_adjacent,
                "opponent_bombs_present": opponent_bombs_present,
                "safety": safety,
            }
        )

    def record_death(self, *, world, victim, explosions: list) -> None:
        if not self.enabled or victim.code_name != "dqn_agent":
            return
        causes = [
            {
                "killer": explosion.owner.name,
                "bomb_position": _position(
                    getattr(
                        explosion,
                        "diagnostic_bomb_position",
                        None,
                    )
                ),
                "placed_round": getattr(
                    explosion,
                    "diagnostic_bomb_placed_round",
                    None,
                ),
                "placed_step": getattr(
                    explosion,
                    "diagnostic_bomb_placed_step",
                    None,
                ),
                "blast_tiles": [
                    _position(position)
                    for position in explosion.blast_coords
                ],
            }
            for explosion in explosions
        ]
        history = list(self.histories.get(victim.name, []))
        classification = classify_death(
            victim.name,
            int(world.step),
            (victim.x, victim.y),
            causes,
            history,
        )
        record = {
            "record_type": "death",
            "scenario": self.scenario,
            "seed": self.seed,
            "round": int(world.round),
            "step": int(world.step),
            "victim": victim.name,
            "death_position": _position((victim.x, victim.y)),
            "last_action": victim.last_action,
            "causes": causes,
            "classification": classification,
            "history": history,
        }
        try:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            self.enabled = False
