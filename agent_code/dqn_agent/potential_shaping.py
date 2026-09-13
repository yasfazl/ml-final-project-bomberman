"""Bounded potential-based shaping for the safe-attack endgame adapter.

The shaping reward follows the policy-invariant form

    F(s, s') = beta * (gamma * Phi(s') - Phi(s)).

Only endgame attack-readiness states have non-zero potential.  Coin, crate,
ordinary pursuit, and unrelated danger states therefore retain the V3 reward.
"""

from __future__ import annotations

import numpy as np

from . import callbacks
from . import game_utils


DEFAULT_GAMMA = 0.90
POTENTIAL_SCALE = 5.0
READY_TO_BOMB_POTENTIAL = 0.10
POST_BOMB_PRESSURE_POTENTIAL = 1.0


def _endgame_objectives_complete(game_state: dict | None) -> bool:
    """Return whether only opponent combat remains."""
    if game_state is None:
        return False
    if game_state.get("coins", []):
        return False
    if np.any(np.asarray(game_state["field"]) == 1):
        return False
    return bool(game_state.get("others", []))


def _safe_post_bomb_pressure(game_state: dict) -> bool:
    """Detect the immediate state after a useful, escapable own bomb.

    Bomb ownership is not exposed by the framework.  Immediately after our
    BOMB action, however, the new bomb is on our current tile; another agent's
    bomb cannot occupy that tile.  This state-only property identifies the
    transition without consulting the selected action.
    """
    position = tuple(game_state["self"][3])
    bomb_timer = next(
        (
            int(timer)
            for bomb_position, timer in game_state.get("bombs", [])
            if tuple(bomb_position) == position
        ),
        None,
    )
    if bomb_timer is None or bomb_timer <= 0:
        return False

    explosion_map = game_state.get("explosion_map")
    if (
        explosion_map is not None
        and np.asarray(explosion_map)[position] > 0
    ):
        return False

    _crate_count, opponent_count = game_utils.bomb_target_counts(
        game_state,
        bomb_position=position,
    )
    if opponent_count == 0:
        return False

    _direction, escape_distance = game_utils.nearest_safe_path(game_state)
    return escape_distance is not None


def attack_readiness_potential(game_state: dict | None) -> float:
    """Return a bounded state potential in ``[0, 1]``.

    Values below ``0.1`` encode progress toward a robust attack tile, ``0.1``
    means a robust opponent-targeting bomb is ready, and ``1.0`` represents
    the immediate safe-pressure state after that bomb is placed.
    """
    if not _endgame_objectives_complete(game_state):
        return 0.0

    if _safe_post_bomb_pressure(game_state):
        return POST_BOMB_PRESSURE_POTENTIAL

    _direction, distance, safe_bomb_now = (
        callbacks.safe_opponent_bombing_path(game_state)
    )
    if safe_bomb_now:
        return READY_TO_BOMB_POTENTIAL
    if distance in (None, 0):
        return 0.0

    field = np.asarray(game_state["field"])
    maximum_distance = max(field.shape[0] + field.shape[1], 1)
    normalized_progress = 1.0 - min(
        float(distance) / maximum_distance,
        1.0,
    )
    return float(READY_TO_BOMB_POTENTIAL * normalized_progress)


def potential_difference(
    old_game_state: dict | None,
    new_game_state: dict | None,
    *,
    gamma: float = DEFAULT_GAMMA,
    scale: float = POTENTIAL_SCALE,
) -> float:
    """Compute ``beta * (gamma * Phi(s') - Phi(s))``."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if scale < 0.0:
        raise ValueError("scale must be non-negative")

    old_potential = attack_readiness_potential(old_game_state)
    new_potential = attack_readiness_potential(new_game_state)
    return float(scale * (gamma * new_potential - old_potential))
