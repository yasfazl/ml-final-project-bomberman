"""Validated training-only configuration for the V6 reward search.

The defaults reproduce the V3 reward function exactly.  Environment
variables are intentionally used only by experimental training processes, so
loading a checkpoint for evaluation never depends on this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from os import environ as process_environ
from typing import Mapping


PROGRESS_REWARD_ENV = "BOMBERMAN_DQN_SAFE_ATTACK_PROGRESS_REWARD"
WAIT_PENALTY_ENV = "BOMBERMAN_DQN_SAFE_ATTACK_WAIT_PENALTY"
TARGETED_BOMB_REWARD_ENV = (
    "BOMBERMAN_DQN_TARGETED_OPPONENT_BOMB_REWARD"
)
FRESH_OPTIMIZER_ENV = "BOMBERMAN_DQN_FRESH_OPTIMIZER"
REPLAY_SEED_ENV = "BOMBERMAN_DQN_REPLAY_SEED"

DEFAULT_PROGRESS_REWARD = 0.5
DEFAULT_WAIT_PENALTY = 1.0
DEFAULT_TARGETED_BOMB_REWARD = 4.0


@dataclass(frozen=True)
class RewardSearchConfig:
    """Reward values and reproducibility controls for one training run."""

    progress_reward: float = DEFAULT_PROGRESS_REWARD
    wait_penalty: float = DEFAULT_WAIT_PENALTY
    targeted_bomb_reward: float = DEFAULT_TARGETED_BOMB_REWARD
    fresh_optimizer: bool = False
    replay_seed: int | None = None

    def checkpoint_metadata(self) -> dict:
        return asdict(self)


def _read_bounded_float(
    source: Mapping[str, str],
    name: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    raw = source.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number, found {raw!r}.") from error
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be in [{lower}, {upper}], found {value}."
        )
    return value


def _read_flag(
    source: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    raw = source.get(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean flag, found {raw!r}."
    )


def _read_optional_seed(
    source: Mapping[str, str],
    name: str,
) -> int | None:
    raw = source.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, found {raw!r}.") from error
    if value < 0:
        raise ValueError(f"{name} must be non-negative, found {value}.")
    return value


def read_reward_search_config(
    source: Mapping[str, str] | None = None,
) -> RewardSearchConfig:
    """Read a bounded configuration, using V3 values when unset."""
    if source is None:
        source = process_environ
    return RewardSearchConfig(
        progress_reward=_read_bounded_float(
            source,
            PROGRESS_REWARD_ENV,
            DEFAULT_PROGRESS_REWARD,
            0.1,
            1.0,
        ),
        wait_penalty=_read_bounded_float(
            source,
            WAIT_PENALTY_ENV,
            DEFAULT_WAIT_PENALTY,
            0.25,
            2.0,
        ),
        targeted_bomb_reward=_read_bounded_float(
            source,
            TARGETED_BOMB_REWARD_ENV,
            DEFAULT_TARGETED_BOMB_REWARD,
            0.5,
            4.0,
        ),
        fresh_optimizer=_read_flag(
            source,
            FRESH_OPTIMIZER_ENV,
            False,
        ),
        replay_seed=_read_optional_seed(source, REPLAY_SEED_ENV),
    )


REWARD_SEARCH_CONFIG = read_reward_search_config()
