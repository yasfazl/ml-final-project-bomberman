"""Replay memory with defensive copies of mutable game features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self._transitions = deque(maxlen=int(capacity))
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._transitions)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self._transitions.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                next_action_mask=np.asarray(
                    next_action_mask,
                    dtype=bool,
                ).copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from {len(self)}."
            )
        return self._random.sample(list(self._transitions), batch_size)

