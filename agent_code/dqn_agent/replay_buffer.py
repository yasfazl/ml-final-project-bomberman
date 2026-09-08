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
    n_steps: int = 1


def _copied_transition(
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    next_action_mask: np.ndarray,
    done: bool,
    n_steps: int = 1,
) -> Transition:
    """Build a transition without retaining mutable caller-owned arrays."""
    if n_steps <= 0:
        raise ValueError("Transition horizon must be positive.")
    return Transition(
        state=np.asarray(state, dtype=np.float32).copy(),
        action=int(action),
        reward=float(reward),
        next_state=np.asarray(next_state, dtype=np.float32).copy(),
        next_action_mask=np.asarray(next_action_mask, dtype=bool).copy(),
        done=bool(done),
        n_steps=int(n_steps),
    )


class NStepAccumulator:
    """Convert consecutive one-step experiences into n-step transitions."""

    def __init__(self, n_steps: int, gamma: float):
        if n_steps <= 0:
            raise ValueError("N-step horizon must be positive.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Gamma must lie in [0, 1].")
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self._pending: deque[Transition] = deque()

    def __len__(self) -> int:
        return len(self._pending)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> list[Transition]:
        """Append one environment step and return each transition now ready."""
        self._pending.append(
            _copied_transition(
                state,
                action,
                reward,
                next_state,
                next_action_mask,
                done,
            )
        )

        ready = []
        if done:
            while self._pending:
                ready.append(self._aggregate_oldest())
                self._pending.popleft()
            return ready

        while len(self._pending) >= self.n_steps:
            ready.append(self._aggregate_oldest())
            self._pending.popleft()
        return ready

    def _aggregate_oldest(self) -> Transition:
        sequence = []
        for transition in self._pending:
            sequence.append(transition)
            if len(sequence) == self.n_steps or transition.done:
                break

        first = sequence[0]
        last = sequence[-1]
        discounted_reward = sum(
            (self.gamma ** offset) * transition.reward
            for offset, transition in enumerate(sequence)
        )
        return _copied_transition(
            first.state,
            first.action,
            discounted_reward,
            last.next_state,
            last.next_action_mask,
            last.done,
            n_steps=len(sequence),
        )


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
        n_steps: int = 1,
    ) -> None:
        self._transitions.append(
            _copied_transition(
                state,
                action,
                reward,
                next_state,
                next_action_mask,
                done,
                n_steps,
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
