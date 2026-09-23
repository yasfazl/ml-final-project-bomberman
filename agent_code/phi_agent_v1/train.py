from collections import namedtuple, deque
from typing import List

import csv
import pickle
from pathlib import Path
import numpy as np

import events as e
from .callbacks import (
    ACTIONS,
    state_to_features,
    MODEL_PATH,
    nearest_coin_path,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ALPHA = 0.01          # learning rate
GAMMA = 0.9           # discount factor

EPSILON_START = 1.0   # initial exploration rate
EPSILON_MIN = 0.1     # minimum exploration rate
EPSILON_DECAY = 0.999 # multiplicative decay per episode
EXPLORE_EPISODES = 100  # pure exploration before decay starts

# ---------------------------------------------------------------------------
# Reward values
# ---------------------------------------------------------------------------

REWARD_COIN_COLLECTED       = 10.0
REWARD_CRATE_DESTROYED      = 0.5
REWARD_KILLED_SELF          = -10.0
REWARD_MOVED_TOWARD_COIN    = 2.0
REWARD_MOVED_AWAY_FROM_COIN = -2.0
REWARD_WAITED               = -1.0
REWARD_INVALID_ACTION       = -1.0

# Custom event names
MOVED_TOWARD_COIN   = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

STATS_PATH = Path(__file__).resolve().parent / "stats.csv"

# ---------------------------------------------------------------------------
# Transition tuple
# ---------------------------------------------------------------------------

Transition = namedtuple("Transition", ("state", "action", "reward"))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def setup_training(self):
    """
    Called once after setup() in callbacks.py.
    Initialises training-only state.
    """
    self.episode_transitions: deque[Transition] = deque()
    self.coins_collected = 0

    # CSV 헤더 작성 (파일 없을 때만)
    if not STATS_PATH.is_file():
        with STATS_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "steps", "coins_collected", "epsilon"])

    self.logger.info(
        f"Training setup complete. "
        f"episodes_trained={self.episodes_trained}, "
        f"epsilon={self.epsilon:.3f}"
    )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step (except the last).
    Appends one transition to the episode buffer.
    """
    # -----------------------------------------------------------------------
    # Custom events: moved toward or away from nearest coin
    # -----------------------------------------------------------------------
    if old_game_state is not None and new_game_state is not None:
        _, old_dist = nearest_coin_path(old_game_state)
        _, new_dist = nearest_coin_path(new_game_state)

        if old_dist is not None and new_dist is not None:
            if new_dist < old_dist:
                events.append(MOVED_TOWARD_COIN)
            elif new_dist > old_dist:
                events.append(MOVED_AWAY_FROM_COIN)

    # -----------------------------------------------------------------------
    # Compute reward and store transition
    # -----------------------------------------------------------------------
    reward = reward_from_events(self, events)
    features = state_to_features(old_game_state)

    if features is not None and self_action is not None:
        self.episode_transitions.append(
            Transition(features, self_action, reward)
        )

    # 코인 수집 카운트
    if e.COIN_COLLECTED in events:
        self.coins_collected += 1


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """
    Called once at the end of each episode.
    Performs the Monte Carlo weight update and saves the model.
    """
    # Store the final transition.
    reward = reward_from_events(self, events)
    features = state_to_features(last_game_state)

    if features is not None and last_action is not None:
        self.episode_transitions.append(
            Transition(features, last_action, reward)
        )

    # -----------------------------------------------------------------------
    # Monte Carlo return calculation and weight update
    # -----------------------------------------------------------------------
    transitions = list(self.episode_transitions)
    G = 0.0

    for transition in reversed(transitions):
        G = transition.reward + GAMMA * G

        action_index = ACTIONS.index(transition.action)
        phi = transition.state

        # Linear Q-value for the taken action.
        q_current = self.model[action_index] @ phi

        # Gradient descent step:
        #   w <- w + alpha * (G - Q(s,a)) * phi
        self.model[action_index] += ALPHA * (G - q_current) * phi

    self.logger.info(
        f"Episode {self.episodes_trained + 1} complete. "
        f"Transitions: {len(transitions)}, "
        f"epsilon: {self.epsilon:.4f}"
    )

    # -----------------------------------------------------------------------
    # Epsilon decay (starts after EXPLORE_EPISODES)
    # -----------------------------------------------------------------------
    self.episodes_trained += 1

    if self.episodes_trained > EXPLORE_EPISODES:
        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY,
        )

    # -----------------------------------------------------------------------
    # Clear episode buffer
    # -----------------------------------------------------------------------
    self.episode_transitions.clear()

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------
    saved_data = {
        "weights": self.model,
        "epsilon": self.epsilon,
        "episodes_trained": self.episodes_trained,
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(saved_data, f)

    self.logger.info(f"Model saved to {MODEL_PATH}.")

    # -----------------------------------------------------------------------
    # CSV 기록
    # -----------------------------------------------------------------------
    with STATS_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            self.episodes_trained,
            last_game_state["step"],
            self.coins_collected,
            round(self.epsilon, 4),
        ])

    self.coins_collected = 0


def reward_from_events(self, events: List[str]) -> float:
    """
    Maps game events to scalar rewards.
    """
    reward_map = {
        e.COIN_COLLECTED:     REWARD_COIN_COLLECTED,
        e.CRATE_DESTROYED:    REWARD_CRATE_DESTROYED,
        e.KILLED_SELF:        REWARD_KILLED_SELF,
        e.WAITED:             REWARD_WAITED,
        e.INVALID_ACTION:     REWARD_INVALID_ACTION,
        MOVED_TOWARD_COIN:    REWARD_MOVED_TOWARD_COIN,
        MOVED_AWAY_FROM_COIN: REWARD_MOVED_AWAY_FROM_COIN,
    }

    total = sum(reward_map.get(event, 0.0) for event in events)

    self.logger.debug(
        f"Reward {total:.3f} from events: {', '.join(events)}"
    )

    return total