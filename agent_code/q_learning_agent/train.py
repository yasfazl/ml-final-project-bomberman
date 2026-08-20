from typing import List
import pickle

import numpy as np

import events as e
from .callbacks import (
    ACTIONS,
    MODEL_PATH,
    nearest_coin_path,
    state_to_features,
    valid_action_indices,
)


# ---------------------------------------------------------------------------
# Q-learning hyperparameters
# ---------------------------------------------------------------------------

ALPHA = 0.01
GAMMA = 0.90

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
PURE_EXPLORATION_EPISODES = 100


# ---------------------------------------------------------------------------
# Task 1 reward configuration
# ---------------------------------------------------------------------------

REWARD_COIN_COLLECTED = 10.0
REWARD_MOVED_TOWARD_COIN = 0.5
REWARD_MOVED_AWAY_FROM_COIN = -0.5
REWARD_WAITED = -0.2
REWARD_INVALID_ACTION = -1.0

# A small cost encourages faster coin collection and makes
# repeated toward/away cycles unprofitable.
STEP_PENALTY = -0.1


# Custom events
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"


def setup_training(self):
    """
    Initialize training-only statistics.
    """
    self.round_reward = 0.0
    self.round_td_errors = []
    self.training_steps = 0

    if self.episodes_trained == 0:
        self.epsilon = EPSILON_START

    self.logger.info(
        f"TD Q-learning initialized: "
        f"episodes={self.episodes_trained}, "
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
    Called after each non-terminal action.

    Performs one TD Q-learning update immediately.
    """
    add_coin_distance_event(
        old_game_state,
        new_game_state,
        events,
    )

    reward = reward_from_events(self, events)

    td_error = update_q_learning(
        self=self,
        old_game_state=old_game_state,
        action=self_action,
        new_game_state=new_game_state,
        reward=reward,
    )

    self.round_reward += reward
    self.training_steps += 1

    if td_error is not None:
        self.round_td_errors.append(abs(td_error))

    self.logger.debug(
        f"Action={self_action}, "
        f"events={events}, "
        f"reward={reward:.3f}, "
        f"TD-error={td_error}"
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """
    Handle the terminal transition, decay epsilon, and save the model.
    """
    reward = reward_from_events(self, events)

    # new_game_state=None marks a terminal transition.
    td_error = update_q_learning(
        self=self,
        old_game_state=last_game_state,
        action=last_action,
        new_game_state=None,
        reward=reward,
    )

    self.round_reward += reward

    if td_error is not None:
        self.round_td_errors.append(abs(td_error))

    self.episodes_trained += 1

    # Keep full exploration for the first episodes.
    if self.episodes_trained > PURE_EXPLORATION_EPISODES:
        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY,
        )

    save_model(self)

    if self.round_td_errors:
        mean_absolute_td_error = float(
            np.mean(self.round_td_errors)
        )
    else:
        mean_absolute_td_error = 0.0

    self.logger.info(
        f"Episode {self.episodes_trained}: "
        f"reward={self.round_reward:.2f}, "
        f"epsilon={self.epsilon:.3f}, "
        f"mean_abs_td_error={mean_absolute_td_error:.3f}"
    )

    self.round_reward = 0.0
    self.round_td_errors = []


def update_q_learning(
    self,
    old_game_state: dict,
    action: str,
    new_game_state: dict | None,
    reward: float,
) -> float | None:
    """
    Apply one semi-gradient linear Q-learning update.

    Q(s, a) = w_a^T phi(s)

    TD target:
        terminal:      r
        non-terminal:  r + gamma * max_a' Q(s', a')
    """
    if old_game_state is None:
        return None

    if action not in ACTIONS:
        self.logger.warning(f"Unknown action: {action}")
        return None

    features = state_to_features(old_game_state)

    if features is None:
        return None

    action_index = ACTIONS.index(action)

    # Current prediction Q(s, a)
    current_q_value = float(
        self.model[action_index] @ features
    )

    if new_game_state is None:
        # No future value after a terminal state.
        target = reward
    else:
        next_features = state_to_features(new_game_state)
        next_valid_indices = valid_action_indices(new_game_state)

        next_q_values = self.model @ next_features

        best_next_q_value = float(
            np.max(next_q_values[next_valid_indices])
        )

        target = (
            reward
            + GAMMA * best_next_q_value
        )

    td_error = target - current_q_value

    # Semi-gradient linear Q-learning update
    self.model[action_index] += (
        ALPHA * td_error * features
    )

    return float(td_error)


def add_coin_distance_event(
    old_game_state: dict,
    new_game_state: dict,
    events: List[str],
):
    """
    Add a custom event based on the BFS path distance to the
    nearest visible coin.

    Distance shaping is skipped when a coin was collected or
    when the set of visible coins changed.
    """
    if old_game_state is None or new_game_state is None:
        return

    if e.COIN_COLLECTED in events:
        return

    old_coins = set(old_game_state.get("coins", []))
    new_coins = set(new_game_state.get("coins", []))

    if not old_coins or old_coins != new_coins:
        return

    old_distance = nearest_coin_path(old_game_state)[1]
    new_distance = nearest_coin_path(new_game_state)[1]

    if old_distance is None or new_distance is None:
        return

    if new_distance < old_distance:
        events.append(MOVED_TOWARD_COIN)
    elif new_distance > old_distance:
        events.append(MOVED_AWAY_FROM_COIN)


def reward_from_events(
    self,
    events: List[str],
) -> float:
    """
    Convert game events into a scalar Task 1 reward.
    """
    reward_map = {
        e.COIN_COLLECTED: REWARD_COIN_COLLECTED,
        e.WAITED: REWARD_WAITED,
        e.INVALID_ACTION: REWARD_INVALID_ACTION,
        MOVED_TOWARD_COIN: REWARD_MOVED_TOWARD_COIN,
        MOVED_AWAY_FROM_COIN: REWARD_MOVED_AWAY_FROM_COIN,
    }

    reward = STEP_PENALTY

    for event in events:
        reward += reward_map.get(event, 0.0)

    self.logger.debug(
        f"Reward={reward:.3f} from events={events}"
    )

    return float(reward)


def save_model(self):
    """
    Save Q-learning weights and training state.

    A temporary file is used to avoid leaving a partially
    written model if saving is interrupted.
    """
    saved_data = {
        "weights": self.model,
        "epsilon": self.epsilon,
        "episodes_trained": self.episodes_trained,
    }

    temporary_path = MODEL_PATH.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(saved_data, file)

    temporary_path.replace(MODEL_PATH)