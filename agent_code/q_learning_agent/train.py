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
from .game_utils import (
    bomb_target_counts,
    earliest_danger_times,
    nearest_safe_path,
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
# Reward configuration
# ---------------------------------------------------------------------------

REWARD_COIN_COLLECTED = 10.0
REWARD_COIN_FOUND = 0.5
REWARD_MOVED_TOWARD_COIN = 0.5
REWARD_MOVED_AWAY_FROM_COIN = -0.5

REWARD_CRATE_DESTROYED = 2.0
REWARD_BOMB_TARGETED_CRATE = 1.0
REWARD_BOMB_TARGETED_OPPONENT = 4.0
REWARD_USELESS_BOMB = -5.0

REWARD_MOVED_TOWARD_SAFETY = 1.5
REWARD_REACHED_SAFETY = 4.0
REWARD_MOVED_AWAY_FROM_SAFETY = -2.5
REWARD_WAITED_IN_DANGER = -3.0
REWARD_MOVED_INTO_TRAP = -8.0

REWARD_KILLED_OPPONENT = 20.0
REWARD_KILLED_SELF = -30.0
REWARD_GOT_KILLED = -20.0
REWARD_SURVIVED_ROUND = 5.0

REWARD_WAITED = -0.2
REWARD_INVALID_ACTION = -2.0
STEP_PENALTY = -0.05


# Custom events
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

BOMB_TARGETED_CRATE = "BOMB_TARGETED_CRATE"
BOMB_TARGETED_OPPONENT = "BOMB_TARGETED_OPPONENT"
USELESS_BOMB = "USELESS_BOMB"

MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
REACHED_SAFETY = "REACHED_SAFETY"
MOVED_AWAY_FROM_SAFETY = "MOVED_AWAY_FROM_SAFETY"
WAITED_IN_DANGER = "WAITED_IN_DANGER"
MOVED_INTO_TRAP = "MOVED_INTO_TRAP"


def setup_training(self):
    """Initialize training-only statistics."""
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
    """Add shaped events and perform one TD Q-learning update."""
    add_coin_distance_event(old_game_state, new_game_state, events)
    add_bomb_placement_event(old_game_state, self_action, events)
    add_escape_events(
        old_game_state,
        self_action,
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
        f"Action={self_action}, events={events}, "
        f"reward={reward:.3f}, TD-error={td_error}"
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """Handle the terminal update, epsilon decay, and model saving."""
    add_bomb_placement_event(last_game_state, last_action, events)

    reward = reward_from_events(self, events)

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
    """Apply one semi-gradient linear Q-learning update."""
    if old_game_state is None:
        return None

    if action not in ACTIONS:
        self.logger.warning(f"Unknown action: {action}")
        return None

    features = state_to_features(old_game_state)

    if features is None:
        return None

    action_index = ACTIONS.index(action)
    current_q_value = float(self.model[action_index] @ features)

    if new_game_state is None:
        target = reward
    else:
        next_features = state_to_features(new_game_state)
        next_valid_indices = valid_action_indices(new_game_state)
        next_q_values = self.model @ next_features

        best_next_q_value = float(
            np.max(next_q_values[next_valid_indices])
        )
        target = reward + GAMMA * best_next_q_value

    td_error = target - current_q_value

    self.model[action_index] += ALPHA * td_error * features

    return float(td_error)


def add_coin_distance_event(
    old_game_state: dict,
    new_game_state: dict,
    events: List[str],
):
    """Add an event for progress toward the nearest visible coin."""
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


def add_bomb_placement_event(
    old_game_state: dict,
    action: str,
    events: List[str],
):
    """Describe the targets of a successfully placed bomb."""
    if old_game_state is None or action != "BOMB":
        return

    if e.BOMB_DROPPED not in events:
        return

    crate_count, opponent_count = bomb_target_counts(old_game_state)

    if crate_count > 0:
        events.append(BOMB_TARGETED_CRATE)

    if opponent_count > 0:
        events.append(BOMB_TARGETED_OPPONENT)

    if crate_count == 0 and opponent_count == 0:
        events.append(USELESS_BOMB)


def _current_position_is_dangerous(game_state: dict) -> bool:
    if game_state is None:
        return False

    position = tuple(game_state["self"][3])
    danger_times = earliest_danger_times(game_state)
    return not np.isinf(danger_times[position])


def add_escape_events(
    old_game_state: dict,
    action: str,
    new_game_state: dict,
    events: List[str],
):
    """Reward progress along a time-aware escape route."""
    if old_game_state is None or new_game_state is None:
        return

    if not _current_position_is_dangerous(old_game_state):
        return

    _old_direction, old_distance = nearest_safe_path(old_game_state)

    if not _current_position_is_dangerous(new_game_state):
        events.append(REACHED_SAFETY)
        return

    _new_direction, new_distance = nearest_safe_path(new_game_state)

    if action == "WAIT":
        events.append(WAITED_IN_DANGER)

    if new_distance is None:
        events.append(MOVED_INTO_TRAP)
        return

    if old_distance is None:
        return

    if new_distance < old_distance:
        events.append(MOVED_TOWARD_SAFETY)
    elif new_distance >= old_distance:
        events.append(MOVED_AWAY_FROM_SAFETY)


def reward_from_events(self, events: List[str]) -> float:
    """Convert game and custom events into a scalar reward."""
    reward_map = {
        e.COIN_COLLECTED: REWARD_COIN_COLLECTED,
        e.COIN_FOUND: REWARD_COIN_FOUND,
        e.CRATE_DESTROYED: REWARD_CRATE_DESTROYED,
        e.KILLED_OPPONENT: REWARD_KILLED_OPPONENT,
        e.KILLED_SELF: REWARD_KILLED_SELF,
        e.GOT_KILLED: REWARD_GOT_KILLED,
        e.SURVIVED_ROUND: REWARD_SURVIVED_ROUND,
        e.WAITED: REWARD_WAITED,
        e.INVALID_ACTION: REWARD_INVALID_ACTION,
        MOVED_TOWARD_COIN: REWARD_MOVED_TOWARD_COIN,
        MOVED_AWAY_FROM_COIN: REWARD_MOVED_AWAY_FROM_COIN,
        BOMB_TARGETED_CRATE: REWARD_BOMB_TARGETED_CRATE,
        BOMB_TARGETED_OPPONENT: REWARD_BOMB_TARGETED_OPPONENT,
        USELESS_BOMB: REWARD_USELESS_BOMB,
        MOVED_TOWARD_SAFETY: REWARD_MOVED_TOWARD_SAFETY,
        REACHED_SAFETY: REWARD_REACHED_SAFETY,
        MOVED_AWAY_FROM_SAFETY: REWARD_MOVED_AWAY_FROM_SAFETY,
        WAITED_IN_DANGER: REWARD_WAITED_IN_DANGER,
        MOVED_INTO_TRAP: REWARD_MOVED_INTO_TRAP,
    }

    reward = STEP_PENALTY

    for event in events:
        reward += reward_map.get(event, 0.0)

    self.logger.debug(
        f"Reward={reward:.3f} from events={events}"
    )

    return float(reward)


def save_model(self):
    """Atomically save Q-learning weights and training state."""
    saved_data = {
        "weights": self.model,
        "epsilon": self.epsilon,
        "episodes_trained": self.episodes_trained,
    }

    temporary_path = MODEL_PATH.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(saved_data, file)

    temporary_path.replace(MODEL_PATH)