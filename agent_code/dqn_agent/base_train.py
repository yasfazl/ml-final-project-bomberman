# Local one-step crate-efficiency fine-tuning (version 2).
from typing import List
import pickle

import numpy as np

import events as e
from .base_callbacks import (
    ACTIONS,
    MODEL_PATH,
    nearest_coin_path,
    state_to_features,
    valid_action_indices,
)
from .game_utils import (
    best_crate_bombing_path,
    bomb_target_counts,
    earliest_danger_times,
    nearest_crate_bombing_path,
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
FINE_TUNE_EPSILON_START = 0.15

# Features 0-31 belong to the already trained, stable Task 2 agent.
# Fine-tuning updates only the seven additive crate-efficiency features.
BASE_FEATURE_DIM = 32


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

REWARD_COIN_COLLECTED = 15.0
REWARD_COIN_FOUND = 0.5
REWARD_MOVED_TOWARD_COIN = 1.0
REWARD_MOVED_AWAY_FROM_COIN = -1.0

REWARD_CRATE_DESTROYED = 2.0
REWARD_MOVED_TOWARD_CRATE = 0.75
REWARD_MOVED_AWAY_FROM_CRATE = -0.75

# A successfully placed bomb receives this reward once per targeted crate.
# The slightly stronger immediate signal makes multi-crate positions valuable
# without waiting several steps for the explosion event.
REWARD_BOMB_TARGETED_CRATE = 1.25
REWARD_BOMB_TARGETED_OPPONENT = 4.0
REWARD_USELESS_BOMB = -5.0
REWARD_BOMB_WHILE_COIN_VISIBLE = -4.0

# New crate-efficiency shaping.
REWARD_MOVED_TOWARD_BETTER_BOMB_SPOT = 0.5
REWARD_MOVED_AWAY_FROM_BETTER_BOMB_SPOT = -0.5
REWARD_BOMBED_BEFORE_BETTER_BOMB_SPOT = -1.0

REWARD_MOVED_TOWARD_SAFETY = 1.5
REWARD_REACHED_SAFETY = 4.0
REWARD_MOVED_AWAY_FROM_SAFETY = -2.5
REWARD_WAITED_IN_DANGER = -3.0
REWARD_MOVED_INTO_TRAP = -8.0

REWARD_KILLED_OPPONENT = 20.0
REWARD_KILLED_SELF = -30.0
REWARD_GOT_KILLED = -20.0
REWARD_SURVIVED_ROUND = 0.0

REWARD_WAITED = -0.2
REWARD_WAITED_WITH_GOAL = -2.0
REWARD_INVALID_ACTION = -2.0
STEP_PENALTY = -0.05


# ---------------------------------------------------------------------------
# Custom events
# ---------------------------------------------------------------------------

MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

MOVED_TOWARD_CRATE = "MOVED_TOWARD_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"

MOVED_TOWARD_BETTER_BOMB_SPOT = (
    "MOVED_TOWARD_BETTER_BOMB_SPOT"
)
MOVED_AWAY_FROM_BETTER_BOMB_SPOT = (
    "MOVED_AWAY_FROM_BETTER_BOMB_SPOT"
)
BOMBED_BEFORE_BETTER_BOMB_SPOT = (
    "BOMBED_BEFORE_BETTER_BOMB_SPOT"
)

BOMB_TARGETED_CRATE = "BOMB_TARGETED_CRATE"
BOMB_TARGETED_OPPONENT = "BOMB_TARGETED_OPPONENT"
USELESS_BOMB = "USELESS_BOMB"
BOMB_WHILE_COIN_VISIBLE = "BOMB_WHILE_COIN_VISIBLE"
WAITED_WITH_GOAL = "WAITED_WITH_GOAL"

MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
REACHED_SAFETY = "REACHED_SAFETY"
MOVED_AWAY_FROM_SAFETY = "MOVED_AWAY_FROM_SAFETY"
WAITED_IN_DANGER = "WAITED_IN_DANGER"
MOVED_INTO_TRAP = "MOVED_INTO_TRAP"


def setup_training(self):
    """Initialize training-only statistics for safe fine-tuning."""
    self.round_reward = 0.0
    self.round_td_errors = []
    self.training_steps = 0

    # Protect the already-good coin, bomb, and escape behaviour. The model
    # still uses every feature for Q-value calculation, but gradient updates
    # are restricted to the new crate-efficiency columns.
    self.freeze_base_features = True

    if self.episodes_trained == 0:
        self.epsilon = EPSILON_START
    else:
        # A mature stable model is usually already at epsilon=0.05. Give the
        # seven new weights enough exploration to observe better bomb spots.
        self.epsilon = max(
            self.epsilon,
            FINE_TUNE_EPSILON_START,
        )

    self.logger.info(
        f"TD Q-learning crate-efficiency fine-tuning initialized: "
        f"episodes={self.episodes_trained}, "
        f"epsilon={self.epsilon:.3f}, "
        f"frozen_features=0:{BASE_FEATURE_DIM}"
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
    add_crate_navigation_event(old_game_state, new_game_state, events)
    add_crate_efficiency_navigation_event(
        old_game_state,
        new_game_state,
        events,
    )
    add_bomb_placement_event(old_game_state, self_action, events)
    add_waiting_event(old_game_state, self_action, events)
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
    """Apply one semi-gradient linear Q-learning update.

    During crate-efficiency fine-tuning, Q-values still use the complete
    model, while only columns 32 onward receive gradient updates. This keeps
    the trained Task 2 behaviour bit-for-bit unchanged in columns 0-31.

    Tests and optional fresh training can omit ``freeze_base_features`` to
    retain the original full-model update behaviour.
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

    if getattr(self, "freeze_base_features", False):
        if self.model.shape[1] <= BASE_FEATURE_DIM:
            raise ValueError(
                "Crate-efficiency fine-tuning requires features "
                f"after index {BASE_FEATURE_DIM - 1}, but model shape is "
                f"{self.model.shape}."
            )

        self.model[
            action_index,
            BASE_FEATURE_DIM:,
        ] += (
            ALPHA
            * td_error
            * features[BASE_FEATURE_DIM:]
        )
    else:
        self.model[action_index] += (
            ALPHA * td_error * features
        )

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
    """Describe the targets and efficiency of a placed bomb."""
    if old_game_state is None or action != "BOMB":
        return

    if e.BOMB_DROPPED not in events:
        return

    crate_count, opponent_count = bomb_target_counts(old_game_state)

    # Penalize taking a one-crate bomb when a strictly better safe position
    # is reachable nearby. Do not penalize a bomb that already targets an
    # opponent, since that has a separate high-value purpose.
    (
        _best_direction,
        best_distance,
        best_crate_count,
    ) = best_crate_bombing_path(old_game_state)

    better_crate_position_exists = (
        best_distance not in (None, 0)
        and best_crate_count > crate_count
    )

    if better_crate_position_exists and opponent_count == 0:
        events.append(BOMBED_BEFORE_BETTER_BOMB_SPOT)

    # Add one event per targeted crate, so a two-crate bomb receives twice
    # the immediate placement credit of a one-crate bomb.
    for _ in range(crate_count):
        events.append(BOMB_TARGETED_CRATE)

    if opponent_count > 0:
        events.append(BOMB_TARGETED_OPPONENT)

    if crate_count == 0 and opponent_count == 0:
        events.append(USELESS_BOMB)

    if old_game_state.get("coins", []):
        events.append(BOMB_WHILE_COIN_VISIBLE)


def add_crate_navigation_event(
    old_game_state: dict,
    new_game_state: dict,
    events: List[str],
):
    """Reward progress toward any reachable safe crate-bombing tile."""
    if old_game_state is None or new_game_state is None:
        return

    if old_game_state.get("coins", []):
        return

    if old_game_state.get("bombs", []) or new_game_state.get("bombs", []):
        return

    if not np.array_equal(
        old_game_state["field"],
        new_game_state["field"],
    ):
        return

    old_distance = nearest_crate_bombing_path(old_game_state)[1]
    new_distance = nearest_crate_bombing_path(new_game_state)[1]

    if old_distance is None or new_distance is None:
        return

    if new_distance < old_distance:
        events.append(MOVED_TOWARD_CRATE)
    elif new_distance > old_distance:
        events.append(MOVED_AWAY_FROM_CRATE)


def add_crate_efficiency_navigation_event(
    old_game_state: dict,
    new_game_state: dict,
    events: List[str],
):
    """Reward movement toward a higher-yield safe bomb position.

    The event is active only when the old position has a strictly better
    candidate nearby. Crate layouts and active bombs must remain unchanged,
    so this shaping never interferes with coin collection or bomb escape.
    """
    if old_game_state is None or new_game_state is None:
        return

    if (
        old_game_state.get("coins", [])
        or new_game_state.get("coins", [])
    ):
        return

    if (
        old_game_state.get("bombs", [])
        or new_game_state.get("bombs", [])
    ):
        return

    if not np.array_equal(
        old_game_state["field"],
        new_game_state["field"],
    ):
        return

    (
        _old_direction,
        old_distance,
        old_best_crate_count,
    ) = best_crate_bombing_path(old_game_state)

    old_current_crate_count, _old_opponent_count = bomb_target_counts(
        old_game_state
    )

    old_has_better_position = (
        old_distance not in (None, 0)
        and old_best_crate_count > old_current_crate_count
    )

    if not old_has_better_position:
        return

    (
        _new_direction,
        new_distance,
        new_best_crate_count,
    ) = best_crate_bombing_path(new_game_state)

    if new_distance is None:
        events.append(MOVED_AWAY_FROM_BETTER_BOMB_SPOT)
        return

    if new_best_crate_count > old_best_crate_count:
        events.append(MOVED_TOWARD_BETTER_BOMB_SPOT)
    elif new_best_crate_count < old_best_crate_count:
        events.append(MOVED_AWAY_FROM_BETTER_BOMB_SPOT)
    elif new_distance < old_distance:
        events.append(MOVED_TOWARD_BETTER_BOMB_SPOT)
    elif new_distance > old_distance:
        events.append(MOVED_AWAY_FROM_BETTER_BOMB_SPOT)


def _current_position_is_dangerous(game_state: dict) -> bool:
    if game_state is None:
        return False

    position = tuple(game_state["self"][3])
    danger_times = earliest_danger_times(game_state)
    return not np.isinf(danger_times[position])


def add_waiting_event(
    old_game_state: dict,
    action: str,
    events: List[str],
):
    """Penalize avoidable waiting while a reachable goal exists."""
    if old_game_state is None or action != "WAIT":
        return

    if old_game_state.get("bombs", []):
        return

    coin_distance = nearest_coin_path(old_game_state)[1]
    crate_distance = nearest_crate_bombing_path(old_game_state)[1]

    has_coin_goal = coin_distance is not None
    has_crate_goal = crate_distance is not None

    if has_coin_goal or has_crate_goal:
        events.append(WAITED_WITH_GOAL)


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
        WAITED_WITH_GOAL: REWARD_WAITED_WITH_GOAL,
        e.INVALID_ACTION: REWARD_INVALID_ACTION,
        MOVED_TOWARD_COIN: REWARD_MOVED_TOWARD_COIN,
        MOVED_AWAY_FROM_COIN: REWARD_MOVED_AWAY_FROM_COIN,
        MOVED_TOWARD_CRATE: REWARD_MOVED_TOWARD_CRATE,
        MOVED_AWAY_FROM_CRATE: REWARD_MOVED_AWAY_FROM_CRATE,
        MOVED_TOWARD_BETTER_BOMB_SPOT:
            REWARD_MOVED_TOWARD_BETTER_BOMB_SPOT,
        MOVED_AWAY_FROM_BETTER_BOMB_SPOT:
            REWARD_MOVED_AWAY_FROM_BETTER_BOMB_SPOT,
        BOMBED_BEFORE_BETTER_BOMB_SPOT:
            REWARD_BOMBED_BEFORE_BETTER_BOMB_SPOT,
        BOMB_TARGETED_CRATE: REWARD_BOMB_TARGETED_CRATE,
        BOMB_TARGETED_OPPONENT: REWARD_BOMB_TARGETED_OPPONENT,
        USELESS_BOMB: REWARD_USELESS_BOMB,
        BOMB_WHILE_COIN_VISIBLE: REWARD_BOMB_WHILE_COIN_VISIBLE,
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
        "feature_dim": int(self.model.shape[1]),
        "training_stage": "local_crate_efficiency_fine_tuning_v2",
    }

    temporary_path = MODEL_PATH.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(saved_data, file)

    temporary_path.replace(MODEL_PATH)
