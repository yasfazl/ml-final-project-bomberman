from collections import deque
from pathlib import Path
import pickle
import random

import numpy as np
import settings as s

from .game_utils import (
    bomb_target_counts,
    earliest_danger_times,
    has_escape_route_after_bomb,
    nearest_safe_path,
)


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

MOVE_ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

DIRECTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

# Feature vector:
# 0: bias
# 1-4: whether UP, RIGHT, DOWN, LEFT are valid
# 5: whether at least one visible coin exists
# 6-9: first direction of the shortest path to a coin
# 10: normalized distance to the nearest reachable coin
# 11: danger urgency at the current position
# 12-16: danger urgency after UP, RIGHT, DOWN, LEFT, WAIT
# 17: whether the agent currently has a bomb available
# 18: whether placing a bomb leaves a timed escape route
# 19: normalized number of crates hit by a bomb placed here
# 20: whether an opponent would be hit by a bomb placed here
# 21-24: first direction of the shortest safe escape path
# 25: normalized distance along that escape path
FEATURE_DIM = 26

MODEL_PATH = Path(__file__).resolve().parent / "q_model.pkl"


def setup(self):
    """
    Initialize or load the Q-learning model.

    Smaller compatible models are migrated to 26 features by preserving
    their old weights and initializing new weights to zero.
    """
    self.model_path = MODEL_PATH

    self.model = np.zeros(
        (len(ACTIONS), FEATURE_DIM),
        dtype=np.float64,
    )
    self.epsilon = 1.0
    self.episodes_trained = 0

    if not self.model_path.is_file():
        self.logger.info("No saved model found. Starting from scratch.")
        return

    try:
        with self.model_path.open("rb") as file:
            saved_data = pickle.load(file)

        saved_weights = np.asarray(
            saved_data["weights"],
            dtype=np.float64,
        )

        if saved_weights.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional model, found "
                f"shape {saved_weights.shape}."
            )

        if saved_weights.shape[0] != len(ACTIONS):
            raise ValueError(
                f"Expected {len(ACTIONS)} action rows, found "
                f"shape {saved_weights.shape}."
            )

        if saved_weights.shape[1] > FEATURE_DIM:
            raise ValueError(
                f"Saved model has {saved_weights.shape[1]} features, "
                f"but this agent expects {FEATURE_DIM}."
            )

        if saved_weights.shape[1] < FEATURE_DIM:
            old_feature_dimension = saved_weights.shape[1]
            migrated_weights = np.zeros(
                (len(ACTIONS), FEATURE_DIM),
                dtype=np.float64,
            )
            migrated_weights[:, :old_feature_dimension] = saved_weights
            saved_weights = migrated_weights

            self.logger.info(
                f"Migrated model from {old_feature_dimension} "
                f"to {FEATURE_DIM} features."
            )

        self.model = saved_weights
        self.epsilon = float(saved_data.get("epsilon", 1.0))
        self.episodes_trained = int(
            saved_data.get("episodes_trained", 0)
        )

        self.logger.info(
            f"Loaded Q-learning model after "
            f"{self.episodes_trained} training episodes."
        )

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        pickle.PickleError,
    ) as error:
        self.logger.warning(
            f"Could not load saved model: {error}. "
            "Starting with a new model."
        )


def act(self, game_state: dict) -> str:
    """Select an action using an epsilon-greedy strategy."""
    if game_state is None:
        return "WAIT"

    features = state_to_features(game_state)
    valid_indices = valid_action_indices(game_state)

    if self.train and random.random() < self.epsilon:
        action_index = random.choice(valid_indices)
        self.logger.debug(
            f"Exploration: selected {ACTIONS[action_index]} "
            f"with epsilon={self.epsilon:.3f}"
        )
        return ACTIONS[action_index]

    q_values = self.model @ features

    masked_q_values = np.full(len(ACTIONS), -np.inf)
    masked_q_values[valid_indices] = q_values[valid_indices]

    best_q_value = np.max(masked_q_values)
    best_indices = np.flatnonzero(
        np.isclose(masked_q_values, best_q_value)
    )
    action_index = int(np.random.choice(best_indices))

    self.logger.debug(
        f"Exploitation: selected {ACTIONS[action_index]}, "
        f"Q-values={q_values}"
    )

    return ACTIONS[action_index]


def valid_action_indices(game_state: dict) -> list[int]:
    """
    Return physically legal actions plus safe, useful bomb placement.

    BOMB is allowed only when:
    - the agent has a bomb available;
    - at least one crate or opponent is in the blast;
    - a time-aware escape route exists.
    """
    field = game_state["field"]
    x, y = game_state["self"][3]

    blocked_positions = {
        position
        for position, _timer in game_state.get("bombs", [])
    }
    blocked_positions.update(
        other_agent[3]
        for other_agent in game_state.get("others", [])
    )

    valid_indices = []

    for action_index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DIRECTIONS[action]
        next_position = (x + dx, y + dy)

        if (
            field[next_position] == 0
            and next_position not in blocked_positions
        ):
            valid_indices.append(action_index)

    valid_indices.append(ACTIONS.index("WAIT"))

    if bool(game_state["self"][2]):
        crate_count, opponent_count = bomb_target_counts(game_state)
        has_target = crate_count > 0 or opponent_count > 0

        if has_target and has_escape_route_after_bomb(game_state):
            valid_indices.append(ACTIONS.index("BOMB"))

    return valid_indices


def danger_urgency(danger_time: float) -> float:
    """Convert an explosion timer into a normalized danger value."""
    if np.isinf(danger_time):
        return 0.0

    return 1.0 / (float(danger_time) + 1.0)


def state_to_features(game_state: dict) -> np.ndarray | None:
    """Convert the game state into a low-dimensional feature vector."""
    if game_state is None:
        return None

    features = np.zeros(FEATURE_DIM, dtype=np.float64)
    features[0] = 1.0

    valid_indices = valid_action_indices(game_state)

    for action_index in range(4):
        features[1 + action_index] = float(
            action_index in valid_indices
        )

    coins = game_state.get("coins", [])

    if coins:
        features[5] = 1.0

        direction_index, distance = nearest_coin_path(game_state)

        if direction_index is not None:
            features[6 + direction_index] = 1.0

        field = game_state["field"]
        maximum_distance = field.shape[0] + field.shape[1]

        if distance is None:
            features[10] = 1.0
        else:
            features[10] = min(
                distance / maximum_distance,
                1.0,
            )

    danger_times = earliest_danger_times(game_state)
    current_position = game_state["self"][3]

    features[11] = danger_urgency(
        danger_times[current_position]
    )

    safety_actions = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]

    for feature_offset, action in enumerate(safety_actions):
        action_index = ACTIONS.index(action)

        if action == "WAIT":
            next_position = current_position
        else:
            dx, dy = DIRECTIONS[action]
            next_position = (
                current_position[0] + dx,
                current_position[1] + dy,
            )

        if action != "WAIT" and action_index not in valid_indices:
            # Walls, crates, bombs, and opponents make a movement unusable.
            features[12 + feature_offset] = 1.0
        else:
            features[12 + feature_offset] = danger_urgency(
                danger_times[next_position]
            )

    bomb_available = bool(game_state["self"][2])
    features[17] = float(bomb_available)

    if bomb_available:
        safe_bomb = has_escape_route_after_bomb(game_state)
        crate_count, opponent_count = bomb_target_counts(game_state)

        features[18] = float(safe_bomb)
        features[19] = min(crate_count / 4.0, 1.0)
        features[20] = float(opponent_count > 0)

    escape_direction, escape_distance = nearest_safe_path(game_state)

    if escape_direction is not None:
        features[21 + escape_direction] = 1.0

    if escape_distance not in (None, 0):
        features[25] = min(
            escape_distance / max(float(s.BOMB_TIMER), 1.0),
            1.0,
        )

    return features


def nearest_coin_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """Use breadth-first search to find the nearest reachable coin."""
    field = game_state["field"]
    start = game_state["self"][3]
    targets = set(game_state.get("coins", []))

    if not targets:
        return None, None

    blocked_positions = {
        position
        for position, _timer in game_state.get("bombs", [])
    }
    blocked_positions.update(
        other_agent[3]
        for other_agent in game_state.get("others", [])
    )

    queue = deque([(start, None, 0)])
    visited = {start}

    while queue:
        position, first_direction, distance = queue.popleft()

        if position in targets and position != start:
            return first_direction, distance

        x, y = position

        for direction_index, action in enumerate(MOVE_ACTIONS):
            dx, dy = DIRECTIONS[action]
            next_position = (x + dx, y + dy)

            if next_position in visited:
                continue

            if next_position in blocked_positions:
                continue

            if field[next_position] != 0:
                continue

            visited.add(next_position)

            if first_direction is None:
                next_first_direction = direction_index
            else:
                next_first_direction = first_direction

            queue.append(
                (
                    next_position,
                    next_first_direction,
                    distance + 1,
                )
            )

    return None, None