from collections import deque
from pathlib import Path
import pickle
import random

import numpy as np


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
FEATURE_DIM = 11

MODEL_PATH = Path(__file__).resolve().parent / "q_model.pkl"


def setup(self):
    """
    Initialize or load the Q-learning model.

    The model contains one weight vector for each action:
        model.shape == (number_of_actions, number_of_features)
    """
    self.model_path = MODEL_PATH

    # Defaults for a new model.
    self.model = np.zeros((len(ACTIONS), FEATURE_DIM), dtype=np.float64)
    self.epsilon = 1.0
    self.episodes_trained = 0

    if self.model_path.is_file():
        try:
            with self.model_path.open("rb") as file:
                saved_data = pickle.load(file)

            saved_weights = saved_data["weights"]

            expected_shape = (len(ACTIONS), FEATURE_DIM)
            if saved_weights.shape != expected_shape:
                raise ValueError(
                    f"Expected model shape {expected_shape}, "
                    f"but found {saved_weights.shape}."
                )

            self.model = saved_weights
            self.epsilon = saved_data.get("epsilon", 1.0)
            self.episodes_trained = saved_data.get("episodes_trained", 0)

            self.logger.info(
                f"Loaded Q-learning model after "
                f"{self.episodes_trained} training episodes."
            )

        except (OSError, KeyError, TypeError, ValueError, pickle.PickleError) as error:
            self.logger.warning(
                f"Could not load saved model: {error}. "
                "Starting with a new model."
            )
    else:
        self.logger.info("No saved model found. Starting from scratch.")


def act(self, game_state: dict) -> str:
    """
    Select an action using an epsilon-greedy strategy.

    During training:
        - with probability epsilon, explore using a random valid action;
        - otherwise, choose the valid action with the highest Q-value.

    During evaluation:
        - always choose the valid action with the highest Q-value.
    """
    if game_state is None:
        return "WAIT"

    features = state_to_features(game_state)
    valid_indices = valid_action_indices(game_state)

    # Exploration is only used during training.
    if self.train and random.random() < self.epsilon:
        action_index = random.choice(valid_indices)
        self.logger.debug(
            f"Exploration: selected {ACTIONS[action_index]} "
            f"with epsilon={self.epsilon:.3f}"
        )
        return ACTIONS[action_index]

    # The linear model calculates one Q-value for each action.
    q_values = self.model @ features

    # Invalid actions must not be selected.
    masked_q_values = np.full(len(ACTIONS), -np.inf)
    masked_q_values[valid_indices] = q_values[valid_indices]

    best_q_value = np.max(masked_q_values)

    # Random tie-breaking prevents a fixed preference when Q-values are equal.
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
    Return the indices of currently valid actions.

    For Task 1:
        - legal movement actions are allowed;
        - WAIT is allowed;
        - BOMB is intentionally disabled.
    """
    field = game_state["field"]
    x, y = game_state["self"][3]

    blocked_positions = {
        position for position, _timer in game_state["bombs"]
    }
    blocked_positions.update(
        other_agent[3] for other_agent in game_state["others"]
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

    # WAIT is always available.
    valid_indices.append(ACTIONS.index("WAIT"))

    # BOMB is not enabled during Task 1.
    return valid_indices


def state_to_features(game_state: dict) -> np.ndarray | None:
    """
    Convert the game state into a low-dimensional feature vector.
    """
    if game_state is None:
        return None

    features = np.zeros(FEATURE_DIM, dtype=np.float64)

    # A constant feature allows the model to learn a general action bias.
    features[0] = 1.0

    valid_indices = valid_action_indices(game_state)

    # Features 1-4: valid movement directions.
    for action_index in range(4):
        features[1 + action_index] = float(
            action_index in valid_indices
        )

    coins = game_state["coins"]

    if coins:
        # Feature 5: at least one visible coin exists.
        features[5] = 1.0

        direction_index, distance = nearest_coin_path(game_state)

        if direction_index is not None:
            # Features 6-9: first direction on the shortest coin path.
            features[6 + direction_index] = 1.0

        field = game_state["field"]
        maximum_distance = field.shape[0] + field.shape[1]

        if distance is None:
            # A coin exists but is currently unreachable.
            features[10] = 1.0
        else:
            features[10] = min(
                distance / maximum_distance,
                1.0,
            )

    return features


def nearest_coin_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """
    Use breadth-first search to find the nearest reachable coin.

    Returns:
        direction_index:
            0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        distance:
            length of the shortest path to that coin
    """
    field = game_state["field"]
    start = game_state["self"][3]
    targets = set(game_state["coins"])

    if not targets:
        return None, None

    blocked_positions = {
        position for position, _timer in game_state["bombs"]
    }
    blocked_positions.update(
        other_agent[3] for other_agent in game_state["others"]
    )

    # Each queue item contains:
    # (position, first_direction_index, path_distance)
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