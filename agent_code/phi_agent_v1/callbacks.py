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
# --- Inherited from Task 1 (teammate's code) ---
# 0:    bias
# 1-4:  whether UP, RIGHT, DOWN, LEFT are valid moves
# 5:    whether at least one visible coin exists
# 6-9:  first direction of the shortest BFS path to a coin (one-hot)
# 10:   normalized distance to the nearest reachable coin
# --- New for Task 2 ---
# 11:   danger_countdown: 4=safe, 0-3=steps until nearby bomb explodes
# 12:   explosion ongoing UP    (boolean)
# 13:   explosion ongoing RIGHT (boolean)
# 14:   explosion ongoing DOWN  (boolean)
# 15:   explosion ongoing LEFT  (boolean)
# 16:   explosion_efficiency: (crates in blast range) / 9
FEATURE_DIM = 17

# Bomb blast range in tiles (matches settings.py BOMB_POWER = 3)
BOMB_POWER = 3

# Maximum crates a single bomb can destroy (board geometry constraint)
MAX_CRATES = 9

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

    For Task 2:
        - legal movement actions are allowed;
        - WAIT is allowed;
        - BOMB is allowed only when the agent has no active bomb.
    """
    field = game_state["field"]
    x, y = game_state["self"][3]
    can_place_bomb = game_state["self"][2]

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

    # BOMB is available only when no own bomb is currently active.
    if can_place_bomb:
        valid_indices.append(ACTIONS.index("BOMB"))

    return valid_indices


def state_to_features(game_state: dict) -> np.ndarray | None:
    """
    Convert the game state into a low-dimensional feature vector.
    """
    if game_state is None:
        return None

    features = np.zeros(FEATURE_DIM, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Features 0-10: inherited from Task 1 (teammate's code, unchanged)
    # -----------------------------------------------------------------------

    # Feature 0: bias
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

    # -----------------------------------------------------------------------
    # Features 11-16: new Task 2 features
    # -----------------------------------------------------------------------

    x, y = game_state["self"][3]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    # Feature 11: danger_countdown
    # 4 = safe (not in any bomb's blast path)
    # 0-3 = steps remaining until the threatening bomb explodes
    features[11] = danger_countdown(x, y, field, bombs)

    # Features 12-15: ongoing explosion in each direction
    # True if any of the immediately adjacent tiles in that direction
    # currently has an active explosion (explosion_map > 0).
    for direction_index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DIRECTIONS[action]
        nx, ny = x + dx, y + dy

        # Check bounds before indexing.
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            features[12 + direction_index] = float(
                explosion_map[nx, ny] > 0
            )

    # Feature 16: explosion_efficiency
    # Fraction of crates (out of MAX_CRATES) that a bomb placed at the
    # agent's current position would destroy.
    features[16] = blast_crate_count(x, y, field) / MAX_CRATES

    return features


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def danger_countdown(
    x: int,
    y: int,
    field: np.ndarray,
    bombs: list,
) -> float:
    """
    Return the minimum countdown among all bombs whose blast path
    covers (x, y), or 4.0 if the agent is not in any blast path.

    Countdown values from the framework: 0 = explodes this step,
    4 = just placed (safe sentinel).
    """
    min_countdown = 4  # sentinel for "safe" (timer=4 means just placed, not yet dangerous)

    for (bx, by), timer in bombs:
        if _in_blast_path(x, y, bx, by, field):
            if timer < min_countdown:
                min_countdown = timer

    return float(min_countdown)


def _in_blast_path(
    x: int,
    y: int,
    bx: int,
    by: int,
    field: np.ndarray,
) -> bool:
    """
    Return True if tile (x, y) lies within the blast path of a bomb
    at (bx, by).

    The blast travels up to BOMB_POWER tiles in each cardinal direction
    and is blocked only by stone walls (field == -1). Crates are destroyed
    but do not stop the blast. The bomb tile itself is also dangerous.
    """
    if x == bx and y == by:
        return True

    # Check along each cardinal direction.
    for dx, dy in DIRECTIONS.values():
        for step in range(1, BOMB_POWER + 1):
            tx, ty = bx + dx * step, by + dy * step

            # Stop only at stone walls.
            if field[tx, ty] == -1:
                break

            if tx == x and ty == y:
                return True

    return False


def blast_crate_count(x: int, y: int, field: np.ndarray) -> int:
    """
    Count how many crates a bomb placed at (x, y) would destroy.
    Blast travels up to BOMB_POWER tiles in each direction and stops
    at stone walls; crates are counted and then the blast stops.
    """
    count = 0

    for dx, dy in DIRECTIONS.values():
        for step in range(1, BOMB_POWER + 1):
            tx, ty = x + dx * step, y + dy * step

            if field[tx, ty] == -1:
                break

            if field[tx, ty] == 1:
                count += 1

    return count


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
