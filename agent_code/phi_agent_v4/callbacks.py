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
# --- Task 2: danger awareness ---
# 11:   danger_countdown: 0.0=safe, 0.25/0.5/0.75/1.0=bomb covering this tile
#       with 3/2/1/0 steps left. Larger = more urgent.
# 12:   explosion ongoing UP    (boolean)
# 13:   explosion ongoing RIGHT (boolean)
# 14:   explosion ongoing DOWN  (boolean)
# 15:   explosion ongoing LEFT  (boolean)
# 16:   explosion_efficiency: (crates in blast range) / 9
# --- Task 2: escape planning ---
# 17:   bomb_here_traps_me: 1 if dropping a bomb at the current position would
#       leave no safe tile reachable within BOMB_TIMER steps, else 0
# 18:   escape_dir_UP    (one-hot: first step of shortest escape path)
# 19:   escape_dir_RIGHT
# 20:   escape_dir_DOWN
# 21:   escape_dir_LEFT
FEATURE_DIM = 22

# Set to True to disable bomb placement (e.g. for coin-heaven training).
# Set to False to allow bombs (Task 2 and beyond).
DISABLE_BOMB = False

# Bomb blast range in tiles (matches settings.py BOMB_POWER = 3)
BOMB_POWER = 3

# Steps until a freshly placed bomb explodes (matches settings.py BOMB_TIMER = 4)
BOMB_TIMER = 4

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
    """
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
    Return the indices of currently valid actions.
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

    valid_indices.append(ACTIONS.index("WAIT"))

    if can_place_bomb and not DISABLE_BOMB:
        valid_indices.append(ACTIONS.index("BOMB"))

    return valid_indices


def state_to_features(game_state: dict) -> np.ndarray | None:
    """
    Convert the game state into a low-dimensional feature vector.
    """
    if game_state is None:
        return None

    features = np.zeros(FEATURE_DIM, dtype=np.float64)

    # Feature 0: bias
    features[0] = 1.0

    valid_indices = valid_action_indices(game_state)

    # Features 1-4: valid movement directions.
    for action_index in range(4):
        features[1 + action_index] = float(action_index in valid_indices)

    coins = game_state["coins"]

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
            features[10] = min(distance / maximum_distance, 1.0)

    x, y = game_state["self"][3]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    # Feature 11: how urgently the current tile is threatened (0.0 = safe).
    # danger_countdown already returns 0.0 when no bomb covers the tile, so the
    # coin-heaven case needs no special handling.
    features[11] = danger_countdown(x, y, field, bombs)

    # Features 12-15: ongoing explosion in each adjacent direction.
    for direction_index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DIRECTIONS[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            features[12 + direction_index] = float(explosion_map[nx, ny] > 0)

    # Feature 16: explosion_efficiency
    features[16] = blast_crate_count(x, y, field) / MAX_CRATES

    # Feature 17: would a bomb dropped HERE trap us? (hypothetical=True)
    #
    # The previous version asked "can we escape the bombs that already exist?".
    # With no bomb on the board that is trivially true, so the feature was 1.0 in
    # essentially every state and carried no information. Worse, train.py already
    # hands out SUICIDE_BOMB (-20) for bombing a tile with no escape route, so the
    # agent was being punished for something no feature let it see.
    #
    # Asking the hypothetical question instead makes the feature vary (a dead end
    # gives 1.0, open ground gives 0.0) and lines it up with that penalty.
    # Polarity: 1.0 means "do not drop a bomb here".
    if DISABLE_BOMB:
        # Bombs are disabled, so the question is meaningless.
        features[17] = 0.0
    else:
        _, escapable_if_bombed = escape_path(x, y, field, bombs, hypothetical=True)
        features[17] = float(not escapable_if_bombed)

    # Features 18-21: first step of the escape route from the bombs that are
    # actually on the board right now (hypothetical=False). Unchanged.
    if not DISABLE_BOMB:
        escape_dir, can_escape = escape_path(x, y, field, bombs, hypothetical=False)
        if can_escape and escape_dir is not None:
            features[18 + escape_dir] = 1.0

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
    Return how urgently (x, y) is threatened, on a 0.0 - 1.0 scale.

        0.0  = safe: no bomb covers this tile
        0.25 = a bomb covers it, 3 steps left
        1.00 = a bomb covers it and explodes now

    Safe must map to 0.0, not to a non-zero constant. Roughly 90% of steps are
    safe, so a non-zero "safe" value would be a second copy of the bias feature:
    the two weights could trade off freely against each other, which leaves them
    undetermined and lets a feature carrying no information dominate the Q-values.

    NOTE: larger now means MORE dangerous. The previous version returned the raw
    timer (4.0 = safe), so any comparison of two danger values has the opposite
    sense from before -- see game_events_occurred in train.py.
    """
    min_countdown = None

    for (bx, by), timer in bombs:
        if _in_blast_path(x, y, bx, by, field):
            if min_countdown is None or timer < min_countdown:
                min_countdown = timer

    if min_countdown is None:
        return 0.0

    return (BOMB_TIMER - min_countdown) / BOMB_TIMER


def _in_blast_path(
    x: int,
    y: int,
    bx: int,
    by: int,
    field: np.ndarray,
) -> bool:
    """
    Return True if (x, y) is in the blast path of a bomb at (bx, by).
    Blast is blocked only by stone walls; crates do not stop it.
    """
    if x == bx and y == by:
        return True

    for dx, dy in DIRECTIONS.values():
        for step in range(1, BOMB_POWER + 1):
            tx, ty = bx + dx * step, by + dy * step

            if field[tx, ty] == -1:
                break

            if tx == x and ty == y:
                return True

    return False


def blast_crate_count(x: int, y: int, field: np.ndarray) -> int:
    """
    Count crates a bomb at (x, y) would destroy.
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


def escape_path(
    x: int,
    y: int,
    field: np.ndarray,
    bombs: list,
    hypothetical: bool = False,
) -> tuple[int | None, bool]:
    """
    Find the shortest escape path from (x, y) given active bombs.

    When hypothetical=False (default), only uses existing bombs.
    Use this for features 17-21 so the escape direction stays valid
    at every step during an escape sequence.

    When hypothetical=True, also adds a virtual bomb at (x, y) to simulate
    placing a bomb at the current position. Use this in train.py for
    GOOD_BOMB / SUICIDE_BOMB assessment.

    Returns:
        escape_dir: direction index (0=UP,1=RIGHT,2=DOWN,3=LEFT) of first
                    step on the escape path, or None if no escape exists.
        can_escape: True if a safe tile is reachable within BOMB_TIMER steps.
    """
    danger_tiles = set()

    # Include existing bomb blast paths.
    for (bx, by), _ in bombs:
        danger_tiles.add((bx, by))
        for dx, dy in DIRECTIONS.values():
            for step in range(1, BOMB_POWER + 1):
                tx, ty = bx + dx * step, by + dy * step
                if field[tx, ty] == -1:
                    break
                danger_tiles.add((tx, ty))

    # Optionally add a hypothetical bomb at current position.
    if hypothetical:
        danger_tiles.add((x, y))
        for dx, dy in DIRECTIONS.values():
            for step in range(1, BOMB_POWER + 1):
                tx, ty = x + dx * step, y + dy * step
                if field[tx, ty] == -1:
                    break
                danger_tiles.add((tx, ty))

    # BFS from current position to find nearest safe tile within BOMB_TIMER steps.
    queue = deque([(( x, y), None, 0)])
    visited = {(x, y)}

    while queue:
        position, first_direction, distance = queue.popleft()

        if distance > BOMB_TIMER:
            break

        px, py = position

        # Safe tile reached.
        if position not in danger_tiles:
            return first_direction, True

        for direction_index, action in enumerate(MOVE_ACTIONS):
            ddx, ddy = DIRECTIONS[action]
            next_pos = (px + ddx, py + ddy)
            nx, ny = next_pos

            if next_pos in visited:
                continue
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0:
                continue

            visited.add(next_pos)
            first_dir = direction_index if first_direction is None else first_direction
            queue.append((next_pos, first_dir, distance + 1))

    return None, False


def nearest_coin_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """
    BFS to nearest reachable coin.
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
            next_first_direction = direction_index if first_direction is None else first_direction
            queue.append((next_position, next_first_direction, distance + 1))

    return None, None


def nearest_target_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """
    BFS to nearest reachable target: coins first, crates as fallback.
    """
    if game_state["coins"]:
        direction, distance = nearest_coin_path(game_state)
        if direction is not None:
            return direction, distance

    field = game_state["field"]
    start = game_state["self"][3]

    blocked_positions = {
        position for position, _timer in game_state["bombs"]
    }
    blocked_positions.update(
        other_agent[3] for other_agent in game_state["others"]
    )

    queue = deque([(start, None, 0)])
    visited = {start}

    while queue:
        position, first_direction, distance = queue.popleft()

        x, y = position

        for direction_index, action in enumerate(MOVE_ACTIONS):
            dx, dy = DIRECTIONS[action]
            nx, ny = x + dx, y + dy
            next_position = (nx, ny)

            if next_position in visited:
                continue
            if next_position in blocked_positions:
                continue

            first_dir = direction_index if first_direction is None else first_direction

            if field[nx, ny] == 1:
                return first_dir, distance + 1

            if field[nx, ny] != 0:
                continue

            visited.add(next_position)
            queue.append((next_position, first_dir, distance + 1))

    return None, None
