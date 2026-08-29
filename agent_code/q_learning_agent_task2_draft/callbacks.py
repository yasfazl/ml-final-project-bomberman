"""
Task 2 Q-learning agent -- starting point.

This is the Task 1 agent extended to crates and bombs.  It keeps the same
structure (linear Q-function with one weight vector per action, BFS features)
and fixes the three things that made the first Task 2 attempt walk in circles:

  1. Exactly one constant feature.
     The previous version had a `danger_countdown` feature that returned a
     constant 4.0 whenever no bomb was on the board.  That feature is
     collinear with the bias, and because it is four times larger it makes the
     constant part of ||phi||^2 grow from 1 to 17.  The per-action offsets
     w_a[0] then move roughly seventeen times faster than the direction
     features, so the greedy policy ends up choosing whichever action carries
     the largest offset instead of the one the BFS points at.  Here every
     feature lives in [0, 1] and only feature 0 is constant.

  2. Bootstrapped learning instead of Monte Carlo returns (see train.py).

  3. Reward shaping that uses the same BFS distance as the features
     (see train.py).

Deliberately left out, as things to design and measure:
  * an escape-route search, so the agent knows *where* to run;
  * an exact bomb-timing model (the danger features below only look at the
    raw timer, not at whether the agent can actually get out in time);
  * any safety filter on the action set;
  * loop detection, opponent handling, other function approximators.
"""

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

# Same order as MOVE_ACTIONS, indexable by action index 0..3.
MOVE_DELTAS = [DIRECTIONS[action] for action in MOVE_ACTIONS]

# Game rules, mirrored from settings.py.
BOMB_POWER = 3
BOMB_TIMER = 4

# A bomb destroying this many crates already gets the maximum feature value.
CRATE_SATURATION = 4.0

# Set to True to forbid bomb placement, e.g. to check the movement part of the
# agent in the "coin-heaven" scenario.  Set to False for Task 2.
DISABLE_BOMB = False

# Feature vector.  Every entry is in [0, 1] and only feature 0 is constant.
#
#   0        bias
#   1 -  4   whether UP, RIGHT, DOWN, LEFT are valid moves
#   5        whether a target exists at all
#   6 -  9   first direction of the shortest BFS path to the target (one-hot)
#  10        normalised distance to the target
#  11        the agent stands in the blast range of a bomb
#  12        how soon that bomb goes off, in [0, 1]
#  13 - 16   moving in this direction leads into a blast range
#  17        crates a bomb dropped here would destroy (normalised)
FEATURE_DIM = 18

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

    valid_indices = valid_action_indices(game_state)

    # Exploration is only used during training.
    if self.train and random.random() < self.epsilon:
        action_index = random.choice(valid_indices)
        self.logger.debug(
            f"Exploration: selected {ACTIONS[action_index]} "
            f"with epsilon={self.epsilon:.3f}"
        )
        return ACTIONS[action_index]

    features = state_to_features(game_state)

    # The linear model calculates one Q-value for each action.
    q_values = self.model @ features

    # Invalid actions must not be selected.
    masked_q_values = np.full(len(ACTIONS), -np.inf)
    masked_q_values[valid_indices] = q_values[valid_indices]

    best_q_value = np.max(masked_q_values)

    # Random tie-breaking prevents a fixed preference when Q-values are equal.
    best_indices = np.flatnonzero(np.isclose(masked_q_values, best_q_value))
    action_index = int(np.random.choice(best_indices))

    self.logger.debug(
        f"Exploitation: selected {ACTIONS[action_index]}, "
        f"Q-values={np.round(q_values, 2)}"
    )

    return ACTIONS[action_index]


def valid_action_indices(game_state: dict) -> list[int]:
    """
    Return the indices of currently valid actions.

    Valid means "accepted by the game engine", i.e. everything that does not
    produce an INVALID_ACTION event:
        - legal movement actions;
        - WAIT;
        - BOMB, if the agent has no active bomb.

    Note that this says nothing about whether an action is a good idea.
    Filtering out suicidal actions here is one of the things left to try.
    """
    field = game_state["field"]
    x, y = game_state["self"][3]
    can_place_bomb = game_state["self"][2]

    blocked_positions = {position for position, _timer in game_state["bombs"]}
    blocked_positions.update(other_agent[3] for other_agent in game_state["others"])

    valid_indices = []

    for action_index, (dx, dy) in enumerate(MOVE_DELTAS):
        next_position = (x + dx, y + dy)

        if field[next_position] == 0 and next_position not in blocked_positions:
            valid_indices.append(action_index)

    # WAIT is always available.
    valid_indices.append(ACTIONS.index("WAIT"))

    # BOMB is available only when no own bomb is currently active
    # and bombing is not globally disabled.
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

    field = game_state["field"]
    x, y = game_state["self"][3]

    # -----------------------------------------------------------------------
    # Features 0-10: where to walk next (Task 1, with crates as extra targets)
    # -----------------------------------------------------------------------

    # A constant feature allows the model to learn a general action bias.
    # This is the *only* constant feature; see the module docstring.
    features[0] = 1.0

    valid_indices = valid_action_indices(game_state)

    # Features 1-4: valid movement directions.
    for action_index in range(4):
        features[1 + action_index] = float(action_index in valid_indices)

    direction_index, distance, _target_is_coin = nearest_target(game_state)

    if distance is None:
        # Nothing to walk to; treat that as "maximally far away".
        features[10] = 1.0
    else:
        features[5] = 1.0

        if direction_index is not None:
            # Features 6-9: first direction on the shortest path.
            features[6 + direction_index] = 1.0

        maximum_distance = field.shape[0] + field.shape[1]
        features[10] = min(distance / maximum_distance, 1.0)

    # -----------------------------------------------------------------------
    # Features 11-17: bombs and crates
    # -----------------------------------------------------------------------

    danger = danger_map(field, game_state["bombs"], game_state["explosion_map"])

    # Features 11-12: is the agent inside a blast range, and how soon does it
    # go off?  A timer of 0 gives 1.0, a fresh bomb gives 0.0.  Note this is
    # zero whenever the agent is safe, so it is not a second bias.
    if np.isfinite(danger[x, y]):
        features[11] = 1.0
        features[12] = float(np.clip(1.0 - danger[x, y] / BOMB_TIMER, 0.0, 1.0))

    # Features 13-16: the neighbouring tile in this direction is inside a
    # blast range.
    for action_index, (dx, dy) in enumerate(MOVE_DELTAS):
        features[13 + action_index] = float(np.isfinite(danger[x + dx, y + dy]))

    # Feature 17: how many crates a bomb dropped here would destroy.
    features[17] = min(crate_count(x, y, field) / CRATE_SATURATION, 1.0)

    return features


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def blast_range(bx: int, by: int, field: np.ndarray) -> list[tuple[int, int]]:
    """
    Tiles covered by a bomb at (bx, by).

    Mirrors Bomb.get_blast_coords in items.py: the blast reaches up to
    BOMB_POWER tiles in each cardinal direction and is stopped by stone walls
    only.  Crates are destroyed but do not stop the blast.
    """
    coords = [(bx, by)]

    for dx, dy in MOVE_DELTAS:
        for step in range(1, BOMB_POWER + 1):
            tx, ty = bx + dx * step, by + dy * step

            if field[tx, ty] == -1:
                break

            coords.append((tx, ty))

    return coords


def danger_map(
    field: np.ndarray,
    bombs: list,
    explosion_map: np.ndarray,
) -> np.ndarray:
    """
    For every tile, the smallest bomb timer that threatens it, or np.inf if
    nothing does.  Tiles that are burning right now count as timer 0.

    This is a rough model: it says *whether* a tile is dangerous, not whether
    the agent could still walk out of it in time.  Turning this into a proper
    time-aware model is one of the obvious extensions.
    """
    danger = np.full(field.shape, np.inf)

    for (bx, by), timer in bombs:
        for tile in blast_range(bx, by, field):
            danger[tile] = min(danger[tile], timer)

    danger[explosion_map > 0] = 0.0

    return danger


def crate_count(x: int, y: int, field: np.ndarray) -> int:
    """
    Number of crates a bomb dropped at (x, y) would destroy.
    """
    return sum(1 for tile in blast_range(x, y, field) if field[tile] == 1)


def nearest_target(game_state: dict) -> tuple[int | None, int | None, bool]:
    """
    Use breadth-first search to find the nearest thing worth walking to.

    Coins have priority.  If no coin is reachable, the target is the nearest
    tile from which a bomb would destroy at least one crate.  Standing on such
    a tile already counts, in which case the direction is None and the
    distance is 0 -- that is the situation in which the agent should bomb.

    Returns:
        direction_index: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, or None
        distance:        length of the shortest path, or None
        target_is_coin:  True if the target is a coin
    """
    field = game_state["field"]
    start = game_state["self"][3]
    coins = set(game_state["coins"])

    blocked_positions = {position for position, _timer in game_state["bombs"]}
    blocked_positions.update(other_agent[3] for other_agent in game_state["others"])

    crate_spot = None

    # Each queue item contains:
    # (position, first_direction_index, path_distance)
    queue = deque([(start, None, 0)])
    visited = {start}

    while queue:
        position, first_direction, distance = queue.popleft()

        if position in coins:
            return first_direction, distance, True

        if crate_spot is None and crate_count(position[0], position[1], field) > 0:
            crate_spot = (first_direction, distance)

        x, y = position

        for direction_index, (dx, dy) in enumerate(MOVE_DELTAS):
            next_position = (x + dx, y + dy)

            if next_position in visited:
                continue

            if next_position in blocked_positions:
                continue

            if field[next_position] != 0:
                continue

            visited.add(next_position)

            queue.append(
                (
                    next_position,
                    direction_index if first_direction is None else first_direction,
                    distance + 1,
                )
            )

    if crate_spot is not None:
        return crate_spot[0], crate_spot[1], False

    return None, None, False


def target_distance(game_state: dict) -> tuple[int | None, bool]:
    """
    Distance to the current target and its kind.  Used by the reward shaping
    in train.py so that rewards and features measure the same thing.
    """
    if game_state is None:
        return None, False

    _direction, distance, target_is_coin = nearest_target(game_state)

    return distance, target_is_coin
