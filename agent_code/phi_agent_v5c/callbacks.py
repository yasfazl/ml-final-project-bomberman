"""
Task 2 agent, action-conditioned linear Q-function.

    Q(s, a) = w . phi(s, a)          (one shared weight vector)

instead of the previous

    Q(s, a) = W[a] . phi(s)          (one weight row per action)

WHY
---
With a weight row per action, everything learned while moving LEFT was written
to the LEFT row only. Measured on the previous model at epsilon=0, LEFT was
0.9% of all actions, so the LEFT row was starved while the UP/DOWN rows were
trained on 97% of the data.

Worse, every direction-agnostic feature (bias, danger, crate_eff, trap) ended up
with four *different* weights across the four move actions, even though the
quantity it measures has nothing to do with direction. Those differences are
sampling noise, not signal: the arena is rotation-symmetric (start corners are
permuted in environment.py, crates are i.i.d.), so no direction is inherently
better. Measured spread of that noise was 10.68 in total, while the escape
signal it had to beat was only 0.26 - 2.81. Escape accuracy per direction was
UP 98.6% / RIGHT 85.2% / DOWN 90.6% / LEFT 77.0%; equalising the noise by hand
lifted the total from 88.3% to 95.6%.

HOW
---
The four direction blocks are rolled into the frame of the action being scored,
so index 0 always means "the way I am considering going":

    slot 0 = ahead        slot 1 = right-hand
    slot 2 = behind       slot 3 = left-hand

(ACTIONS is ordered clockwise, so np.roll(block, -a) produces exactly this.)

Rolling is a permutation, so no information is lost. What is lost -- deliberately
-- is the ability to say "UP is intrinsically better than LEFT".

The four move actions then share one set of weights, and the direction-agnostic
features contribute the same amount to all four, so they cancel in the argmax
over directions. They still separate move / WAIT / BOMB, which have their own
blocks.

    move weights 25 + wait 9 + bomb 7 = 41 parameters   (previously 6 x 22 = 132)
"""
from collections import deque
from pathlib import Path
import pickle
import random

import numpy as np


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

MOVE_ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

# Clockwise. np.roll(block, -a) therefore yields [ahead, right, behind, left].
DIRECTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

# ---------------------------------------------------------------------------
# phi(s, a) layout
# ---------------------------------------------------------------------------
# MOVE block -- active only when a is UP/RIGHT/DOWN/LEFT. Shared by all four.
#   0        move bias
#   1 -  4   can walk this way          [ahead, right, behind, left]
#   5 -  8   BFS target lies this way   [ahead, right, behind, left]
#   9 - 12   explosion ongoing this way [ahead, right, behind, left]
#  13 - 16   live bomb covers the tile  [ahead, right, behind, left]
#  17 - 20   escape route starts here   [ahead, right, behind, left]
#  21        a coin is visible
#  22        normalised distance to nearest target
#  23        danger_countdown of the tile we stand on
#  24        explosion_efficiency of the tile we stand on
#  25        dropping a bomb here would trap us
#  26        danger x escape-ahead        (interaction)
#  27        danger x explosion-ahead     (interaction)
#  28        target distance x target-ahead (interaction)
#
# Slots 13-16 complete the time axis that features 9-12 only half covered:
#
#     9 - 12   fire that has ALREADY gone off   (explosion_map)
#    13 - 16   fire that is ABOUT to go off     (live bombs), relative to here
#
# Without them nothing described the tile the agent was about to step onto. Only
# feature 23 mentions bomb danger and it describes the tile already occupied, and
# the escape block (17-20) switches off the moment the agent reaches safety --
# escape_path returns direction None once the current tile is clear, so the blast
# map it computed internally is discarded. Standing safe one step after its own
# bomb, the agent therefore had exactly one direction signal left, the BFS target,
# which points back at the crate it just bombed. It walked back in and died.
#
# One tile of lookahead is enough: dying requires stepping onto a covered tile,
# and every such step is preceded by a state where that tile is "ahead". The
# value is graded (0.25 -> 1.0), so "just placed, safe to cross" stays separable
# from "detonating now".
#
# WAIT block -- rotation-invariant summaries only, so no direction bias can
# creep back in through this block.
#  25        wait bias
#  26        number of legal moves / 4
#  27        a coin is visible
#  28        normalised distance to nearest coin
#  29        danger_countdown
#  30        explosion_efficiency
#  31        bomb here would trap us
#  32        an escape route exists
#  33        adjacent ongoing explosions / 4
#
# BOMB block
#  34        bomb bias
#  35        number of legal moves / 4
#  36        danger_countdown
#  37        explosion_efficiency
#  38        bomb here would trap us
#  39        a coin is visible
#  40        an escape route exists
MOVE_OFFSET, MOVE_DIM = 0, 29
WAIT_OFFSET, WAIT_DIM = 29, 9
BOMB_OFFSET, BOMB_DIM = 38, 7

FEATURE_DIM = MOVE_DIM + WAIT_DIM + BOMB_DIM   # 45

# Interaction features 26-28. Set to False to ablate them; the vector keeps its
# length either way, so models stay loadable across the switch.
USE_INTERACTIONS = True

# Direction block 13-16, "a live bomb covers the tile that way". Set to False to
# reproduce the 41-feature behaviour, which escaped its own bomb perfectly and
# then stepped straight back into the blast. Length is unchanged either way.
USE_DANGER_AHEAD = True

# Which BFS supplies the "target" direction block.
#
#   True  -> nearest_target_path: coins first, crates as fallback
#   False -> nearest_coin_path:   coins only (what Task 1 shipped)
#
# game_state["coins"] holds only *collectable* coins (environment.py:407), and
# in classic the coins start inside crates, so early in a round that list is
# empty. With coins-only targeting the whole target block is then zero, and in a
# safe tile with no explosion and no bomb NOTHING in the move block distinguishes
# the four directions except local geometry -- which is the same every time the
# agent stands there, so it walks in circles.
#
# train.py already hands out MOVED_TOWARD_TARGET / MOVED_AWAY_FROM_TARGET using
# nearest_target_path, i.e. it rewarded walking toward crates while the features
# refused to show where the crates were. train.py now reads the distance out of
# this summary instead of running its own BFS, so this switch moves the feature
# AND the reward together. That is deliberate: the two disagreeing is exactly
# the bug that made the agent walk in circles.
USE_CRATE_TARGETS = True

# Set to True to disable bomb placement (e.g. for coin-heaven training).
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
    Initialise or load the Q-learning model.

    The model is now a single weight vector, not one row per action.
    """
    self.model_path = MODEL_PATH

    self.model = np.zeros(FEATURE_DIM, dtype=np.float64)
    self.epsilon = 1.0
    self.episodes_trained = 0

    if self.model_path.is_file():
        try:
            with self.model_path.open("rb") as file:
                saved_data = pickle.load(file)

            saved_weights = saved_data["weights"]

            if saved_weights.shape != (FEATURE_DIM,):
                raise ValueError(
                    f"Expected model shape {(FEATURE_DIM,)}, "
                    f"but found {saved_weights.shape}. A (6, 22) model belongs "
                    "to the previous per-action-row agent and cannot be reused."
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

    summary = state_summary(game_state)
    valid_indices = summary["valid_indices"]

    if self.train and random.random() < self.epsilon:
        action_index = random.choice(valid_indices)
        self.logger.debug(
            f"Exploration: selected {ACTIONS[action_index]} "
            f"with epsilon={self.epsilon:.3f}"
        )
        return ACTIONS[action_index]

    q_values = action_values(self.model, summary)

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



# ---------------------------------------------------------------------------
# phi(s, a)
# ---------------------------------------------------------------------------

def state_summary(game_state: dict) -> dict | None:
    """
    Everything about the state that phi() needs, computed once.

    The BFS calls (nearest_coin_path, escape_path) are the expensive part, so
    they happen here and the six phi() calls that follow are cheap array work.

    Every quantity below is defined exactly as in the previous agent's
    state_to_features; only the way they are laid out into a vector changes.
    """
    if game_state is None:
        return None

    x, y = game_state["self"][3]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    valid_indices = valid_action_indices(game_state)

    walkable = np.zeros(4, dtype=np.float64)
    for action_index in range(4):
        walkable[action_index] = float(action_index in valid_indices)

    # Target direction and distance.
    #
    # coin_visible keeps its Task 1 meaning ("is any collectable coin on the
    # board"). It is deliberately NOT "does a target exist": with crates as a
    # fallback a target almost always exists, so such a feature would be a
    # constant, i.e. a second copy of the block's bias carrying no information.
    target = np.zeros(4, dtype=np.float64)
    target_distance = 0.0
    coin_visible = float(bool(game_state["coins"]))

    if USE_CRATE_TARGETS:
        direction_index, distance = nearest_target_path(game_state)
    elif game_state["coins"]:
        direction_index, distance = nearest_coin_path(game_state)
    else:
        direction_index, distance = None, None

    if direction_index is not None:
        target[direction_index] = 1.0

        maximum_distance = field.shape[0] + field.shape[1]

        if distance is None:
            target_distance = 1.0
        else:
            target_distance = min(distance / maximum_distance, 1.0)

    danger = danger_countdown(x, y, field, bombs)

    explosion = np.zeros(4, dtype=np.float64)
    for direction_index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DIRECTIONS[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            explosion[direction_index] = float(explosion_map[nx, ny] > 0)

    # How much MORE threatened the neighbouring tile is than the one we stand on.
    #
    # The absolute version of this feature (just danger_countdown of the tile
    # ahead) made things worse, and the reason is geometry: a blast is a cross of
    # radius BOMB_POWER, so walking out of one means walking along covered tiles.
    # The feature therefore fired on every step of an escape, taxing exactly the
    # moves that save the agent -- while WAIT, having no tile "ahead", paid
    # nothing. Measured on that version, right after dropping a bomb WAIT beat
    # every move at 2, 3 and 4 exits, so the agent burned the first of its four
    # escape moves standing still and then could not cover the distance.
    #
    # Taking the difference removes that. One bomb gives every tile it covers the
    # same countdown, so the term is exactly 0 while escaping, and the clamp keeps
    # it 0 when stepping from a covered tile to a clear one. What survives is the
    # case it was built for: standing safe and stepping into a live blast.
    neighbour_danger = np.zeros(4, dtype=np.float64)

    if USE_DANGER_AHEAD and bombs:
        here = danger_countdown(x, y, field, bombs)

        for direction_index, action in enumerate(MOVE_ACTIONS):
            dx, dy = DIRECTIONS[action]
            nx, ny = x + dx, y + dy

            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                ahead = danger_countdown(nx, ny, field, bombs)
                neighbour_danger[direction_index] = max(0.0, ahead - here)

    crate_efficiency = blast_crate_count(x, y, field) / MAX_CRATES

    if DISABLE_BOMB:
        # Bombs are disabled, so the question is meaningless.
        trap = 0.0
    else:
        _, escapable_if_bombed = escape_path(x, y, field, bombs, hypothetical=True)
        trap = float(not escapable_if_bombed)

    escape = np.zeros(4, dtype=np.float64)
    escape_exists = 0.0

    if not DISABLE_BOMB:
        escape_dir, can_escape = escape_path(x, y, field, bombs, hypothetical=False)
        if can_escape and escape_dir is not None:
            escape[escape_dir] = 1.0
            escape_exists = 1.0

    return {
        "valid_indices": valid_indices,
        "walkable": walkable,
        "target": target,
        "coin_visible": coin_visible,
        "target_distance": target_distance,
        # Unnormalised BFS distance, so train.py can fire MOVED_TOWARD_TARGET
        # off the same number the features use instead of repeating the BFS.
        "target_distance_raw": distance,
        "danger": danger,
        "explosion": explosion,
        "neighbour_danger": neighbour_danger,
        "crate_efficiency": crate_efficiency,
        "trap": trap,
        "escape": escape,
        "escape_exists": escape_exists,
    }


def phi(summary: dict, action_index: int) -> np.ndarray:
    """
    Build the feature vector for one (state, action) pair.

    For a move action the four direction blocks are rolled so that slot 0 is
    the direction being scored. For WAIT and BOMB there is no direction, so
    their blocks hold rotation-invariant summaries (counts and scalars) only.
    """
    vector = np.zeros(FEATURE_DIM, dtype=np.float64)

    if action_index < 4:
        # np.roll(block, -a) puts the scored direction in slot 0 and, because
        # ACTIONS is ordered clockwise, leaves [ahead, right, behind, left].
        walkable  = np.roll(summary["walkable"],  -action_index)
        target    = np.roll(summary["target"],    -action_index)
        explosion = np.roll(summary["explosion"], -action_index)
        bomb_ahead = np.roll(summary["neighbour_danger"], -action_index)
        escape    = np.roll(summary["escape"],    -action_index)

        vector[0]      = 1.0
        vector[1:5]    = walkable
        vector[5:9]    = target
        vector[9:13]   = explosion
        vector[13:17]  = bomb_ahead
        vector[17:21]  = escape
        vector[21]     = summary["coin_visible"]
        vector[22]     = summary["target_distance"]
        vector[23]     = summary["danger"]
        vector[24]     = summary["crate_efficiency"]
        vector[25]     = summary["trap"]

        if USE_INTERACTIONS:
            # A linear model cannot form products on its own, and tying the
            # weights across directions removes the crude stand-in the
            # per-action rows used to provide. These three are the products the
            # policy actually needs: how much the escape direction matters
            # depends on how urgent the danger is.
            vector[26] = summary["danger"] * escape[0]
            vector[27] = summary["danger"] * explosion[0]
            vector[28] = summary["target_distance"] * target[0]

    elif action_index == ACTIONS.index("WAIT"):
        vector[WAIT_OFFSET + 0] = 1.0
        vector[WAIT_OFFSET + 1] = summary["walkable"].sum() / 4.0
        vector[WAIT_OFFSET + 2] = summary["coin_visible"]
        vector[WAIT_OFFSET + 3] = summary["target_distance"]
        vector[WAIT_OFFSET + 4] = summary["danger"]
        vector[WAIT_OFFSET + 5] = summary["crate_efficiency"]
        vector[WAIT_OFFSET + 6] = summary["trap"]
        vector[WAIT_OFFSET + 7] = summary["escape_exists"]
        vector[WAIT_OFFSET + 8] = summary["explosion"].sum() / 4.0

    else:
        vector[BOMB_OFFSET + 0] = 1.0
        vector[BOMB_OFFSET + 1] = summary["walkable"].sum() / 4.0
        vector[BOMB_OFFSET + 2] = summary["danger"]
        vector[BOMB_OFFSET + 3] = summary["crate_efficiency"]
        vector[BOMB_OFFSET + 4] = summary["trap"]
        vector[BOMB_OFFSET + 5] = summary["coin_visible"]
        vector[BOMB_OFFSET + 6] = summary["escape_exists"]

    return vector


def action_values(model: np.ndarray, summary: dict) -> np.ndarray:
    """
    Q(s, a) for all six actions. Unmasked -- callers apply valid_indices.
    """
    return np.array(
        [model @ phi(summary, action_index) for action_index in range(len(ACTIONS))]
    )


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
