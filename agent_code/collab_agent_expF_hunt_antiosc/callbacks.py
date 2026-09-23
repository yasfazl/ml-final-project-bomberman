from collections import deque
from pathlib import Path
import os
import pickle
import random

import numpy as np
import settings as s

from .game_utils import (
    action_has_survival_route,
    best_crate_bombing_path,
    blast_tiles,
    bomb_has_robust_escape_route,
    bomb_target_counts,
    earliest_danger_times,
    has_escape_route_after_bomb,
    nearest_crate_bombing_path,
    nearest_safe_path,
    tile_is_safe_at_time,
    time_indexed_danger_schedule,
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
# 26: whether a reachable safe crate-bombing tile exists
# 27-30: first direction toward that crate-bombing tile
# 31: normalized distance to that crate-bombing tile
# 32: whether a better nearby safe crate-bombing position exists
# 33-36: first direction toward that better position
# 37: normalized distance to that better position
# 38: normalized crate yield at the best nearby position
# 39-42: first direction of the shortest path to the nearest opponent
# 43: how close that opponent is, 1 = adjacent, 0 = none or far away
#
# The hunt block, 39-43, is filled in only while the field holds no crate and
# no coin is reachable.  Two measurements decide that gate.  First, once the
# crates are gone the agent has nothing left to walk towards: 59.5% of its
# decisions happen in that state and half of them are spent waiting.  Second,
# an opponent is inside the blast of a bomb it could drop in only 11% of those
# states, so the chances it does get come to it rather than being sought out.
# The block stays empty while crates or coins remain, so the coin rate of 2.59
# per round, which is where most of the score comes from, cannot be disturbed.
FEATURE_DIM = 44

# A safe agent must also make progress.  If the learned policy selects WAIT
# three times at the same safe position, temporarily remove WAIT when the
# existing safety filters have already approved a move toward a coin or crate
# objective.
ANTI_STALL_WAIT_LIMIT = 3

# Once no crate and no reachable coin are left, the hunt features can leave the
# agent stepping back and forth between two tiles.  When its last four
# positions read A, B, A, B, the move back to A is taken out of the candidates,
# as long as the first move toward the nearest opponent is still approved and
# is a different move.
ANTI_OSCILLATION_HISTORY = 4

# Moving first must provide a material improvement.  A one-crate gain was too
# common in smoke evaluation and reduced score by delaying otherwise useful
# bombs; a gain of two keeps the guard focused on clear 1 -> 3 opportunities.
CRATE_DEFERRAL_MIN_GAIN = 2

MODEL_PATH = Path(__file__).resolve().parent / "q_model.pkl"


def setup(self):
    """
    Initialize or load the Q-learning model.

    Smaller compatible models are migrated to FEATURE_DIM features by
    preserving their old weights and initializing new weights to zero.
    """
    self.model_path = MODEL_PATH

    self.model = np.zeros(
        (len(ACTIONS), FEATURE_DIM),
        dtype=np.float64,
    )
    self.epsilon = 1.0
    self.episodes_trained = 0
    self.last_own_bomb_position = None
    self.last_seen_round = None
    self.consecutive_safe_waits = 0
    self.last_wait_position = None
    self.recent_positions = deque(maxlen=ANTI_OSCILLATION_HISTORY)

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

    current_round = game_state.get("round")

    if getattr(self, "last_seen_round", None) != current_round:
        self.last_seen_round = current_round
        self.last_own_bomb_position = None
        self.consecutive_safe_waits = 0
        self.last_wait_position = None
        self.recent_positions = deque(maxlen=ANTI_OSCILLATION_HISTORY)

    self.recent_positions.append(tuple(game_state["self"][3]))
    _clear_inactive_own_bomb_memory(self, game_state)

    features = state_to_features(game_state)
    valid_indices = valid_action_indices(game_state)
    survivable_indices = _time_expanded_candidate_indices(
        game_state,
        valid_indices,
    )
    candidate_indices = _post_bomb_candidate_indices(
        self,
        game_state,
        survivable_indices,
    )
    candidate_indices = _crate_bomb_deferral_candidate_indices(
        game_state,
        candidate_indices,
    )
    candidate_indices = _anti_stall_candidate_indices(
        self,
        game_state,
        candidate_indices,
    )
    candidate_indices = _anti_oscillation_candidate_indices(
        self,
        game_state,
        candidate_indices,
    )

    if self.train and random.random() < self.epsilon:
        action_index = random.choice(candidate_indices)
        chosen_action = ACTIONS[action_index]
        _remember_own_bomb_if_selected(self, chosen_action, game_state)
        _record_anti_stall_choice(self, chosen_action, game_state)
        self.logger.debug(
            f"Exploration: selected {chosen_action} "
            f"with epsilon={self.epsilon:.3f}"
        )
        _trace_decision(
            game_state,
            features,
            candidate_indices,
            None,
            chosen_action,
            "explore",
        )
        return chosen_action

    q_values = self.model @ features

    masked_q_values = np.full(len(ACTIONS), -np.inf)
    masked_q_values[candidate_indices] = q_values[candidate_indices]

    best_q_value = np.max(masked_q_values)
    best_indices = np.flatnonzero(
        np.isclose(masked_q_values, best_q_value)
    )
    action_index = int(np.random.choice(best_indices))
    chosen_action = ACTIONS[action_index]
    _remember_own_bomb_if_selected(self, chosen_action, game_state)
    _record_anti_stall_choice(self, chosen_action, game_state)

    self.logger.debug(
        f"Exploitation: selected {chosen_action}, "
        f"Q-values={q_values}"
    )

    _trace_decision(
        game_state,
        features,
        candidate_indices,
        q_values,
        chosen_action,
        "exploit",
    )

    return chosen_action


def _current_position_has_known_danger(game_state: dict) -> bool:
    """Return whether known danger will reach the current tile."""
    position = tuple(game_state["self"][3])
    danger = time_indexed_danger_schedule(game_state)
    return bool(np.any(danger[1:, position[0], position[1]]))


def _progress_direction_index(game_state: dict) -> int | None:
    """Return the first movement direction toward the active objective."""
    if game_state.get("coins", []):
        direction_index, distance = nearest_coin_path(game_state)

        if direction_index is not None and distance not in (None, 0):
            return int(direction_index)

    # A visible but currently unreachable coin must not suppress crate
    # progress.  Opening the board is the only way to reach it.
    direction_index, distance = nearest_crate_bombing_path(game_state)

    if direction_index is None or distance in (None, 0):
        return None

    return int(direction_index)


def _state_after_movement(
    game_state: dict,
    direction_index: int,
) -> dict:
    """Return a shallow state copy with the agent moved one tile."""
    action = ACTIONS[direction_index]
    dx, dy = DIRECTIONS[action]
    current_position = tuple(game_state["self"][3])
    next_position = (
        current_position[0] + dx,
        current_position[1] + dy,
    )

    simulated_state = dict(game_state)
    agent_info = list(game_state["self"])
    agent_info[3] = next_position
    simulated_state["self"] = tuple(agent_info)
    return simulated_state


def _crate_bomb_deferral_candidate_indices(
    game_state: dict,
    candidate_indices: list[int],
) -> list[int]:
    """Move one tile before bombing when it safely improves crate yield.

    This is intentionally restricted to safe crate mode.  It never defers an
    opponent-targeting bomb, never runs when known danger reaches the current
    tile, and never interrupts pursuit of a reachable coin.  Distant bombs do
    not disable the optimization.  The selected movement must already have
    survived the ordinary candidate filters, and placing a bomb at the
    destination must pass the robust escape check.
    """
    bomb_index = ACTIONS.index("BOMB")
    if bomb_index not in candidate_indices:
        return candidate_indices

    if _current_position_has_known_danger(game_state):
        return candidate_indices

    coin_direction, coin_distance = nearest_coin_path(game_state)
    if coin_direction is not None and coin_distance not in (None, 0):
        return candidate_indices

    current_crate_count, current_opponent_count = bomb_target_counts(
        game_state
    )
    if current_crate_count < 1 or current_opponent_count > 0:
        return candidate_indices

    (
        better_direction,
        better_distance,
        better_crate_count,
    ) = best_crate_bombing_path(
        game_state,
        max_search_distance=1,
    )

    if (
        better_direction is None
        or better_distance != 1
        or (
            better_crate_count - current_crate_count
            < CRATE_DEFERRAL_MIN_GAIN
        )
        or better_direction not in candidate_indices
    ):
        return candidate_indices

    simulated_state = _state_after_movement(
        game_state,
        int(better_direction),
    )
    simulated_crate_count, _ = bomb_target_counts(simulated_state)

    if (
        simulated_crate_count - current_crate_count
        < CRATE_DEFERRAL_MIN_GAIN
    ):
        return candidate_indices

    if not bomb_has_robust_escape_route(simulated_state):
        return candidate_indices

    return [int(better_direction)]


def _anti_stall_candidate_indices(
    self,
    game_state: dict,
    candidate_indices: list[int],
) -> list[int]:
    """Break repeated safe WAIT loops without weakening safety filters.

    This guard runs after the time-expanded and own-bomb filters.  It removes
    WAIT only when the agent has already waited at least three times at the
    same position, no known explosion reaches that position, and the approved
    candidate set contains the first move toward a visible coin or a safe
    crate-bombing tile.
    """
    wait_index = ACTIONS.index("WAIT")

    if wait_index not in candidate_indices:
        return candidate_indices

    if (
        getattr(self, "consecutive_safe_waits", 0)
        < ANTI_STALL_WAIT_LIMIT
    ):
        return candidate_indices

    current_position = tuple(game_state["self"][3])
    if getattr(self, "last_wait_position", None) != current_position:
        return candidate_indices

    if _current_position_has_known_danger(game_state):
        return candidate_indices

    progress_index = _progress_direction_index(game_state)
    if progress_index not in candidate_indices:
        return candidate_indices

    filtered_indices = [
        action_index
        for action_index in candidate_indices
        if action_index != wait_index
    ]

    return filtered_indices or candidate_indices


def _anti_oscillation_candidate_indices(
    self,
    game_state: dict,
    candidate_indices: list[int],
) -> list[int]:
    """Break a two-tile back-and-forth loop once crates and coins are gone.

    Like the WAIT guard above, this never picks a move.  It only removes the
    move back to the previous tile, and only when the last four positions
    alternate between two tiles, no known explosion reaches the current tile,
    and the approved candidates still contain a different first move toward
    the nearest opponent.
    """
    if np.any(game_state["field"] == 1):
        return candidate_indices

    coin_direction, _coin_distance = nearest_coin_path(game_state)
    if coin_direction is not None:
        return candidate_indices

    recent_positions = list(getattr(self, "recent_positions", []))
    if len(recent_positions) < ANTI_OSCILLATION_HISTORY:
        return candidate_indices

    first, second, third, current = recent_positions[-4:]
    if first != third or second != current or first == second:
        return candidate_indices

    if _current_position_has_known_danger(game_state):
        return candidate_indices

    return_index = None
    for direction_index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DIRECTIONS[action]
        if (current[0] + dx, current[1] + dy) == third:
            return_index = direction_index

    if return_index is None or return_index not in candidate_indices:
        return candidate_indices

    opponent_direction, _opponent_distance = nearest_opponent_path(game_state)
    if (
        opponent_direction is None
        or opponent_direction == return_index
        or opponent_direction not in candidate_indices
    ):
        return candidate_indices

    filtered_indices = [
        action_index
        for action_index in candidate_indices
        if action_index != return_index
    ]

    return filtered_indices or candidate_indices


def _record_anti_stall_choice(
    self,
    chosen_action: str,
    game_state: dict,
) -> None:
    """Remember consecutive safe WAIT selections at one position."""
    current_position = tuple(game_state["self"][3])

    if (
        chosen_action == "WAIT"
        and not _current_position_has_known_danger(game_state)
    ):
        if (
            getattr(self, "last_wait_position", None)
            == current_position
        ):
            self.consecutive_safe_waits = (
                getattr(self, "consecutive_safe_waits", 0) + 1
            )
        else:
            self.consecutive_safe_waits = 1

        self.last_wait_position = current_position
        return

    self.consecutive_safe_waits = 0
    self.last_wait_position = None


def _remember_own_bomb_if_selected(
    self,
    chosen_action: str,
    game_state: dict,
) -> None:
    """Remember the position where this agent chose to place a bomb."""
    if chosen_action == "BOMB":
        self.last_own_bomb_position = tuple(game_state["self"][3])


def _clear_inactive_own_bomb_memory(
    self,
    game_state: dict,
) -> None:
    """Forget own-bomb memory once the remembered bomb is no longer active."""
    remembered_position = getattr(
        self,
        "last_own_bomb_position",
        None,
    )

    if remembered_position is None:
        return

    active_bomb_positions = {
        tuple(position)
        for position, _timer in game_state.get("bombs", [])
    }

    if remembered_position not in active_bomb_positions:
        self.last_own_bomb_position = None


def _post_bomb_candidate_indices(
    self,
    game_state: dict,
    valid_indices: list[int],
) -> list[int]:
    """Apply minimal escape commitment while this agent's own bomb is active."""
    remembered_position = getattr(
        self,
        "last_own_bomb_position",
        None,
    )

    if remembered_position is None:
        return valid_indices

    bombs = [
        (tuple(position), int(timer))
        for position, timer in game_state.get("bombs", [])
    ]
    active_bomb_positions = {
        position
        for position, _timer in bombs
    }

    if remembered_position not in active_bomb_positions:
        self.last_own_bomb_position = None
        return valid_indices

    field = game_state["field"]
    current_position = tuple(game_state["self"][3])
    danger_times = earliest_danger_times(game_state)
    remembered_blast = set(
        blast_tiles(field, remembered_position)
    )

    movement_indices = [0, 1, 2, 3]
    wait_index = ACTIONS.index("WAIT")

    if current_position in remembered_blast:
        escape_candidates: set[int] = set()

        direction_index, distance = nearest_safe_path(game_state)

        if (
            direction_index is not None
            and distance is not None
            and distance > 0
            and direction_index in valid_indices
        ):
            escape_candidates.add(direction_index)

        for action_index in movement_indices:
            if action_index not in valid_indices:
                continue

            action = ACTIONS[action_index]
            dx, dy = DIRECTIONS[action]
            next_position = (
                current_position[0] + dx,
                current_position[1] + dy,
            )

            if np.isinf(danger_times[next_position]):
                escape_candidates.add(action_index)

        if escape_candidates:
            return sorted(escape_candidates)

        return valid_indices

    filtered_indices = []

    for action_index in valid_indices:
        if action_index in movement_indices:
            action = ACTIONS[action_index]
            dx, dy = DIRECTIONS[action]
            next_position = (
                current_position[0] + dx,
                current_position[1] + dy,
            )

            reenters_own_danger = (
                next_position in remembered_blast
                and not tile_is_safe_at_time(
                    danger_times[next_position],
                    arrival_time=1,
                )
            )

            if reenters_own_danger:
                continue

        filtered_indices.append(action_index)

    if (
        wait_index in filtered_indices
        and not tile_is_safe_at_time(
            danger_times[current_position],
            arrival_time=1,
        )
    ):
        filtered_indices.remove(wait_index)

    if filtered_indices:
        return filtered_indices

    return valid_indices


def _time_expanded_candidate_indices(
    game_state: dict,
    valid_indices: list[int],
) -> list[int]:
    """Remove actions that have no route through currently known danger.

    The learned feature vector and Q-values remain unchanged.  This function
    filters only the final candidate set and always falls back to the original
    non-empty set when the short-horizon model cannot find a survivor.
    """
    if not valid_indices:
        return valid_indices

    explosion_map = game_state.get("explosion_map")
    active_explosion = (
        explosion_map is not None
        and bool(np.any(np.asarray(explosion_map) > 0))
    )
    bomb_index = ACTIONS.index("BOMB")
    requires_check = (
        bool(game_state.get("bombs", []))
        or active_explosion
        or bomb_index in valid_indices
    )

    if not requires_check:
        return valid_indices

    survivable_indices = [
        action_index
        for action_index in valid_indices
        if (
            bomb_has_robust_escape_route(game_state)
            if ACTIONS[action_index] == "BOMB"
            else action_has_survival_route(
                game_state,
                ACTIONS[action_index],
            )
        )
    ]

    if survivable_indices:
        return survivable_indices

    non_bomb_indices = [
        action_index
        for action_index in valid_indices
        if ACTIONS[action_index] != "BOMB"
    ]

    if non_bomb_indices:
        return non_bomb_indices

    return valid_indices


def _trace_decision(
    game_state: dict,
    features: np.ndarray,
    valid_indices: list[int],
    q_values: np.ndarray | None,
    chosen_action: str,
    mode: str,
):
    """Print one diagnostic line when BOMBERMAN_TRACE=1."""
    if os.environ.get("BOMBERMAN_TRACE") != "1":
        return

    valid_actions = [ACTIONS[index] for index in valid_indices]

    if q_values is None:
        valid_q_values = {}
    else:
        valid_q_values = {
            ACTIONS[index]: round(float(q_values[index]), 3)
            for index in valid_indices
        }

    coin_direction, coin_distance = nearest_coin_path(game_state)
    escape_direction, escape_distance = nearest_safe_path(game_state)
    crate_direction, crate_distance = nearest_crate_bombing_path(
        game_state
    )
    (
        best_crate_direction,
        best_crate_distance,
        best_crate_count,
    ) = best_crate_bombing_path(
        game_state,
    )
    crate_targets, opponent_targets = bomb_target_counts(game_state)

    print(
        "[TRACE] "
        f"step={game_state.get('step')} "
        f"position={tuple(game_state['self'][3])} "
        f"bombs={game_state.get('bombs', [])} "
        f"coin={(coin_direction, coin_distance)} "
        f"crate={(crate_direction, crate_distance)} "
        f"best_crate={(best_crate_direction, best_crate_distance, best_crate_count)} "
        f"escape={(escape_direction, escape_distance)} "
        f"targets={(crate_targets, opponent_targets)} "
        f"valid={valid_actions} "
        f"q={valid_q_values} "
        f"mode={mode} "
        f"chosen={chosen_action}",
        flush=True,
    )


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

    # Features 39-43: where the nearest opponent is.  Filled in only when
    # nothing else is worth walking to, so this block never competes with the
    # coin or crate directions for the same move.
    field = game_state["field"]

    if not np.any(field == 1):
        coin_direction_now, _coin_distance_now = nearest_coin_path(game_state)

        if coin_direction_now is None:
            opponent_direction, opponent_distance = nearest_opponent_path(
                game_state
            )

            if opponent_direction is not None:
                features[39 + opponent_direction] = 1.0

            if opponent_distance not in (None, 0):
                maximum_distance = float(
                    field.shape[0] + field.shape[1]
                )
                features[43] = 1.0 - min(
                    opponent_distance / maximum_distance,
                    1.0,
                )

    # Visible coins keep priority. Features 0-31 retain exactly the same
    # meaning as in the stable Task 2 model.
    if not coins:
        crate_direction, crate_distance = nearest_crate_bombing_path(
            game_state
        )

        if crate_distance is not None:
            features[26] = 1.0

        if crate_direction is not None:
            features[27 + crate_direction] = 1.0

        if crate_distance not in (None, 0):
            field = game_state["field"]
            maximum_distance = field.shape[0] + field.shape[1]
            features[31] = min(
                crate_distance / maximum_distance,
                1.0,
            )

        # New additive features compare the current bomb yield with the
        # best safe bombing position no more than four movements away.
        # They are disabled while bombs are active so escape remains the
        # unambiguous immediate goal.
        if not game_state.get("bombs", []):
            (
                best_direction,
                best_distance,
                best_crate_count,
            ) = best_crate_bombing_path(game_state)

            current_crate_count, _opponent_count = bomb_target_counts(
                game_state
            )

            features[38] = min(
                best_crate_count / 4.0,
                1.0,
            )

            better_position_exists = (
                best_distance is not None
                and best_crate_count > current_crate_count
            )

            if better_position_exists:
                features[32] = 1.0

                if best_direction is not None:
                    features[33 + best_direction] = 1.0

                if best_distance not in (None, 0):
                    field = game_state["field"]
                    maximum_distance = (
                        field.shape[0] + field.shape[1]
                    )
                    features[37] = min(
                        best_distance / maximum_distance,
                        1.0,
                    )

    return features


def nearest_opponent_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """Use breadth-first search to find the nearest reachable opponent.

    The opponents cannot be placed in ``blocked_positions`` the way
    ``nearest_coin_path`` places them: they are the goal here, and blocking
    them would make every goal unreachable and the search would always return
    nothing.  Only live bombs block, and arriving on a tile an opponent stands
    on ends the search.
    """
    field = game_state["field"]
    start = tuple(game_state["self"][3])
    targets = {
        tuple(other_agent[3])
        for other_agent in game_state.get("others", [])
    }

    if not targets:
        return None, None

    blocked_positions = {
        position
        for position, _timer in game_state.get("bombs", [])
    }

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
