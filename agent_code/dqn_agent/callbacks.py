"""Callbacks for an opponent-aware warm-started Double DQN agent."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import random

import numpy as np
import torch

from .base_callbacks import (
    ACTIONS,
    FEATURE_DIM as BASE_FEATURE_DIM,
    _anti_stall_candidate_indices,
    _clear_inactive_own_bomb_memory,
    _crate_bomb_deferral_candidate_indices,
    _post_bomb_candidate_indices,
    _record_anti_stall_choice,
    _remember_own_bomb_if_selected,
    _time_expanded_candidate_indices,
    state_to_features as base_state_to_features,
    valid_action_indices,
)
from .game_utils import (
    DIRECTIONS,
    bomb_has_robust_escape_route,
    bomb_target_counts,
)
from .model import DQN, initialize_from_linear_q


HIDDEN_DIM = 128
ENDGAME_FEATURE_DIM = 6
PREVIOUS_FEATURE_DIM = BASE_FEATURE_DIM + ENDGAME_FEATURE_DIM
SAFE_ATTACK_FEATURE_DIM = 7
FEATURE_DIM = PREVIOUS_FEATURE_DIM + SAFE_ATTACK_FEATURE_DIM
CHECKPOINT_VERSION = 3
ENDGAME_EPSILON_START = 0.20
ENDGAME_EPSILON_MIN = 0.05
ENDGAME_WAIT_LIMIT = 2
DQN_MODEL_PATH = Path(__file__).resolve().parent / "dqn_model.pt"
ENDGAME_DQN_MODEL_PATH = (
    Path(__file__).resolve().parent / "dqn_model_trial8.pt"
)
LINEAR_TEACHER_PATH = Path(__file__).resolve().parent / "q_model.pkl"


def _select_device(logger) -> torch.device:
    requested = os.environ.get("BOMBERMAN_DQN_DEVICE", "cpu").lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested not in {"cpu", "mps", "cuda"}:
        logger.warning(
            f"Unknown BOMBERMAN_DQN_DEVICE={requested!r}; using CPU."
        )
    elif requested != "cpu":
        logger.warning(
            f"Requested DQN device {requested!r} is unavailable; using CPU."
        )
    return torch.device("cpu")


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_linear_teacher(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        saved_data = pickle.load(file)
    weights = np.asarray(saved_data["weights"], dtype=np.float32)
    expected = (len(ACTIONS), BASE_FEATURE_DIM)
    if weights.shape != expected:
        raise ValueError(
            f"Expected teacher weights with shape {expected}, "
            f"found {weights.shape}."
        )
    if not np.isfinite(weights).all():
        raise ValueError("Teacher weights contain non-finite values.")
    return weights


def _migrate_network_state(state: dict, old_feature_dim: int) -> dict:
    """Expand a v1/v2 input layer without changing its Q-values."""
    if old_feature_dim == FEATURE_DIM:
        return state
    if old_feature_dim not in {BASE_FEATURE_DIM, PREVIOUS_FEATURE_DIM}:
        raise ValueError(
            f"Cannot migrate DQN with {old_feature_dim} features."
        )

    migrated = {name: tensor.clone() for name, tensor in state.items()}
    input_weight_name = "network.0.weight"
    old_weight = migrated[input_weight_name]
    expected_shape = (HIDDEN_DIM, old_feature_dim)
    if tuple(old_weight.shape) != expected_shape:
        raise ValueError(
            f"Expected first-layer shape {expected_shape}, "
            f"found {tuple(old_weight.shape)}."
        )

    expanded_weight = old_weight.new_zeros(HIDDEN_DIM, FEATURE_DIM)
    expanded_weight[:, :old_feature_dim] = old_weight
    migrated[input_weight_name] = expanded_weight
    return migrated


def _load_endgame_policy(self) -> None:
    """Load the optional Trial-8 policy without modifying V3-75."""
    self.endgame_dqn_model_path = ENDGAME_DQN_MODEL_PATH
    self.endgame_policy_net = None
    if not self.endgame_dqn_model_path.is_file():
        self.logger.info(
            "Trial-8 endgame checkpoint not found; using V3-75 for all "
            "board phases."
        )
        return

    try:
        checkpoint = _load_torch_checkpoint(
            self.endgame_dqn_model_path,
            self.device,
        )
        checkpoint_version = int(checkpoint["checkpoint_version"])
        if checkpoint_version not in {1, 2, CHECKPOINT_VERSION}:
            raise ValueError("Unsupported endgame DQN checkpoint version.")
        saved_feature_dim = int(checkpoint["feature_dim"])
        if saved_feature_dim not in {
            BASE_FEATURE_DIM,
            PREVIOUS_FEATURE_DIM,
            FEATURE_DIM,
        }:
            raise ValueError("Endgame DQN feature dimension mismatch.")
        if list(checkpoint["actions"]) != ACTIONS:
            raise ValueError("Endgame DQN action order mismatch.")

        policy_state = _migrate_network_state(
            checkpoint["policy_state"],
            saved_feature_dim,
        )
        # Constructing a second network normally consumes PyTorch RNG values.
        # Preserve the RNG state so V8 has exactly the same tie-breaking
        # stream as V3-75 until their selected policies genuinely diverge.
        torch_rng_state = torch.random.get_rng_state()
        try:
            endgame_policy_net = DQN(
                FEATURE_DIM,
                len(ACTIONS),
                HIDDEN_DIM,
            ).to(self.device)
        finally:
            torch.random.set_rng_state(torch_rng_state)
        endgame_policy_net.load_state_dict(policy_state)
        endgame_policy_net.eval()
        for parameter in endgame_policy_net.parameters():
            parameter.requires_grad_(False)
        self.endgame_policy_net = endgame_policy_net
        self.logger.info(
            "Loaded Trial-8 as the zero-crate endgame policy."
        )
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        self.logger.warning(
            f"Could not load Trial-8 endgame checkpoint: {error}. "
            "Using V3-75 for all board phases."
        )


def _policy_net_for_state(self, game_state: dict):
    """Select V3-75 during crate play and Trial-8 after all crates."""
    crates_remain = bool(np.any(np.asarray(game_state["field"]) == 1))
    endgame_policy_net = getattr(self, "endgame_policy_net", None)
    if (
        not crates_remain
        and not getattr(self, "train", False)
        and endgame_policy_net is not None
    ):
        return endgame_policy_net, "trial8_endgame"
    return self.policy_net, "v3_75"


def nearest_opponent_path(
    game_state: dict,
) -> tuple[int | None, int | None]:
    """Find the shortest walkable path to a tile beside an opponent."""
    field = game_state["field"]
    width, height = field.shape
    start = tuple(game_state["self"][3])
    opponents = {
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    }
    if not opponents:
        return None, None

    if any(
        abs(start[0] - opponent[0]) + abs(start[1] - opponent[1]) == 1
        for opponent in opponents
    ):
        return None, 0

    blocked = {
        tuple(position)
        for position, _timer in game_state.get("bombs", [])
    }
    blocked.update(opponents)
    queue = [(start, None, 0)]
    queue_index = 0
    visited = {start}

    while queue_index < len(queue):
        position, first_direction, distance = queue[queue_index]
        queue_index += 1

        if any(
            abs(position[0] - opponent[0])
            + abs(position[1] - opponent[1])
            == 1
            for opponent in opponents
        ):
            return first_direction, distance

        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            next_position = (position[0] + dx, position[1] + dy)
            if not (
                0 <= next_position[0] < width
                and 0 <= next_position[1] < height
            ):
                continue
            if next_position in visited or next_position in blocked:
                continue
            if field[next_position] != 0:
                continue

            visited.add(next_position)
            queue.append(
                (
                    next_position,
                    direction_index
                    if first_direction is None
                    else first_direction,
                    distance + 1,
                )
            )

    return None, None


def endgame_opponent_pursuit_active(game_state: dict | None) -> bool:
    """Enable pursuit only on a safe board with no coin or crate goal."""
    if game_state is None:
        return False
    if game_state.get("coins", []) or np.any(game_state["field"] == 1):
        return False
    if not game_state.get("others", []):
        return False
    if game_state.get("bombs", []):
        return False

    explosion_map = game_state.get("explosion_map")
    if explosion_map is not None and np.any(np.asarray(explosion_map) > 0):
        return False

    _crate_count, opponent_count = bomb_target_counts(game_state)
    if opponent_count > 0:
        return False

    direction, distance = nearest_opponent_path(game_state)
    return direction is not None and distance not in (None, 0)


def _safe_endgame_board(game_state: dict | None) -> bool:
    """Return whether attack planning cannot affect normal objectives."""
    if game_state is None:
        return False
    if game_state.get("coins", []) or np.any(game_state["field"] == 1):
        return False
    if not game_state.get("others", []):
        return False
    if game_state.get("bombs", []):
        return False
    explosion_map = game_state.get("explosion_map")
    return not (
        explosion_map is not None
        and np.any(np.asarray(explosion_map) > 0)
    )


def _state_with_agent_position(
    game_state: dict,
    position: tuple[int, int],
) -> dict:
    simulated_state = dict(game_state)
    agent = list(game_state["self"])
    agent[3] = position
    simulated_state["self"] = tuple(agent)
    return simulated_state


def safe_opponent_bombing_path(
    game_state: dict | None,
) -> tuple[int | None, int | None, bool]:
    """Find a safe attack tile when the current direct bomb is unsafe.

    The mode is intentionally narrower than ordinary opponent pursuit: it is
    considered only after all coins and crates are gone and an opponent is
    already in the current blast line.  The returned boolean says that a
    robust opponent-targeting bomb can be placed on the current tile.
    """
    if not _safe_endgame_board(game_state):
        return None, None, False
    if not bool(game_state["self"][2]):
        return None, None, False

    _crate_count, opponent_count = bomb_target_counts(game_state)
    if opponent_count == 0:
        return None, None, False
    if bomb_has_robust_escape_route(game_state):
        return None, 0, True

    field = game_state["field"]
    width, height = field.shape
    start = tuple(game_state["self"][3])
    opponents = {
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    }
    queue = [(start, None, 0)]
    queue_index = 0
    visited = {start}

    while queue_index < len(queue):
        position, first_direction, distance = queue[queue_index]
        queue_index += 1

        if distance > 0:
            simulated_state = _state_with_agent_position(
                game_state,
                position,
            )
            _crates, simulated_opponents = bomb_target_counts(
                simulated_state
            )
            if (
                simulated_opponents > 0
                and bomb_has_robust_escape_route(simulated_state)
            ):
                return first_direction, distance, False

        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            next_position = (position[0] + dx, position[1] + dy)
            if not (
                0 <= next_position[0] < width
                and 0 <= next_position[1] < height
            ):
                continue
            if next_position in visited or next_position in opponents:
                continue
            if field[next_position] != 0:
                continue
            visited.add(next_position)
            queue.append(
                (
                    next_position,
                    direction_index
                    if first_direction is None
                    else first_direction,
                    distance + 1,
                )
            )

    return None, None, False


def endgame_safe_attack_active(game_state: dict | None) -> bool:
    """Return whether the safe-attack feature block is active."""
    _direction, distance, safe_bomb_now = safe_opponent_bombing_path(
        game_state
    )
    return safe_bomb_now or distance not in (None, 0)


def state_to_features(game_state: dict) -> np.ndarray | None:
    """Append pursuit and safe-attack features to the stable vector."""
    base_features = base_state_to_features(game_state)
    if base_features is None:
        return None

    features = np.zeros(FEATURE_DIM, dtype=np.float64)
    features[:BASE_FEATURE_DIM] = base_features
    if endgame_opponent_pursuit_active(game_state):
        direction, distance = nearest_opponent_path(game_state)
        features[39] = 1.0
        if direction is not None:
            features[40 + direction] = 1.0
        if distance not in (None, 0):
            field = game_state["field"]
            maximum_distance = field.shape[0] + field.shape[1]
            features[44] = min(distance / maximum_distance, 1.0)

    attack_direction, attack_distance, safe_bomb_now = (
        safe_opponent_bombing_path(game_state)
    )
    if safe_bomb_now or attack_distance not in (None, 0):
        features[45] = 1.0
        if attack_direction is not None:
            features[46 + attack_direction] = 1.0
        field = game_state["field"]
        maximum_distance = field.shape[0] + field.shape[1]
        if attack_distance not in (None, 0):
            features[50] = min(
                attack_distance / maximum_distance,
                1.0,
            )
        features[51] = float(safe_bomb_now)
    return features


def setup(self):
    """Load v3, migrate v1/v2, or warm-start from the linear model."""
    self.dqn_model_path = DQN_MODEL_PATH
    self.linear_teacher_path = LINEAR_TEACHER_PATH
    self.device = _select_device(self.logger)
    self.policy_net = DQN(FEATURE_DIM, len(ACTIONS), HIDDEN_DIM).to(self.device)
    self.target_net = DQN(FEATURE_DIM, len(ACTIONS), HIDDEN_DIM).to(self.device)

    self.epsilon = 0.20
    self.episodes_trained = 0
    self.environment_steps = 0
    self.optimizer_steps = 0
    self.warm_started = False
    self.pending_optimizer_state = None
    self.feature_migrated = False
    self.endgame_epsilon = ENDGAME_EPSILON_START
    self.active_policy_name = None

    self.last_own_bomb_position = None
    self.last_seen_round = None
    self.consecutive_safe_waits = 0
    self.last_wait_position = None
    self.recent_positions = []

    loaded_checkpoint = False
    if self.dqn_model_path.is_file():
        try:
            checkpoint = _load_torch_checkpoint(
                self.dqn_model_path,
                self.device,
            )
            checkpoint_version = int(checkpoint["checkpoint_version"])
            if checkpoint_version not in {1, 2, CHECKPOINT_VERSION}:
                raise ValueError("Unsupported DQN checkpoint version.")
            saved_feature_dim = int(checkpoint["feature_dim"])
            if saved_feature_dim not in {
                BASE_FEATURE_DIM,
                PREVIOUS_FEATURE_DIM,
                FEATURE_DIM,
            }:
                raise ValueError("DQN checkpoint feature dimension mismatch.")
            if list(checkpoint["actions"]) != ACTIONS:
                raise ValueError("DQN checkpoint action order mismatch.")

            policy_state = _migrate_network_state(
                checkpoint["policy_state"],
                saved_feature_dim,
            )
            target_state = _migrate_network_state(
                checkpoint["target_state"],
                saved_feature_dim,
            )
            self.policy_net.load_state_dict(policy_state)
            self.target_net.load_state_dict(target_state)
            self.epsilon = float(checkpoint.get("epsilon", 0.20))
            self.episodes_trained = int(
                checkpoint.get("episodes_trained", 0)
            )
            self.environment_steps = int(
                checkpoint.get("environment_steps", 0)
            )
            self.optimizer_steps = int(
                checkpoint.get("optimizer_steps", 0)
            )
            self.warm_started = bool(checkpoint.get("warm_started", False))
            self.feature_migrated = saved_feature_dim != FEATURE_DIM
            self.endgame_epsilon = float(
                checkpoint.get("endgame_epsilon", ENDGAME_EPSILON_START)
            )
            if self.feature_migrated:
                self.pending_optimizer_state = None
            else:
                self.pending_optimizer_state = checkpoint.get(
                    "optimizer_state"
                )
            loaded_checkpoint = True
            self.logger.info(
                "Loaded DQN checkpoint after "
                f"{self.episodes_trained} episodes."
            )
            if self.feature_migrated:
                self.logger.info(
                    f"Migrated DQN input from {saved_feature_dim} "
                    f"to {FEATURE_DIM} features; "
                    "existing Q-values are unchanged."
                )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
            self.logger.warning(
                f"Could not load DQN checkpoint: {error}. "
                "Trying the stable linear warm start."
            )

    if not loaded_checkpoint:
        try:
            teacher_weights = _load_linear_teacher(self.linear_teacher_path)
            expanded_teacher = np.zeros(
                (len(ACTIONS), FEATURE_DIM),
                dtype=np.float32,
            )
            expanded_teacher[:, :BASE_FEATURE_DIM] = teacher_weights
            initialize_from_linear_q(self.policy_net, expanded_teacher)
            self.warm_started = True
            self.logger.info(
                f"Warm-started the {FEATURE_DIM}-feature DQN exactly from "
                "the stable "
                "39-feature Q-learning model."
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            pickle.PickleError,
        ) as error:
            self.epsilon = 1.0
            self.logger.warning(
                f"Could not load the linear teacher: {error}. "
                "Using a randomly initialized DQN."
            )
        self.target_net.load_state_dict(self.policy_net.state_dict())

    self.policy_net.eval()
    self.target_net.eval()
    _load_endgame_policy(self)


def _movement_destination(
    game_state: dict,
    action_index: int,
) -> tuple[int, int] | None:
    if action_index >= 4:
        return None
    position = tuple(game_state["self"][3])
    dx, dy = DIRECTIONS[action_index]
    return position[0] + dx, position[1] + dy


def _endgame_cycle_candidate_indices(
    self,
    game_state: dict,
    candidate_indices: list[int],
) -> list[int]:
    """Reject the next repeated A-B-A-B step when pursuit can progress."""
    if not endgame_opponent_pursuit_active(game_state):
        return candidate_indices

    direction, _distance = nearest_opponent_path(game_state)
    if direction not in candidate_indices:
        return candidate_indices

    wait_index = ACTIONS.index("WAIT")
    filtered = list(candidate_indices)
    if (
        wait_index in filtered
        and getattr(self, "consecutive_safe_waits", 0)
        >= ENDGAME_WAIT_LIMIT
    ):
        filtered.remove(wait_index)

    history = getattr(self, "recent_positions", [])
    current_position = tuple(game_state["self"][3])
    if len(history) >= 2 and current_position == history[-2]:
        previous_position = history[-1]
        returning_indices = {
            action_index
            for action_index in filtered
            if _movement_destination(game_state, action_index)
            == previous_position
        }
        if direction not in returning_indices:
            without_return = [
                action_index
                for action_index in filtered
                if action_index not in returning_indices
            ]
            if direction in without_return:
                filtered = without_return

    return filtered or candidate_indices


def _record_endgame_position(self, game_state: dict) -> None:
    history = list(getattr(self, "recent_positions", []))
    history.append(tuple(game_state["self"][3]))
    self.recent_positions = history[-4:]


def candidate_action_indices(self, game_state: dict) -> list[int]:
    """Apply v2.2 safety filters plus the endgame cycle guard."""
    current_round = game_state.get("round")
    if getattr(self, "last_seen_round", None) != current_round:
        self.last_seen_round = current_round
        self.last_own_bomb_position = None
        self.consecutive_safe_waits = 0
        self.last_wait_position = None
        self.recent_positions = []

    _clear_inactive_own_bomb_memory(self, game_state)
    candidates = valid_action_indices(game_state)
    candidates = _time_expanded_candidate_indices(game_state, candidates)
    candidates = _post_bomb_candidate_indices(self, game_state, candidates)
    candidates = _crate_bomb_deferral_candidate_indices(
        game_state,
        candidates,
    )
    candidates = _anti_stall_candidate_indices(
        self,
        game_state,
        candidates,
    )
    return _endgame_cycle_candidate_indices(self, game_state, candidates)


def candidate_action_mask(self, game_state: dict) -> np.ndarray:
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[candidate_action_indices(self, game_state)] = True
    return mask


def act(self, game_state: dict) -> str:
    """Choose an epsilon-greedy action among approved candidates."""
    if game_state is None:
        return "WAIT"

    features = state_to_features(game_state)
    candidates = candidate_action_indices(self, game_state)
    endgame_adapter_active = (
        endgame_opponent_pursuit_active(game_state)
        or endgame_safe_attack_active(game_state)
    )
    exploration_rate = (
        self.endgame_epsilon if endgame_adapter_active else self.epsilon
    )
    active_policy_net, policy_name = _policy_net_for_state(
        self,
        game_state,
    )
    if getattr(self, "active_policy_name", None) != policy_name:
        self.logger.info(f"DQN phase policy changed to {policy_name}.")
        self.active_policy_name = policy_name

    if self.train and random.random() < exploration_rate:
        action_index = random.choice(candidates)
    else:
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        active_policy_net.eval()
        with torch.no_grad():
            q_values = active_policy_net(feature_tensor).squeeze(0)
        masked = torch.full_like(q_values, -torch.inf)
        masked[candidates] = q_values[candidates]
        best_value = torch.max(masked)
        best_indices = torch.nonzero(
            torch.isclose(masked, best_value),
            as_tuple=False,
        ).flatten()
        action_index = int(
            best_indices[
                torch.randint(len(best_indices), (1,), device=self.device)
            ].item()
        )

    chosen_action = ACTIONS[action_index]
    _remember_own_bomb_if_selected(self, chosen_action, game_state)
    _record_anti_stall_choice(self, chosen_action, game_state)
    _record_endgame_position(self, game_state)
    self.logger.debug(
        f"DQN selected {chosen_action} with epsilon={exploration_rate:.3f} "
        f"using {policy_name}."
    )
    return chosen_action
