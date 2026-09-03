"""Callbacks for a warm-started Double DQN Bomberman agent."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import random

import numpy as np
import torch

from ..q_learning_agent.callbacks import (
    ACTIONS,
    FEATURE_DIM,
    _anti_stall_candidate_indices,
    _clear_inactive_own_bomb_memory,
    _crate_bomb_deferral_candidate_indices,
    _post_bomb_candidate_indices,
    _record_anti_stall_choice,
    _remember_own_bomb_if_selected,
    _time_expanded_candidate_indices,
    state_to_features,
    valid_action_indices,
)
from .model import DQN, initialize_from_linear_q


HIDDEN_DIM = 128
CHECKPOINT_VERSION = 1
DQN_MODEL_PATH = Path(__file__).resolve().parent / "dqn_model.pt"
LINEAR_TEACHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "q_learning_agent"
    / "q_model.pkl"
)


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
        # Compatibility with PyTorch versions before ``weights_only``.
        return torch.load(path, map_location=device)


def _load_linear_teacher(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        saved_data = pickle.load(file)
    weights = np.asarray(saved_data["weights"], dtype=np.float32)
    expected = (len(ACTIONS), FEATURE_DIM)
    if weights.shape != expected:
        raise ValueError(
            f"Expected teacher weights with shape {expected}, "
            f"found {weights.shape}."
        )
    if not np.isfinite(weights).all():
        raise ValueError("Teacher weights contain non-finite values.")
    return weights


def setup(self):
    """Load a DQN checkpoint or exactly warm-start from the v2.2 model."""
    self.dqn_model_path = DQN_MODEL_PATH
    self.linear_teacher_path = LINEAR_TEACHER_PATH
    self.device = _select_device(self.logger)
    self.policy_net = DQN(FEATURE_DIM, len(ACTIONS), HIDDEN_DIM).to(
        self.device
    )
    self.target_net = DQN(FEATURE_DIM, len(ACTIONS), HIDDEN_DIM).to(
        self.device
    )

    self.epsilon = 0.20
    self.episodes_trained = 0
    self.environment_steps = 0
    self.optimizer_steps = 0
    self.warm_started = False
    self.pending_optimizer_state = None

    self.last_own_bomb_position = None
    self.last_seen_round = None
    self.consecutive_safe_waits = 0
    self.last_wait_position = None

    loaded_checkpoint = False
    if self.dqn_model_path.is_file():
        try:
            checkpoint = _load_torch_checkpoint(
                self.dqn_model_path,
                self.device,
            )
            if int(checkpoint["checkpoint_version"]) != CHECKPOINT_VERSION:
                raise ValueError("Unsupported DQN checkpoint version.")
            if int(checkpoint["feature_dim"]) != FEATURE_DIM:
                raise ValueError("DQN checkpoint feature dimension mismatch.")
            if list(checkpoint["actions"]) != ACTIONS:
                raise ValueError("DQN checkpoint action order mismatch.")

            self.policy_net.load_state_dict(checkpoint["policy_state"])
            self.target_net.load_state_dict(checkpoint["target_state"])
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
            self.warm_started = bool(
                checkpoint.get("warm_started", False)
            )
            self.pending_optimizer_state = checkpoint.get(
                "optimizer_state"
            )
            loaded_checkpoint = True
            self.logger.info(
                "Loaded DQN checkpoint after "
                f"{self.episodes_trained} episodes."
            )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
            self.logger.warning(
                f"Could not load DQN checkpoint: {error}. "
                "Trying the stable linear warm start."
            )

    if not loaded_checkpoint:
        try:
            teacher_weights = _load_linear_teacher(
                self.linear_teacher_path
            )
            initialize_from_linear_q(self.policy_net, teacher_weights)
            self.warm_started = True
            self.logger.info(
                "Warm-started DQN exactly from the stable 39-feature "
                "Q-learning model."
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


def candidate_action_indices(self, game_state: dict) -> list[int]:
    """Apply the exact v2.2 safety and efficiency candidate filters."""
    current_round = game_state.get("round")
    if getattr(self, "last_seen_round", None) != current_round:
        self.last_seen_round = current_round
        self.last_own_bomb_position = None
        self.consecutive_safe_waits = 0
        self.last_wait_position = None

    _clear_inactive_own_bomb_memory(self, game_state)
    candidates = valid_action_indices(game_state)
    candidates = _time_expanded_candidate_indices(game_state, candidates)
    candidates = _post_bomb_candidate_indices(
        self,
        game_state,
        candidates,
    )
    candidates = _crate_bomb_deferral_candidate_indices(
        game_state,
        candidates,
    )
    return _anti_stall_candidate_indices(
        self,
        game_state,
        candidates,
    )


def candidate_action_mask(self, game_state: dict) -> np.ndarray:
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[candidate_action_indices(self, game_state)] = True
    return mask


def act(self, game_state: dict) -> str:
    """Choose an epsilon-greedy action among v2.2-approved candidates."""
    if game_state is None:
        return "WAIT"

    features = state_to_features(game_state)
    candidates = candidate_action_indices(self, game_state)

    if self.train and random.random() < self.epsilon:
        action_index = random.choice(candidates)
    else:
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(feature_tensor).squeeze(0)
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
    self.logger.debug(
        f"DQN selected {chosen_action} with epsilon={self.epsilon:.3f}."
    )
    return chosen_action

