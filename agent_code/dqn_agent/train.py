"""Adapter-only Double DQN training for opponent-aware endgames."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn

from ..q_learning_agent.train import (
    add_bomb_placement_event,
    add_coin_distance_event,
    add_crate_efficiency_navigation_event,
    add_crate_navigation_event,
    add_escape_events,
    add_waiting_event,
    reward_from_events as base_reward_from_events,
)
from .callbacks import (
    ACTIONS,
    CHECKPOINT_VERSION,
    ENDGAME_EPSILON_MIN,
    FEATURE_DIM,
    PREVIOUS_FEATURE_DIM,
    _state_with_agent_position,
    candidate_action_mask,
    endgame_opponent_pursuit_active,
    endgame_safe_attack_active,
    nearest_opponent_path,
    safe_opponent_bombing_path,
    state_to_features,
)
from .replay_buffer import ReplayBuffer


GAMMA = 0.90
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000
REPLAY_WARMUP = 1_000
TARGET_SYNC_INTERVAL = 1_000
GRADIENT_CLIP_NORM = 10.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

MOVED_TOWARD_OPPONENT = "MOVED_TOWARD_OPPONENT"
MOVED_AWAY_FROM_OPPONENT = "MOVED_AWAY_FROM_OPPONENT"
WAITED_DURING_ENDGAME = "WAITED_DURING_ENDGAME"
MOVED_TOWARD_SAFE_ATTACK_POSITION = (
    "MOVED_TOWARD_SAFE_ATTACK_POSITION"
)
MOVED_AWAY_FROM_SAFE_ATTACK_POSITION = (
    "MOVED_AWAY_FROM_SAFE_ATTACK_POSITION"
)
WAITED_DURING_SAFE_ATTACK = "WAITED_DURING_SAFE_ATTACK"

REWARD_MOVED_TOWARD_OPPONENT = 0.5
REWARD_MOVED_AWAY_FROM_OPPONENT = -0.5
REWARD_WAITED_DURING_ENDGAME = -1.0
REWARD_MOVED_TOWARD_SAFE_ATTACK_POSITION = 0.5
REWARD_MOVED_AWAY_FROM_SAFE_ATTACK_POSITION = -0.5
REWARD_WAITED_DURING_SAFE_ATTACK = -1.0


def setup_training(self):
    """Train only the new safe-attack adapter input connections."""
    self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    for parameter in self.policy_net.parameters():
        parameter.requires_grad_(False)
    adapter_weight = self.policy_net.network[0].weight
    adapter_weight.requires_grad_(True)

    def keep_only_safe_attack_feature_gradients(gradient):
        masked_gradient = gradient.clone()
        masked_gradient[:, :PREVIOUS_FEATURE_DIM] = 0.0
        return masked_gradient

    self.safe_attack_gradient_hook = adapter_weight.register_hook(
        keep_only_safe_attack_feature_gradients
    )
    self.optimizer = torch.optim.Adam([adapter_weight], lr=LEARNING_RATE)
    if self.pending_optimizer_state is not None:
        try:
            self.optimizer.load_state_dict(self.pending_optimizer_state)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self.logger.warning(
                f"Could not restore optimizer state: {error}."
            )
    self.pending_optimizer_state = None
    self.round_reward = 0.0
    self.round_losses = []
    self.policy_net.train()
    self.target_net.eval()
    self.logger.info(
        "Double DQN training initialized: "
        f"episodes={self.episodes_trained}, epsilon={self.epsilon:.3f}, "
        f"endgame_epsilon={self.endgame_epsilon:.3f}, "
        "trainable=safe_attack_input_adapter."
    )


def add_endgame_opponent_navigation_event(
    old_game_state: dict,
    new_game_state: dict,
    events: List[str],
) -> None:
    """Reward only the progress caused by our movement toward an opponent."""
    if old_game_state is None or new_game_state is None:
        return
    if not endgame_opponent_pursuit_active(old_game_state):
        return

    _old_direction, old_distance = nearest_opponent_path(old_game_state)
    if old_distance is None:
        return

    comparison_state = dict(old_game_state)
    comparison_agent = list(old_game_state["self"])
    comparison_agent[3] = tuple(new_game_state["self"][3])
    comparison_state["self"] = tuple(comparison_agent)
    _new_direction, new_distance = nearest_opponent_path(comparison_state)
    if new_distance is None:
        return
    if new_distance < old_distance:
        events.append(MOVED_TOWARD_OPPONENT)
    elif new_distance > old_distance:
        events.append(MOVED_AWAY_FROM_OPPONENT)


def add_endgame_waiting_event(
    game_state: dict,
    action: str,
    events: List[str],
) -> None:
    if action == "WAIT" and endgame_opponent_pursuit_active(game_state):
        events.append(WAITED_DURING_ENDGAME)


def add_safe_attack_navigation_event(
    old_game_state: dict,
    new_game_state: dict,
    events: List[str],
) -> None:
    """Shape progress toward a robust opponent-targeting bomb tile."""
    if old_game_state is None or new_game_state is None:
        return

    _old_direction, old_distance, safe_bomb_now = (
        safe_opponent_bombing_path(old_game_state)
    )
    if safe_bomb_now or old_distance in (None, 0):
        return

    comparison_state = _state_with_agent_position(
        old_game_state,
        tuple(new_game_state["self"][3]),
    )
    _new_direction, new_distance, new_safe_bomb_now = (
        safe_opponent_bombing_path(comparison_state)
    )
    if new_safe_bomb_now or (
        new_distance is not None and new_distance < old_distance
    ):
        events.append(MOVED_TOWARD_SAFE_ATTACK_POSITION)
    elif new_distance is None or new_distance > old_distance:
        events.append(MOVED_AWAY_FROM_SAFE_ATTACK_POSITION)


def add_safe_attack_waiting_event(
    game_state: dict,
    action: str,
    events: List[str],
) -> None:
    """Discourage stalling only while the safe-attack adapter is active."""
    if action == "WAIT" and endgame_safe_attack_active(game_state):
        events.append(WAITED_DURING_SAFE_ATTACK)


def reward_from_events(self, events: List[str]) -> float:
    """Add endgame shaping without modifying v2.2's reward function."""
    reward = base_reward_from_events(self, events)
    endgame_rewards = {
        MOVED_TOWARD_OPPONENT: REWARD_MOVED_TOWARD_OPPONENT,
        MOVED_AWAY_FROM_OPPONENT: REWARD_MOVED_AWAY_FROM_OPPONENT,
        WAITED_DURING_ENDGAME: REWARD_WAITED_DURING_ENDGAME,
        MOVED_TOWARD_SAFE_ATTACK_POSITION: (
            REWARD_MOVED_TOWARD_SAFE_ATTACK_POSITION
        ),
        MOVED_AWAY_FROM_SAFE_ATTACK_POSITION: (
            REWARD_MOVED_AWAY_FROM_SAFE_ATTACK_POSITION
        ),
        WAITED_DURING_SAFE_ATTACK: REWARD_WAITED_DURING_SAFE_ATTACK,
    }
    return float(
        reward
        + sum(endgame_rewards.get(event, 0.0) for event in events)
    )


def _features_or_zeros(game_state: dict | None) -> np.ndarray:
    if game_state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    return np.asarray(state_to_features(game_state), dtype=np.float32)


def _store_transition(
    self,
    old_game_state: dict,
    action: str,
    reward: float,
    new_game_state: dict | None,
) -> None:
    if old_game_state is None or action not in ACTIONS:
        return

    done = new_game_state is None
    if done:
        next_mask = np.zeros(len(ACTIONS), dtype=bool)
    else:
        next_mask = candidate_action_mask(self, new_game_state)
    self.replay_buffer.add(
        state=_features_or_zeros(old_game_state),
        action=ACTIONS.index(action),
        reward=reward,
        next_state=_features_or_zeros(new_game_state),
        next_action_mask=next_mask,
        done=done,
    )
    self.environment_steps += 1


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Shape a transition, add it to replay, and take one DQN step."""
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
    add_endgame_opponent_navigation_event(
        old_game_state,
        new_game_state,
        events,
    )
    add_endgame_waiting_event(old_game_state, self_action, events)
    add_safe_attack_navigation_event(
        old_game_state,
        new_game_state,
        events,
    )
    add_safe_attack_waiting_event(
        old_game_state,
        self_action,
        events,
    )
    reward = reward_from_events(self, events)
    _store_transition(
        self,
        old_game_state,
        self_action,
        reward,
        new_game_state,
    )
    loss = optimize_model(self)
    self.round_reward += reward
    if loss is not None:
        self.round_losses.append(loss)


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """Store the terminal transition and save an atomic checkpoint."""
    add_bomb_placement_event(last_game_state, last_action, events)
    add_endgame_waiting_event(last_game_state, last_action, events)
    add_safe_attack_waiting_event(last_game_state, last_action, events)
    reward = reward_from_events(self, events)
    _store_transition(
        self,
        last_game_state,
        last_action,
        reward,
        None,
    )
    loss = optimize_model(self)
    self.round_reward += reward
    if loss is not None:
        self.round_losses.append(loss)

    self.episodes_trained += 1
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.endgame_epsilon = max(
        ENDGAME_EPSILON_MIN,
        self.endgame_epsilon * EPSILON_DECAY,
    )
    save_checkpoint(self)

    mean_loss = (
        float(np.mean(self.round_losses)) if self.round_losses else 0.0
    )
    self.logger.info(
        f"DQN episode {self.episodes_trained}: "
        f"reward={self.round_reward:.2f}, epsilon={self.epsilon:.3f}, "
        f"mean_loss={mean_loss:.4f}."
    )
    self.round_reward = 0.0
    self.round_losses = []


def double_dqn_targets(
    policy_net: nn.Module,
    target_net: nn.Module,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    next_action_masks: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = GAMMA,
) -> torch.Tensor:
    """Compute masked Double DQN targets without tracking gradients."""
    with torch.no_grad():
        policy_values = policy_net(next_states)
        masked_policy_values = policy_values.masked_fill(
            ~next_action_masks,
            -torch.inf,
        )
        empty_masks = ~next_action_masks.any(dim=1)
        if empty_masks.any():
            masked_policy_values[empty_masks, 0] = 0.0
        selected_actions = masked_policy_values.argmax(dim=1, keepdim=True)
        target_values = target_net(next_states).gather(
            1,
            selected_actions,
        ).squeeze(1)
        return rewards + gamma * (~dones).float() * target_values


def optimize_model(self) -> float | None:
    """Run one replay update and periodically synchronize the target net."""
    warmup = int(getattr(self, "replay_warmup", REPLAY_WARMUP))
    batch_size = int(getattr(self, "batch_size", BATCH_SIZE))
    if len(self.replay_buffer) < max(warmup, batch_size):
        return None

    transitions = self.replay_buffer.sample(batch_size)
    device = self.device
    states = torch.as_tensor(
        np.stack([item.state for item in transitions]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        [item.action for item in transitions],
        dtype=torch.long,
        device=device,
    )
    rewards = torch.as_tensor(
        [item.reward for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    next_states = torch.as_tensor(
        np.stack([item.next_state for item in transitions]),
        dtype=torch.float32,
        device=device,
    )
    next_masks = torch.as_tensor(
        np.stack([item.next_action_mask for item in transitions]),
        dtype=torch.bool,
        device=device,
    )
    dones = torch.as_tensor(
        [item.done for item in transitions],
        dtype=torch.bool,
        device=device,
    )

    self.policy_net.train()
    current_values = self.policy_net(states).gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)
    targets = double_dqn_targets(
        self.policy_net,
        self.target_net,
        rewards,
        next_states,
        next_masks,
        dones,
    )
    loss = nn.functional.smooth_l1_loss(current_values, targets)
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        [self.policy_net.network[0].weight],
        GRADIENT_CLIP_NORM,
    )
    self.optimizer.step()
    self.optimizer_steps += 1

    sync_interval = int(
        getattr(self, "target_sync_interval", TARGET_SYNC_INTERVAL)
    )
    if self.optimizer_steps % sync_interval == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
    return float(loss.item())


def save_checkpoint(self) -> None:
    """Atomically save network, optimizer, and training counters."""
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": ACTIONS,
        "hidden_dim": self.policy_net.hidden_dim,
        "policy_state": self.policy_net.state_dict(),
        "target_state": self.target_net.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "epsilon": float(self.epsilon),
        "episodes_trained": int(self.episodes_trained),
        "environment_steps": int(self.environment_steps),
        "optimizer_steps": int(self.optimizer_steps),
        "warm_started": bool(self.warm_started),
        "endgame_epsilon": float(self.endgame_epsilon),
        "training_scope": "safe_attack_input_adapter_only",
        "algorithm": "safe_attack_staging_masked_double_dqn_v3",
    }
    model_path = Path(self.dqn_model_path)
    temporary_path = model_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(model_path)
