"""Replay-based Double DQN training for the warm-started Bomberman agent."""

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
    reward_from_events,
)
from .callbacks import (
    ACTIONS,
    CHECKPOINT_VERSION,
    FEATURE_DIM,
    candidate_action_mask,
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


def setup_training(self):
    """Initialize replay memory and the optimizer."""
    self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
    self.optimizer = torch.optim.Adam(
        self.policy_net.parameters(),
        lr=LEARNING_RATE,
    )
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
        f"episodes={self.episodes_trained}, epsilon={self.epsilon:.3f}."
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
    save_checkpoint(self)

    mean_loss = (
        float(np.mean(self.round_losses))
        if self.round_losses
        else 0.0
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
        # Terminal masks are intentionally empty.  Give them a temporary
        # valid action so argmax remains defined; ``dones`` removes its value.
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
        self.policy_net.parameters(),
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
        "algorithm": "warm_started_masked_double_dqn_v1",
    }
    model_path = Path(self.dqn_model_path)
    temporary_path = model_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(model_path)

