from collections import namedtuple, deque
from typing import List

import csv
import pickle
from pathlib import Path
import numpy as np

import events as e
from .callbacks import (
    ACTIONS,
    state_summary,
    phi,
    action_values,
    MODEL_PATH,
    DISABLE_BOMB,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ALPHA = 0.01          # learning rate
GAMMA = 0.9           # discount factor

EPSILON_START = 1.0   # initial exploration rate
EPSILON_MIN = 0.1     # minimum exploration rate
EPSILON_DECAY = 0.999 # multiplicative decay per episode
EXPLORE_EPISODES = 100  # pure exploration before decay starts

# Number of recent positions to track for loop detection
REVISIT_WINDOW = 6

# ---------------------------------------------------------------------------
# Reward values
# ---------------------------------------------------------------------------

REWARD_COIN_COLLECTED         = 10.0
REWARD_CRATE_DESTROYED        = 1.5
REWARD_KILLED_SELF            = -20.0
REWARD_MOVED_TOWARD_TARGET    = 2.0
REWARD_MOVED_AWAY_FROM_TARGET = -2.0
REWARD_MOVED_TO_SAFETY        = 3.0
REWARD_MOVED_INTO_DANGER      = -3.0
REWARD_WAITED                 = -2.0
REWARD_INVALID_ACTION         = -2.0
REWARD_REVISITED_POSITION     = -2.0
REWARD_GOOD_BOMB              = 10.0
REWARD_USELESS_BOMB           = -1.0
REWARD_SUICIDE_BOMB           = -10.0

# Custom event names
MOVED_TOWARD_TARGET    = "MOVED_TOWARD_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
MOVED_TO_SAFETY        = "MOVED_TO_SAFETY"
MOVED_INTO_DANGER      = "MOVED_INTO_DANGER"
REVISITED_POSITION     = "REVISITED_POSITION"
GOOD_BOMB              = "GOOD_BOMB"
USELESS_BOMB           = "USELESS_BOMB"
SUICIDE_BOMB           = "SUICIDE_BOMB"

STATS_PATH = Path(__file__).resolve().parent / "stats.csv"

# ---------------------------------------------------------------------------
# Transition tuple (kept for reference, not used in TD)
# ---------------------------------------------------------------------------

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def setup_training(self):
    """
    Called once after setup() in callbacks.py.
    Initialises training-only state.
    """
    self.coins_collected = 0
    self.bombs_dropped = 0
    self.crates_destroyed = 0
    self.killed_self = 0
    self.invalid_actions = 0
    self.total_reward = 0.0

    # Recent positions for loop detection (stores (x, y) tuples)
    self.visited_positions: deque = deque(maxlen=REVISIT_WINDOW)

    # Write CSV header only if file does not exist yet
    if not STATS_PATH.is_file():
        with STATS_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "steps", "coins_collected",
                "bombs_dropped", "crates_destroyed",
                "killed_self", "invalid_actions",
                "total_reward", "epsilon",
            ])

    self.logger.info(
        f"Training setup complete. "
        f"episodes_trained={self.episodes_trained}, "
        f"epsilon={self.epsilon:.3f}"
    )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step (except the last).
    Fires custom events and performs a TD (Q-learning) weight update.
    """
    # Every custom event below is derived from these two summaries rather than
    # from its own BFS. Two reasons: it halves the BFS work per step, and it
    # makes it impossible for the reward and the features to disagree about
    # where the target is -- which is the bug that had the agent circling
    # (features looked at coins only, the reward at coins *and* crates).
    summary_old = state_summary(old_game_state)
    summary_new = state_summary(new_game_state)

    if summary_old is not None and summary_new is not None:

        # -------------------------------------------------------------------
        # Custom event: moved toward or away from nearest target
        # -------------------------------------------------------------------
        old_dist = summary_old["target_distance_raw"]
        new_dist = summary_new["target_distance_raw"]

        if old_dist is not None and new_dist is not None:
            if new_dist < old_dist:
                events.append(MOVED_TOWARD_TARGET)
            elif new_dist > old_dist:
                events.append(MOVED_AWAY_FROM_TARGET)

        # -------------------------------------------------------------------
        # Custom event: moved to safety or into danger
        # -------------------------------------------------------------------
        if not DISABLE_BOMB:
            old_danger = summary_old["danger"]
            new_danger = summary_new["danger"]

            # danger_countdown returns 0.0 for safe and grows as the threat
            # becomes more urgent, the opposite of the old raw-timer version.
            # These two comparisons are flipped to match; the events fire in
            # exactly the same situations as before.
            if new_danger < old_danger:
                events.append(MOVED_TO_SAFETY)
            elif new_danger > old_danger:
                events.append(MOVED_INTO_DANGER)

        # -------------------------------------------------------------------
        # Custom event: bomb quality assessment
        # Evaluated at the moment of dropping (self_action == "BOMB"),
        # using old_game_state (before the bomb was placed).
        # -------------------------------------------------------------------
        if not DISABLE_BOMB and self_action == "BOMB":
            # summary["trap"] is float(not escapable_if_bombed) and
            # summary["crate_efficiency"] is blast_crate_count / MAX_CRATES, both
            # taken from old_game_state, so these are the same three cases as
            # before -- just without recomputing the escape BFS.
            if summary_old["trap"]:
                events.append(SUICIDE_BOMB)
            elif summary_old["crate_efficiency"] > 0.0:
                events.append(GOOD_BOMB)
            else:
                events.append(USELESS_BOMB)

        # -------------------------------------------------------------------
        # Custom event: revisited position (loop detection)
        # -------------------------------------------------------------------
        new_pos = new_game_state["self"][3]
        if new_pos in self.visited_positions:
            events.append(REVISITED_POSITION)
        self.visited_positions.append(new_pos)

    # -----------------------------------------------------------------------
    # Compute reward and TD update
    # -----------------------------------------------------------------------
    reward = reward_from_events(self, events)

    if summary_old is not None and self_action is not None:
        action_index = ACTIONS.index(self_action)
        phi_old = phi(summary_old, action_index)
        q_current = self.model @ phi_old

        if summary_new is not None:
            # Q-learning: bootstrap with the max Q of the next state, taken over
            # the actions that are actually available there.
            #
            # Taking the max over all six actions instead lets the target borrow
            # the Q-value of a move into a wall. act() masks those out, so they
            # are never executed and never corrected against reality; their
            # weights come from other states where they were legal. Measured on
            # this agent, that unreachable action was the argmax on 99.9% of
            # steps and inflated the target by ~5.6 on average, which cancelled
            # the per-step penalty for standing still.
            valid_next = summary_new["valid_indices"]
            q_next = np.max(action_values(self.model, summary_new)[valid_next])
            td_target = reward + GAMMA * q_next
        else:
            td_target = reward

        # w <- w + alpha * (td_target - Q(s,a)) * phi(s,a)
        #
        # One shared vector now, so this same update is what teaches every
        # direction: a successful escape while moving LEFT lands in the exact
        # slot ("escape route is ahead") that scoring UP will read next time.
        self.model += ALPHA * (td_target - q_current) * phi_old

    # Track per-episode statistics
    if e.COIN_COLLECTED in events:
        self.coins_collected += 1
    if e.BOMB_DROPPED in events:
        self.bombs_dropped += 1
    if e.CRATE_DESTROYED in events:
        self.crates_destroyed += 1
    if e.KILLED_SELF in events:
        self.killed_self += 1
    if e.INVALID_ACTION in events:
        self.invalid_actions += 1
    self.total_reward += reward


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """
    Called once at the end of each episode.
    Performs final TD update, saves model, and logs stats.
    """
    # Final step TD update (terminal state: no next state)
    reward = reward_from_events(self, events)
    summary = state_summary(last_game_state)

    if summary is not None and last_action is not None:
        action_index = ACTIONS.index(last_action)
        phi_last = phi(summary, action_index)
        q_current = self.model @ phi_last
        # Terminal state: no next-state Q-value
        self.model += ALPHA * (reward - q_current) * phi_last

    self.logger.info(
        f"Episode {self.episodes_trained + 1} complete. "
        f"epsilon: {self.epsilon:.4f}"
    )

    # -----------------------------------------------------------------------
    # Epsilon decay (starts after EXPLORE_EPISODES)
    # -----------------------------------------------------------------------
    self.episodes_trained += 1

    if self.episodes_trained > EXPLORE_EPISODES:
        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY,
        )

    # Reset visited positions for next episode
    self.visited_positions.clear()

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------
    saved_data = {
        "weights": self.model,
        "epsilon": self.epsilon,
        "episodes_trained": self.episodes_trained,
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(saved_data, f)

    self.logger.info(f"Model saved to {MODEL_PATH}.")

    # -----------------------------------------------------------------------
    # Write stats to CSV
    # -----------------------------------------------------------------------
    if e.COIN_COLLECTED in events:
        self.coins_collected += 1
    if e.BOMB_DROPPED in events:
        self.bombs_dropped += 1
    if e.CRATE_DESTROYED in events:
        self.crates_destroyed += 1
    if e.KILLED_SELF in events:
        self.killed_self += 1
    if e.INVALID_ACTION in events:
        self.invalid_actions += 1
    self.total_reward += reward

    with STATS_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            self.episodes_trained,
            last_game_state["step"],
            self.coins_collected,
            self.bombs_dropped,
            self.crates_destroyed,
            self.killed_self,
            self.invalid_actions,
            round(self.total_reward, 3),
            round(self.epsilon, 4),
        ])

    # Reset per-episode counters
    self.coins_collected = 0
    self.bombs_dropped = 0
    self.crates_destroyed = 0
    self.killed_self = 0
    self.invalid_actions = 0
    self.total_reward = 0.0


def reward_from_events(self, events: List[str]) -> float:
    """
    Maps game events to scalar rewards.
    """
    reward_map = {
        e.COIN_COLLECTED:          REWARD_COIN_COLLECTED,
        e.CRATE_DESTROYED:         REWARD_CRATE_DESTROYED,
        e.KILLED_SELF:             REWARD_KILLED_SELF,
        e.WAITED:                  REWARD_WAITED,
        e.INVALID_ACTION:          REWARD_INVALID_ACTION,
        MOVED_TOWARD_TARGET:       REWARD_MOVED_TOWARD_TARGET,
        MOVED_AWAY_FROM_TARGET:    REWARD_MOVED_AWAY_FROM_TARGET,
        MOVED_TO_SAFETY:           REWARD_MOVED_TO_SAFETY,
        MOVED_INTO_DANGER:         REWARD_MOVED_INTO_DANGER,
        REVISITED_POSITION:        REWARD_REVISITED_POSITION,
        GOOD_BOMB:                 REWARD_GOOD_BOMB,
        USELESS_BOMB:              REWARD_USELESS_BOMB,
        SUICIDE_BOMB:              REWARD_SUICIDE_BOMB,
    }

    total = sum(reward_map.get(event, 0.0) for event in events)

    self.logger.debug(
        f"Reward {total:.3f} from events: {', '.join(events)}"
    )

    return total
