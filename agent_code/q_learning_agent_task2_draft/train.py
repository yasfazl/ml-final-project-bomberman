"""
Training code for the Task 2 Q-learning agent -- starting point.

Two of the three fixes over the first Task 2 attempt live here.

Fix 2: temporal-difference Q-learning instead of Monte Carlo returns.

    target = r + gamma * max_{a'} Q(s', a')
    w_a   <- w_a + alpha * mean over the batch of (target - Q(s, a)) * phi(s)

    The Monte Carlo return of an episode is the discounted sum of up to 400
    shaped rewards.  Its variance is far larger than anything eighteen binary
    features can explain, and with one weight vector per action that variance
    is absorbed by the per-action offsets w_a[0].  The greedy policy then
    follows those offsets rather than the direction features, which is what
    makes the agent stand still or pace back and forth.  A bootstrapped target
    is bounded and much quieter.

Fix 3: reward shaping based on the same BFS distance the features use.

    The first version compared Manhattan distances while the features used BFS
    distances.  Behind a wall those two disagree, so the reward could pull the
    agent in the direction the features called wrong.  Both now come from
    nearest_target() in callbacks.py.

Ideas that are deliberately not implemented yet:
  * rewards for escaping a blast / staying in one;
  * a penalty for revisiting the same few tiles;
  * n-step returns, prioritised replay, a target network;
  * tuning any of the constants below -- they are reasonable, not optimal.
"""

from collections import namedtuple
from typing import List

import pickle
import random

import numpy as np

import events as e
from .callbacks import (
    ACTIONS,
    MODEL_PATH,
    crate_count,
    state_to_features,
    target_distance,
    valid_action_indices,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ALPHA = 0.02          # learning rate
GAMMA = 0.9           # discount factor

BUFFER_CAPACITY = 50_000
BATCH_SIZE = 128
MIN_BUFFER_SIZE = 1_000     # no updates before the buffer holds this many
UPDATES_AT_ROUND_END = 10   # extra replay updates once the round is over

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 800  # linear decay over this many episodes

LOG_EVERY = 25        # episodes between summary log lines

# ---------------------------------------------------------------------------
# Custom events
# ---------------------------------------------------------------------------

MOVED_TOWARD_TARGET = "MOVED_TOWARD_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
GOOD_BOMB = "GOOD_BOMB"
USELESS_BOMB = "USELESS_BOMB"

# Events that only ever show up in the end_of_round delivery.
TERMINAL_EVENTS = {e.SURVIVED_ROUND, e.KILLED_SELF, e.GOT_KILLED}

# ---------------------------------------------------------------------------
# Reward values
# ---------------------------------------------------------------------------

REWARDS = {
    e.COIN_COLLECTED:        25.0,
    e.CRATE_DESTROYED:        3.0,
    e.COIN_FOUND:             2.0,
    e.KILLED_SELF:          -50.0,
    e.GOT_KILLED:           -50.0,
    e.SURVIVED_ROUND:         5.0,
    e.INVALID_ACTION:        -5.0,
    e.WAITED:                -1.0,

    MOVED_TOWARD_TARGET:      1.5,
    MOVED_AWAY_FROM_TARGET:  -2.0,

    # CRATE_DESTROYED only arrives four steps after the bomb was dropped, so
    # the drop itself is judged straight away as well.
    GOOD_BOMB:                3.0,
    USELESS_BOMB:            -3.0,
}

# ---------------------------------------------------------------------------
# Transition tuple
# ---------------------------------------------------------------------------

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "next_valid")
)


class ReplayBuffer:
    """
    Fixed-size ring buffer of transitions.

    Experience replay is what makes the Q-learning updates behave: consecutive
    steps of one episode are highly correlated, and fitting a linear model on
    correlated samples makes the weights drift with whatever the agent is
    doing right now.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: list[Transition] = []
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, transition: Transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.cursor] = transition
            self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, min(batch_size, len(self.data)))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def setup_training(self):
    """
    Called once after setup() in callbacks.py.
    Initialises training-only state.
    """
    self.buffer = ReplayBuffer(BUFFER_CAPACITY)
    self.training_history = []

    # Step number of the transition stored last; see end_of_round.
    self.last_recorded_step = -1

    # Per-episode counters, reset in end_of_round.
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_crates = 0
    self.episode_steps = 0

    self.epsilon = epsilon_for(self.episodes_trained)

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
    Stores one transition and performs one replay update.
    """
    if old_game_state is None or self_action is None:
        return

    add_custom_events(old_game_state, new_game_state, events)

    reward = reward_from_events(self, events)

    next_valid = np.zeros(len(ACTIONS), dtype=bool)
    if new_game_state is not None:
        next_valid[valid_action_indices(new_game_state)] = True

    self.buffer.push(
        Transition(
            state_to_features(old_game_state),
            ACTIONS.index(self_action),
            reward,
            state_to_features(new_game_state),
            next_valid,
        )
    )

    record_step(self, reward, events)
    self.last_recorded_step = old_game_state["step"]

    optimise(self)


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """
    Called once at the end of each episode.  Stores the terminal transition,
    runs a few more replay updates, decays epsilon and saves the model.
    """
    if last_game_state is not None and last_action is not None:
        # BombeRLeWorld.do_step calls send_game_events() and then end_round(),
        # and both hand over the very same event list.  Whenever the round ends
        # with the agent still alive, the final step therefore arrives twice:
        # once here and once in game_events_occurred.  Counting it twice is
        # what turns "50 of 50 coins" into "51 of 50", so the events that were
        # already processed are dropped and only the genuinely new ones -- the
        # outcome of the round -- are kept.
        already_seen = last_game_state["step"] == self.last_recorded_step

        if already_seen:
            events = [event for event in events if event in TERMINAL_EVENTS]

        add_custom_events(last_game_state, None, events)

        reward = reward_from_events(self, events)

        self.buffer.push(
            Transition(
                state_to_features(last_game_state),
                ACTIONS.index(last_action),
                reward,
                None,   # terminal: nothing to bootstrap from
                np.zeros(len(ACTIONS), dtype=bool),
            )
        )

        self.episode_reward += reward
        self.episode_coins += events.count(e.COIN_COLLECTED)
        self.episode_crates += events.count(e.CRATE_DESTROYED)

        if not already_seen:
            self.episode_steps += 1

    for _ in range(UPDATES_AT_ROUND_END):
        optimise(self)

    # -----------------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------------
    self.episodes_trained += 1
    self.epsilon = epsilon_for(self.episodes_trained)

    self.training_history.append(
        {
            "episode": self.episodes_trained,
            "steps": self.episode_steps,
            "reward": self.episode_reward,
            "coins": self.episode_coins,
            "crates": self.episode_crates,
            "suicide": int(e.KILLED_SELF in events),
            "epsilon": self.epsilon,
        }
    )

    if self.episodes_trained % LOG_EVERY == 0:
        window = self.training_history[-LOG_EVERY:]
        self.logger.info(
            f"Episode {self.episodes_trained}: "
            f"mean reward {np.mean([h['reward'] for h in window]):7.1f}, "
            f"mean coins {np.mean([h['coins'] for h in window]):5.2f}, "
            f"mean crates {np.mean([h['crates'] for h in window]):5.2f}, "
            f"mean steps {np.mean([h['steps'] for h in window]):6.1f}, "
            f"suicides {sum(h['suicide'] for h in window):3d}/{len(window)}, "
            f"epsilon {self.epsilon:.3f}"
        )

    self.last_recorded_step = -1
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_crates = 0
    self.episode_steps = 0

    save_model(self)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def optimise(self):
    """
    One Q-learning step on a random batch from the replay buffer.

    The batch is split by action because each action has its own weight
    vector, and a sample only carries information about the action it took.
    """
    if len(self.buffer) < MIN_BUFFER_SIZE:
        return

    batch = self.buffer.sample(BATCH_SIZE)

    for action_index in range(len(ACTIONS)):
        rows = [t for t in batch if t.action == action_index]

        if not rows:
            continue

        phi = np.array([t.state for t in rows])
        targets = np.array([td_target(self, t) for t in rows])

        td_error = targets - phi @ self.model[action_index]

        self.model[action_index] += ALPHA * (td_error[:, None] * phi).mean(axis=0)


def td_target(self, transition: Transition) -> float:
    """
    r + gamma * max_{a'} Q(s', a'), maximised over the actions that are
    actually available in s'.
    """
    if transition.next_state is None:
        return transition.reward

    q_next = self.model @ transition.next_state
    q_next = q_next[transition.next_valid]

    if q_next.size == 0:
        return transition.reward

    return transition.reward + GAMMA * q_next.max()


def epsilon_for(episode: int) -> float:
    """Linear exploration schedule."""
    fraction = min(episode / EPSILON_DECAY_EPISODES, 1.0)

    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def save_model(self):
    saved_data = {
        "weights": self.model,
        "epsilon": self.epsilon,
        "episodes_trained": self.episodes_trained,
        "training_history": self.training_history,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(saved_data, file)


# ---------------------------------------------------------------------------
# Events and rewards
# ---------------------------------------------------------------------------

def add_custom_events(old_game_state: dict, new_game_state: dict, events: List[str]):
    """
    Append the auxiliary events that carry the shaping reward.

    Both the distance comparison and the bomb judgement are computed from the
    same helpers the features use, so the reward signal and the state
    representation never point in opposite directions.
    """
    # --- Judging a bomb drop straight away --------------------------------
    if e.BOMB_DROPPED in events:
        x, y = old_game_state["self"][3]

        if crate_count(x, y, old_game_state["field"]) > 0:
            events.append(GOOD_BOMB)
        else:
            events.append(USELESS_BOMB)

    if new_game_state is None:
        return

    # --- Walking towards the next coin or crate ---------------------------
    if e.COIN_COLLECTED in events:
        # The target just disappeared, so comparing distances is meaningless.
        return

    old_distance, old_is_coin = target_distance(old_game_state)
    new_distance, new_is_coin = target_distance(new_game_state)

    # Comparing distances only makes sense for the same kind of target.
    if (
        old_distance is not None
        and new_distance is not None
        and old_is_coin == new_is_coin
    ):
        if new_distance < old_distance:
            events.append(MOVED_TOWARD_TARGET)
        elif new_distance > old_distance:
            events.append(MOVED_AWAY_FROM_TARGET)


def reward_from_events(self, events: List[str]) -> float:
    """
    Maps game events to scalar rewards.
    """
    total = sum(REWARDS.get(event, 0.0) for event in events)

    self.logger.debug(f"Reward {total:.2f} from events: {', '.join(events)}")

    return total


def record_step(self, reward: float, events: List[str]):
    """Update the per-episode counters."""
    self.episode_reward += reward
    self.episode_steps += 1
    self.episode_coins += events.count(e.COIN_COLLECTED)
    self.episode_crates += events.count(e.CRATE_DESTROYED)
