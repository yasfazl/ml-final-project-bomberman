# DQN V5: Causal Bomb-Outcome Reward

## Hypothesis

V3 gives a flat `+4` reward when an opponent is in the hypothetical blast
line at bomb-placement time. The opponent can subsequently escape, so this is
a false-positive learning signal. The true kill or self-kill event arrives
when the bomb explodes, normally four steps later, and one-step DQN otherwise
associates it with the action taken at explosion time.

V4 attempted to solve that delay with five-step returns. It accumulated all
intermediate dense shaping rewards and increased bombing, invalid actions, and
suicides. V5 instead redistributes only the outcome caused by the bomb.

## Method

After a successful `BOMB_DROPPED` event, V5:

1. removes the flat `BOMB_TARGETED_OPPONENT` bonus;
2. stages the original one-step replay transition;
3. waits for the owner-specific `BOMB_EXPLODED` event;
4. counts `KILLED_OPPONENT` and `KILLED_SELF` events from that explosion;
5. adds the discounted causal outcome to the staged bomb transition;
6. removes those same causal events from the explosion-time transition to
   prevent double counting.

For a bomb placed at time `t` and resolved after `d` steps:

```text
r_bomb = r_placement_without_flat_bonus
         + gamma**d * (20 * opponent_kills - 30 * self_kills)
```

`GOT_KILLED` remains on the explosion-time movement transition. It represents
the quality of the escape policy, whereas `KILLED_SELF` represents the causal
responsibility of placing the bomb.

A bomb that misses receives no invented success reward and no new miss
penalty. If a round ends before a bomb outcome can be observed, the staged
transition is flushed with its ordinary placement reward.

This is a small event-based reward redistribution inspired by RUDDER, not a
full implementation of RUDDER's learned return-decomposition network.

## Preserved components

- V3's 52-dimensional state representation
- the selected V3-75 checkpoint as initialization
- network architecture and action masks
- time-expanded survival logic
- all coin, crate, pursuit, and escape shaping
- one-step masked Double DQN targets
- training restricted to first-layer columns 45--51

The optimizer is fresh when starting from V3. Optimizer state is restored only
when resuming a checkpoint whose algorithm identifier is already V5.

## Experimental protocol

Train exactly 200 empty-board rounds from V3-75 using seed 4242. Do not tune
reward coefficients. Evaluate once on the established six development seeds:

```text
42, 999, 2026, 7, 314, 65537
```

Run classic evaluation only if empty-board combat passes all primary checks:

- score is at least 95% of V3-75;
- kills are at least 95% of V3-75;
- suicides do not increase;
- kills per 100 bombs improve by at least 10%.

Bomb-count reduction is secondary because tournament score, kills, and
survival are the real objectives. If the experiment fails, preserve it as a
negative result and restore V3-75.

## References

- Arjona-Medina et al., *RUDDER: Return Decomposition for Delayed Rewards*,
  NeurIPS 2019: https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html
- Meisheri et al., *Accelerating Training in Pommerman with Imitation and
  Reinforcement Learning*, 2019: https://arxiv.org/abs/1911.04947
- Wiewiora, *Potential-Based Shaping and Q-Value Initialization are
  Equivalent*, JAIR 2003: https://arxiv.org/abs/1106.5267
