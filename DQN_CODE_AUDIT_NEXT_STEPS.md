# Selected DQN V3-75: code audit and remaining priorities

## What is already strong

- The 39-feature Q-learning policy is preserved exactly in normal modes.
- Endgame pursuit and safe attack are strictly gated away from coins, crates,
  bombs, and explosions.
- Time-expanded danger filtering, post-own-bomb commitment, anti-stall logic,
  and crate-bomb deferral all run before Q-value selection.
- Checkpoint migration expands only the first input layer and has focused tests.
- V3-75 improves classic score and survival while substantially improving
  empty-board kills relative to V2.

## Audit findings

### 1. A game seed does not make the whole evaluation deterministic

`--seed` controls arena construction and randomized engine action order. The
rule-based agents call `np.random.seed()` without a seed and use unseeded
Python `shuffle`; DQN tie-breaking also uses the unseeded PyTorch generator.
Consequently, repeated runs with the same CLI seed need not reproduce the same
trajectory. Existing multi-seed means are useful, but their rows are not
strict common-random-number pairs.

### 2. The engine can invalidate an action that was legal when selected

All agents choose from one snapshot. Their actions are then executed in a
random order. If two agents select the same free destination, the first enters
it and the second receives `INVALID_ACTION`. The current mask blocks current
opponent positions but cannot know their selected actions.

### 3. Safe-attack representation cannot express collision risk

Features `39-51` describe pursuit direction, distance, attack direction, and
self escape. They do not encode whether each candidate destination is also
reachable by an opponent in one action. More training cannot resolve states
that are identical in the feature vector but differ in collision risk.

### 4. Bomb permission and robust bomb planning use different safety tests

The base candidate mask permits a targeted bomb when
`has_escape_route_after_bomb` finds one route. The safe-attack planner calls
the stricter `bomb_has_robust_escape_route`, which asks for disruption-tolerant
exits near opponents. Thus a non-robust bomb can remain selectable while the
safe-attack features recommend moving to a robust bombing position.

### 5. Current safe attack gains kills through high bomb volume

V3-75 improves combat, but its empty-board bomb count and invalid-action count
are much higher than V2 and its kills per 100 bombs are lower. The five-step
experiment amplified that behavior, confirming that additional generic TD
credit is not the right correction.

## Recommended order

1. Use the diagnostics-only experiment to classify invalid actions.
2. If simultaneous movement collisions dominate, test a strictly endgame-only
   contested-destination tie-breaker. Preserve the DQN choice whenever its
   Q-value advantage is material.
3. Separately test a robust-bomb consistency gate: in safe endgame mode, remove
   a fragile `BOMB` candidate only when a robust movement alternative exists.
4. Improve the evaluation protocol by using repeated runs or locally seeded
   opponent RNGs before attempting Bayesian hyperparameter optimization.
5. Freeze the best candidate and finish the report.

## Changes not recommended under the remaining project budget

- More V3 or V4 episodes without a representation change.
- More reward terms before invalid causes are measured.
- A deeper network, recurrent model, or full-network unfreezing.
- Bayesian optimization over many reward values using the current noisy
  evaluation protocol.
