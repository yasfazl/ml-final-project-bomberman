# Potential-Shaping V7

This overlay implements one controlled potential-based reward-shaping
experiment for the DQN safe-attack adapter.

## Scientific change

V7 replaces the safe-attack movement, waiting, and immediate targeted-bomb
bonuses with:

```text
F(s, s') = beta * (gamma * Phi(s') - Phi(s))
```

The bounded state potential is zero while coins or crates remain, zero for
terminal states, below `0.1` while approaching a robust attack tile, `0.1`
when a robust opponent-targeting bomb is ready, and `1.0` immediately after a
useful bomb is placed with an escape path.  With `gamma=0.9` and `beta=5`, the
ready-to-post-bomb transition contributes approximately `+4`, matching the
scale of the replaced V3 bomb bonus without a parameter sweep.

V7 does not change inference features, action masks, network architecture, or
the protected V3-75 checkpoint.  Training is restricted to first-layer input
columns 45-51.

## Install in the Mac repository

From `~/Desktop/bomberman_rl_dqn_safe_attack_v3`, first create the isolated
branch:

```bash
git switch -c experiment/dqn-potential-shaping-v7
```

Copy the overlay files into the repository while preserving their relative
paths.  Do not copy or replace any `.pt` or `.pkl` file; the package contains
none.

## Verify before training

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python -m pytest \
  tests/test_dqn_potential_shaping_v7.py \
  tests/test_dqn_safe_attack_adapter.py \
  tests/test_dqn_reward_bayes_v6.py \
  tests/test_dqn_agent.py \
  tests/test_dqn_self_contained.py \
  -q

shasum -a 256 \
  checkpoints/dqn/selected_dqn_v3_scale075.pt \
  agent_code/dqn_agent/dqn_model.pt
```

Both hashes must be:

```text
944377133b551419cd0a01af50ba9983a04e04366574ec50e32a2c905df27b69
```

## Train exactly once

```bash
caffeinate -i \
  ~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  tools/train_dqn_potential_v7.py
```

The fixed protocol uses V3-75, a fresh optimizer, replay/experiment seed
`60013`, world seed `4242`, the `empty` scenario, and 100 rounds.  The script
refuses an unexpected checkpoint, verifies that only adapter columns 45-51
changed, saves the candidate under `optimization/potential_shaping_v7/`, and
restores V3-75 even if training fails.

Do not install the V7 candidate after training.  Send the printed candidate
hash and `optimization/potential_shaping_v7/summary.json` for evaluation first.

## Selection rule

V7 is a scientific candidate, not an automatic replacement.  Official
classic score is the primary selection metric.  Suicide, invalid actions, and
bomb efficiency are diagnostics and score-tie breakers.  V3-75 remains the
incumbent unless V7 demonstrates a reproducible classic-score improvement.
