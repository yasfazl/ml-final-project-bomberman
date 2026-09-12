# DQN reward Bayesian search V6

## Decision

Use a small sequential Optuna TPE study, not greedy coordinate search.  The
three rewards interact, each complete training/evaluation is expensive, and
the selected V3-75 checkpoint remains the untouched control.

This is the final bounded tuning experiment.  It is not permission to search
more features, network sizes, learning rates, gamma values, or seed sets.

## Search space

| Parameter | Range | V3 value | Meaning |
|---|---:|---:|---|
| `progress_reward` | 0.10–1.00 | 0.50 | `+x` toward and `-x` away from a robust attack tile |
| `wait_penalty` | 0.25–2.00 | 1.00 | `-x` for waiting while safe-attack mode is active |
| `targeted_bomb_reward` | 0.50–4.00 | 4.00 | Immediate reward for a bomb targeting an opponent |

Only first-layer columns `45–51` remain trainable.  Features `0–44`, all later
layers, action masks, replay settings, gamma, and learning rate are unchanged.
Kill/self-kill rewards are intentionally excluded: with V3's one-step update
they occur while the safe-attack inputs are zero, so searching them would not
directly train the adapter.

## Submission-safe base

The DQN now carries exact vendored copies of the frozen Q-learning callbacks,
reward helpers, game utilities, and linear fallback model. Runtime imports no
longer depend on `agent_code/q_learning_agent`, which is essential because the
course submission contains only the selected agent directory. The final
`dqn_model.pt` is still supplied by you because it is the preserved trained
checkpoint; remember that the repository's historical `.gitignore` excludes
it unless it is force-added.

## Experimental controls

- Every trial is copied from the exact same V3-75 checkpoint.
- Every trial uses a fresh Adam optimizer; stale momentum is not restored.
- Python, NumPy, PyTorch, replay sampling, and the bundled rule-based agents
  receive deterministic experiment seeds.
- Trials run sequentially.
- The untouched baseline and all candidates use the same five development
  world seeds.
- The Optuna sampler uses five random startup trials and multivariate TPE.
- The SQLite study is resumable.
- The current `agent_code/dqn_agent/dqn_model.pt` is restored even after an
  exception or keyboard interruption.
- No candidate is installed automatically.

## Objective and constraints

The objective maximizes empty-board score per 100 rounds.  A candidate is
eligible for later validation only if it also:

1. matches or exceeds baseline development score;
2. stays within `+0.5` suicides per 100 rounds of baseline; and
3. retains at least 95% of baseline kills per 100 bombs.

These are search-screening constraints, not final proof.  The best eligible
candidate must still pass the established six-seed comparison and then one
sealed, previously unused holdout evaluation.

## Installation and tests

Run this experiment from the V3 branch, not the rejected V5 branch.  Ensure
the selected checkpoint exists:

```bash
mkdir -p checkpoints/dqn

/bin/cp -f \
  checkpoints/dqn/before_nstep_v4_scale075.pt \
  checkpoints/dqn/selected_dqn_v3_scale075.pt

/bin/cp -f \
  checkpoints/dqn/selected_dqn_v3_scale075.pt \
  agent_code/dqn_agent/dqn_model.pt
```

Install the development-only optimizer and run tests:

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  -m pip install -r requirements-reward-search.txt

~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  -m pytest tests/test_dqn_reward_bayes_v6.py -q

~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  -m pytest tests -q
```

## Pipeline smoke test

This checks orchestration only and uses a separate output directory.  It is
too short to judge rewards.

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  tools/optimize_dqn_rewards.py \
  --baseline-checkpoint checkpoints/dqn/selected_dqn_v3_scale075.pt \
  --output-dir optimization/reward_bayes_v6_smoke \
  --trials 2 \
  --train-rounds 2 \
  --eval-rounds 2 \
  --dev-seeds 1103 2207
```

## Full bounded study

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  tools/optimize_dqn_rewards.py \
  --baseline-checkpoint checkpoints/dqn/selected_dqn_v3_scale075.pt \
  --output-dir optimization/reward_bayes_v6 \
  --trials 15
```

The default development panel is `1103 2207 3301 4409 5501`, with 30 rounds
per seed and 100 additional training rounds per candidate.  Re-running the
same command resumes until 15 completed trials.  Do not increase the budget
after inspecting the results.

Important outputs:

- `optimization/reward_bayes_v6/study.sqlite3`
- `optimization/reward_bayes_v6/trials.csv`
- `optimization/reward_bayes_v6/study_summary.json`
- `optimization/reward_bayes_v6/best_feasible.pt` (only if one exists)

If `decision` is `keep_v3_75`, stop the reward search.  If it is
`validate_best_feasible_on_six_seeds`, do not install the candidate yet; run
the six-seed confirmation first.
