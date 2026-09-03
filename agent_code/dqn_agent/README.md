# Warm-started Double DQN agent

This agent is separate from `q_learning_agent`. It reuses the selected v2.2
39-feature representation, safety shield, anti-stall guard, post-bomb escape,
and conservative crate-bomb deferral.

On its first run, when `dqn_model.pt` is absent, the neural network is
initialized so that its Q-values exactly equal the selected linear
`q_model.pkl`. The initial DQN policy is therefore the proven v2.2 policy,
not a random policy.

## Install PyTorch

From the repository root, using the existing environment:

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python -m ensurepip --upgrade
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python -m pip install -r requirements-dqn.txt
```

## Verify the warm start visually (no training)

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python main.py play \
  --agents dqn_agent rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --seed 42 \
  --n-rounds 3 \
  --update-interval 0.15
```

## First training stage

Do not remove or overwrite `q_learning_agent/q_model.pkl`. Train the separate
DQN for 1,000 classic rounds:

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python main.py play \
  --agents dqn_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --scenario classic \
  --seed 123 \
  --n-rounds 1000 \
  --no-gui
```

The learned checkpoint is saved only as
`agent_code/dqn_agent/dqn_model.pt`. Evaluation commands must omit
`--train 1`.

The default device is CPU because this network is small. To request Apple
Metal explicitly, set `BOMBERMAN_DQN_DEVICE=mps`; the code safely falls back
to CPU when MPS is unavailable.

