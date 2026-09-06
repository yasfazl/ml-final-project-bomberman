# Opponent-aware warm-started Double DQN agent

This agent is separate from `q_learning_agent`. It preserves the selected
v2.2 39-feature representation and every existing safety filter. Six gated
endgame inputs extend the network to 45 dimensions: pursuit-active, four path
directions, and normalized opponent distance.

A version-1 39-feature DQN checkpoint is migrated automatically by adding six
zero input columns. Its existing Q-values therefore remain unchanged. When no
DQN checkpoint exists, the network is exactly warm-started from the selected
linear `q_model.pkl`.

Only the six new first-layer input connections are trainable. These inputs are
zero outside safe endgame pursuit, so training cannot alter coin, crate, or
bomb-escape Q-values. Endgame exploration has a separate epsilon, and a small
position-history guard breaks repeated A-B-A-B movement when a safe pursuit
move is available.

Install PyTorch with the project environment:

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python -m ensurepip --upgrade
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python -m pip install -r requirements-dqn.txt
```

Focused endgame training:

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python main.py play \
  --agents dqn_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --scenario empty \
  --seed 123 \
  --n-rounds 300 \
  --no-gui
```

The learned checkpoint is saved only as
`agent_code/dqn_agent/dqn_model.pt`. Evaluation must omit `--train 1`.

