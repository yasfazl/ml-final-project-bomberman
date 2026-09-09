# Safe-attack warm-started Double DQN agent

The V5 experimental training path uses causal bomb-outcome reward
redistribution. It removes the flat reward for merely targeting an opponent
and assigns discounted owner-specific kill/self-kill outcomes to the original
bomb action. Evaluation behavior and the network architecture are unchanged.
See `DQN_CAUSAL_BOMB_REWARD_V5_README.md` at the repository root.

This agent is separate from `q_learning_agent`. It preserves the selected
v2.2 39-feature representation and every existing safety filter. Six gated
endgame inputs extend the network to 45 dimensions: pursuit-active, four path
directions, and normalized opponent distance. Version 3 adds seven strictly
gated safe-attack inputs, for a total of 52 dimensions: attack-mode active,
four directions toward a robust bombing tile, normalized distance, and
safe-bomb-now.

A version-1 39-feature or version-2 45-feature DQN checkpoint is migrated
automatically by adding zero input columns. Its existing Q-values therefore
remain unchanged. When no DQN checkpoint exists, the network is exactly
warm-started from the selected linear `q_model.pkl`.

Only the seven safe-attack first-layer input connections are trainable. Every
version-2 connection, including the existing pursuit adapter, is frozen.
Safe-attack inputs are zero unless all coins and crates are gone, the board has
no bomb or explosion danger, a bomb is available, and an opponent is already
in the current blast line. The adapter then learns whether to bomb now or move
toward the nearest opponent-targeting tile with a robust escape route.

This is a learned residual behavior, not a deterministic action override.
Coin, crate, danger, and ordinary pursuit decisions are exactly version 2
before and after adapter training because the new inputs are zero there.

Install PyTorch with the project environment:

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python -m ensurepip --upgrade
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python -m pip install -r requirements-dqn.txt
```

Focused safe-attack training:

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python main.py play \
  --agents dqn_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --scenario empty \
  --seed 123 \
  --n-rounds 200 \
  --no-gui
```

The learned checkpoint is saved only as
`agent_code/dqn_agent/dqn_model.pt`. Evaluation must omit `--train 1`.
