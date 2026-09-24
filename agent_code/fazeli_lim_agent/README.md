# Fazeli–Lim Agent

`fazeli_lim_agent` is the final submitted agent for the Bomberman reinforcement-learning project. It is the renamed submission version of our final collaborative Q-learning agent, previously evaluated as **Collab v3** or **Q-V3**.

## Overview

The agent uses Q-learning with a linear action-value approximation over engineered state features. Rather than representing every board configuration in a tabular Q-table, it extracts features describing strategically relevant properties of the current state and estimates the value of each available action.

The feature representation includes information related to:

- reachable coins and crates;
- nearby opponents;
- bombs and imminent explosions;
- escape routes and safe movement;
- the expected usefulness of placing a bomb;
- movement toward relevant objectives;
- invalid or dangerous actions;
- the transition from crate collection to opponent pursuit.

During inference, the agent first constructs the feature vector for each action. It then evaluates the actions using the trained parameters stored in `q_model.pkl`. Unsafe and invalid actions are filtered where appropriate, and the highest-valued remaining action is selected. If multiple actions have the same maximum value, the implementation may choose randomly between them.

The final version also relaxes part of the conservative bomb-escape policy after all crates have been destroyed. This allows the agent to pursue and attack opponents more aggressively during the endgame instead of continuing to behave like a crate-collection agent.

## Files

- `callbacks.py`: agent setup, feature construction, action selection, safety filtering, and model loading.
- `game_utils.py`: board analysis, path finding, danger estimation, and bomb-related utilities.
- `train.py`: Q-learning updates and reward shaping used during training.
- `q_model.pkl`: trained parameters loaded by the final inference agent.
- `my-saved-model.pt`: additional saved model artifact retained with the trained submission.

The submitted agent is intended to run in evaluation mode. Running a normal `play` command does not continue training or modify the trained model files.

## Running the agent

From the repository root, the agent can be run visually with:

```bash
python main.py play \
  --agents \
    fazeli_lim_agent \
    rule_based_agent \
    rule_based_agent \
    rule_based_agent \
  --scenario classic \
  --n-rounds 10
```

For a headless smoke test, add `--no-gui`:

```bash
python main.py play \
  --agents \
    fazeli_lim_agent \
    random_agent \
    random_agent \
    random_agent \
  --scenario classic \
  --n-rounds 1 \
  --no-gui
```

## Final evaluation

The final comparison evaluated each learned agent separately in player slot zero against the same three rule-based agents. Matched world seeds and experiment seeds were used for all candidates.

Across 20 paired seeds and 100 rounds per seed, the principal results were:

| Agent | Score per 100 rounds | Kills per 100 rounds | Suicides per 100 rounds |
|---|---:|---:|---:|
| Fazeli–Lim / Collab v3 | 413.65 | 19.65 | 24.50 |
| Phase-V8 DQN | 408.95 | 16.90 | 9.00 |
| Phi agent | 352.15 | 10.60 | 45.70 |

Official game score was the primary selection criterion. The Fazeli–Lim agent achieved the highest observed mean score, although its advantage over Phase-V8 was not statistically conclusive: the paired mean difference was `+4.70`, with a 95% bootstrap confidence interval of `[-14.05, +24.05]`. It clearly outperformed the Phi agent in this evaluation.

Suicide count was treated as a diagnostic measurement rather than an independent objective. A more aggressive agent may accept additional risk while still obtaining a higher official score through crate destruction, coin collection, and opponent elimination.

## Reproducibility checks

Before submission, the final agent was:

- executed without training in a Linux AMD64 Docker container;
- tested using the extracted submission archive;
- checked to ensure that both model files remained unchanged during evaluation;
- compared using matched evaluation conditions;
- included in the public repository’s `main` branch.

Repository: <https://github.com/yasfazl/ml-final-project-bomberman>
