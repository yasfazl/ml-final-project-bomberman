# DQN invalid-action diagnostics

This experiment changes no agent decision, feature, action mask, reward,
network weight, or checkpoint. It records ground-truth information only when
explicitly enabled through environment variables.

The engine requests every action from the same state snapshot and then
executes agents in randomized order. A movement that was valid when selected
can therefore become invalid when an earlier agent enters the same tile.

Enable recording for the DQN agent:

```bash
export BOMBERMAN_INVALID_DIAGNOSTICS=1
export BOMBERMAN_INVALID_DIAGNOSTICS_AGENT=dqn_agent
export BOMBERMAN_INVALID_DIAGNOSTICS_PATH=diagnostics/invalid_seed42.jsonl
```

After an evaluation, disable recording:

```bash
unset BOMBERMAN_INVALID_DIAGNOSTICS
unset BOMBERMAN_INVALID_DIAGNOSTICS_AGENT
unset BOMBERMAN_INVALID_DIAGNOSTICS_PATH
```

Summarize one or more files:

```bash
python tools/analyze_invalid_actions.py diagnostics/*.jsonl --details
```

The most important category is `simultaneous_agent_collision`: the selected
destination was legal in the DQN's observation, but another agent occupied it
before the DQN's action was executed.
