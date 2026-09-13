# DQN suicide diagnostics V1

This package adds opt-in, policy-neutral death diagnostics to the game engine.
The selected V2 DQN agent is not modified.

For every DQN death, the diagnostic output records:

- the exact explosion owner;
- own-bomb versus opponent-bomb death;
- bomb origin and placement step;
- the last twelve DQN decisions and positions;
- the safety model result at bomb placement and during escape;
- whether the agent left and later re-entered its own blast;
- whether a later opponent bomb or adjacent opponent was observed.

Diagnostics are disabled unless BOMBERMAN_SUICIDE_DIAGNOSTICS=1. When
disabled, no diagnostic file is created and the original game path is used.
No training, reward, feature, action-filter, network, or checkpoint code is
changed.

## Install

~~~bash
cd ~/Desktop/bomberman_rl_dqn_suicide_diag

/bin/cp -f environment.py /tmp/environment_before_suicide_diagnostics.py

unzip -o ~/Desktop/DQN_SUICIDE_DIAGNOSTICS_V1_READY.zip

~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  -m pytest tests -q
~~~

## Collect diagnostic evidence

~~~bash
mkdir -p diagnostics/suicide_v1
export BOMBERMAN_SUICIDE_DIAGNOSTICS=1

for seed in 42 999 2026 7 314 65537; do
  export BOMBERMAN_SUICIDE_DIAGNOSTICS_PATH=\
"diagnostics/suicide_v1/classic_seed${seed}.jsonl"

  ~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python main.py play \
    --agents dqn_agent rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic \
    --seed "${seed}" \
    --n-rounds 100 \
    --no-gui \
    --save-stats "diagnostics/suicide_v1/classic_seed${seed}.json"
done

unset BOMBERMAN_SUICIDE_DIAGNOSTICS
unset BOMBERMAN_SUICIDE_DIAGNOSTICS_PATH
~~~

## Summarize

~~~bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python \
  tools/analyze_suicide_diagnostics.py \
  diagnostics/suicide_v1/*.jsonl \
  --details
~~~
