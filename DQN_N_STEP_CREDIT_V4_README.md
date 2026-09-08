# DQN V4: five-step bomb credit assignment

## Why this experiment exists

The selected V3-75 policy improves empty-board combat and reduces classic-mode
suicides, but it still places many bombs and produces many invalid actions. Its
safe-attack inputs are visible immediately before a bomb is placed and then
become zero as soon as that bomb is active. With one-step DQN targets, the
later kill or suicide therefore cannot directly train the decision that placed
the bomb.

V4 changes only the temporal credit assignment. The feature representation,
network architecture, action masks, reward function, safety filters, and
inference policy remain unchanged.

## Five-step target

For a transition beginning at time `t`, replay stores

```text
R_t^(m) = r_t + gamma*r_(t+1) + ... + gamma^(m-1)*r_(t+m-1)
```

where `m = 5`, except near the end of a round where the remaining shorter
prefix is used. The masked Double DQN target is

```text
y_t = R_t^(m) + gamma^m * (1-done) *
      Q_target(s_(t+m), argmax_valid Q_policy(s_(t+m), a))
```

With `gamma = 0.90`, a reward four actions later still contributes with weight
`0.90^4 = 0.6561` to the original action. This matches the typical delay from
bomb placement to explosion.

## Preservation guarantees

- Start from `selected_dqn_v3_scale075.pt`.
- Features `0-44` and every later network layer remain frozen.
- Only first-layer columns `45-51` are trainable.
- V3 optimizer momentum is discarded when V4 starts; an actual V4 checkpoint
  can resume its own optimizer state.
- The original Q-learning `.pkl` model is never modified.
- Evaluation behavior is identical before V4 training because callbacks and
  action selection are unchanged.

## Recommended first experiment

Train for only 200 rounds on the empty scenario, then stop and validate. Do not
continue training merely because training score rises.

```bash
~/Desktop/bomberman_rl_nstep_agent/.venv/bin/python main.py play \
  --agents dqn_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --scenario empty \
  --seed 4242 \
  --n-rounds 200 \
  --no-gui
```

V4 should be accepted only if it improves bomb/invalid-action efficiency while
retaining V3-75's classic score, kills, and suicide reduction across fixed and
fresh validation seeds.
