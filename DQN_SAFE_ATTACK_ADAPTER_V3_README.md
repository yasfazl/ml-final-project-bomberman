# DQN safe-attack adapter V3

## Purpose

Version 2 is preserved as the selected stable policy. It sometimes reaches an
empty-board combat state where an opponent is in bomb range but placing a bomb
on the current tile is not robustly escapable. The agent can then oscillate or
wait instead of staging a safer attack.

V3 adds a small learned adapter for only that situation. It does not add a
hard action override.

## Feature layout

- `0-38`: selected Q-learning representation, frozen.
- `39-44`: V2 opponent-pursuit adapter, frozen.
- `45`: safe-attack mode active.
- `46-49`: first movement direction toward the nearest robust attack tile.
- `50`: normalized distance to that tile.
- `51`: a robust opponent-targeting bomb is safe on the current tile.

The new block is exactly zero unless:

1. no visible coins or crates remain;
2. at least one opponent remains;
3. there are no active bombs or explosions;
4. the agent has a bomb available; and
5. an opponent is in the current bomb blast line.

## Preservation guarantee

Loading a 45-feature V2 checkpoint expands only the first input layer. Its
seven new columns are initialized to zero, so every pre-training Q-value is
unchanged. During training, a gradient hook zeros columns `0-44`; only columns
`45-51` may change.

## Training and evaluation

Start with a short 200-round empty-board training run. Evaluate against the
unchanged V2 checkpoint on the same six seeds in both `empty` and `classic`
scenarios. Accept V3 only if empty-board combat improves without materially
damaging classic score, coins, suicides, or crate efficiency.

Do not overwrite or delete the V2 checkpoint. Evaluation commands must omit
`--train 1`.
