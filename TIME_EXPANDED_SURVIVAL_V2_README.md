# Time-expanded survivability shield v2.2

This package is designed to be applied on top of the validated
`experiment/post-bomb-escape-v1` code. It does not contain or modify a
trained model.

## What changed

- Builds a separate danger schedule for each future action time instead of
  storing only the earliest danger per tile.
- Uses the blast propagation implemented by this repository's game engine:
  stone walls stop explosions, while crates do not.
- Searches over `(position, time)` so waiting and revisiting a tile after an
  explosion are represented.
- Removes candidate actions that have no route through all currently known
  bomb and explosion danger.
- When an opponent is within four tiles, permits a new bomb only if there are
  at least two distinct exit tiles outside its blast within two movements.
- If all simulated candidates appear unsafe, falls back to non-bomb actions
  and never adds another bomb to an unsolved danger state.
- Tracks repeated safe WAIT selections at one position.
- After three safe waits, temporarily removes WAIT only when the existing
  safety filters have approved a movement toward a reachable coin or safe
  crate-bombing tile.
- Falls back to crate progress when a visible coin is currently unreachable.
- In safe crate mode, defers an immediate bomb when moving one approved tile
  first increases its conservative crate yield by at least two crates.
- The crate deferral never interrupts a reachable coin, an immediate opponent
  bomb target, or a position reached by known explosion danger.
- A distant bomb does not disable crate optimization; both the movement and
  the future bomb must still pass the existing survivability checks.

## Preserved behavior

- `FEATURE_DIM` remains 39.
- Features and learned weights are unchanged.
- The shield filters only the final candidate action set.
- Safe states without a repeated wait retain the original candidate set.
- The anti-stall guard never overrides known danger or a safety-rejected move.
- Immediate 1 -> 2 crate improvements remain with the learned policy; the new
  gate is intentionally limited to material gains such as 1 -> 3.

## Verification

The reviewed source passes 99 tests, including 17 focused v2 safety tests,
9 focused anti-stall tests, and 13 focused crate-deferral tests. The
expected selected model SHA-256 is:

`43d1f9e1402158fdca59a5520f478a0feec6dd0fb2fb4b13db4358257576cdb2`

Do not train this candidate before completing paired evaluation against the
validated post-bomb shield.
