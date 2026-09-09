# DQN conflict-aware guard v1

This overlay addresses simultaneous movement collisions without retraining or
changing the feature vector, network, rewards, or checkpoint format.

## Evidence

The invalid-action diagnostic recorded 829 invalid actions on an empty-board
seed-42 run. Every one was a `simultaneous_agent_collision`: the destination
was free in the state used by the DQN, but an opponent entered it before the
DQN action was executed. Of those records, 652 occurred in endgame mode and
177 in danger mode.

## Policy

The engine asks agents to choose from the same snapshot and then executes the
actions in a randomized order. A movement destination is therefore marked as
*contested* when an opponent is one movement step away from that currently
free tile.

- **Known bomb danger:** contested movement is removed only when another
  movement action has already passed all existing time-expanded safety
  filters. If the contested move is the only possible escape movement, the
  original candidates are kept.
- **Safe endgame pursuit/attack:** the learned best action remains in control.
  A contested best movement is replaced only when an uncontested candidate is
  within 5% of the best action's Q-value scale.
- **Coins and crates:** the soft guard is inactive, preserving the selected V3
  policy in ordinary objective play.

`WAIT` and `BOMB` are never classified as movement collisions. They may serve
as the near-optimal endgame alternative only if the established candidate
filters already approved them.

## Intentionally unchanged

- `dqn_model.pt` and every learned parameter
- 52-feature representation
- policy and target network architecture
- rewards and training code
- bomb, post-bomb, time-expanded, crate-deferral, and anti-stall filters
- diagnostic collection

## Required validation

Run the complete test suite first. Then compare the selected V3-75 checkpoint
with this guard on the same empty and classic seed sets. Keep the guard only if
simultaneous collisions drop materially while score, kills, suicides, and
crate efficiency remain within the predeclared acceptance limits.

