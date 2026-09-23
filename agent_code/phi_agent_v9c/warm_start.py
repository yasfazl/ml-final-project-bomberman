"""
Carry a trained 40-slot model over into this agent's 44-slot layout.

    python -m agent_code.q_learning_agent_task2_enemy.warm_start \
        agent_code/q_learning_agent_task2_phi/q_model.pkl

The four opponent slots start at exactly 0, so on the first step the rebuilt
model scores every action identically to the model it came from. Training then
moves all 44 weights, and the new four grow away from 0 as evidence arrives.

WHY THIS IS NOT np.concatenate([w, np.zeros(4)])
------------------------------------------------
Slot 27 was inserted inside the MOVE block, so everything after it shifted:

        old (40)        new (44)
    MOVE  0 - 26         0 - 26      unchanged
                         27          new
    WAIT 27 - 34        28 - 35      +1
                         36          new
    BOMB 35 - 39        37 - 41      +2
                         42, 43      new

Appending zeros would leave every WAIT weight sitting in a MOVE slot and every
BOMB weight in a WAIT slot -- silently, since the shapes would still match.

WHAT THIS COSTS
---------------
The starting point is a policy shaped by 40 features, so the run explores
around habits that policy already has rather than from scratch. It is cheaper
than retraining, but the comparison it supports is "3000 + N rounds with
opponents features" against "3000 + N rounds without", which means the baseline
has to be given the same N rounds. Comparing against the untouched 3000-round
model would credit the features with N rounds of ordinary training.
"""
from pathlib import Path
import pickle
import sys

import numpy as np

from .callbacks import FEATURE_DIM, MODEL_PATH

OLD_DIM = 40

# (destination slice, source slice) for the blocks that survive unchanged.
BLOCK_MAP = [
    (slice(0, 27), slice(0, 27)),     # MOVE, minus the new slot 27
    (slice(28, 36), slice(27, 35)),   # WAIT
    (slice(37, 42), slice(35, 40)),   # BOMB
]

NEW_SLOTS = [27, 36, 42, 43]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(MODEL_PATH)

    if destination.exists():
        raise SystemExit(
            f"{destination} already exists. Move it aside first -- overwriting "
            "a trained model by accident is not worth the convenience."
        )

    with source.open("rb") as file:
        saved = pickle.load(file)

    old = saved["weights"]

    if old.shape != (OLD_DIM,):
        raise SystemExit(f"{source} holds shape {old.shape}, expected {(OLD_DIM,)}.")

    new = np.zeros(FEATURE_DIM, dtype=np.float64)

    for destination_slice, source_slice in BLOCK_MAP:
        new[destination_slice] = old[source_slice]

    moved = sum(s.stop - s.start for _, s in BLOCK_MAP)
    assert moved == OLD_DIM, f"{moved} of {OLD_DIM} old weights mapped"
    assert all(new[i] == 0.0 for i in NEW_SLOTS)

    saved["weights"] = new

    with destination.open("wb") as file:
        pickle.dump(saved, file)

    print(f"{source}  ->  {destination}")
    print(f"  episodes_trained={saved.get('episodes_trained')}  "
          f"epsilon={saved.get('epsilon')}")
    print(f"  {OLD_DIM} weights carried over, slots {NEW_SLOTS} start at 0.0")


if __name__ == "__main__":
    main()
