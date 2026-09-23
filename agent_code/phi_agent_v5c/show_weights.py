"""
Print the trained weight vector as a readable table.

    python -m agent_code.q_learning_agent_task2_phi.show_weights
    python -m agent_code.q_learning_agent_task2_phi.show_weights path/to/q_model.pkl

The move block is one shared row, so a direction block prints as four numbers
[ahead, right, behind, left] instead of the 4x4 grid the per-action-row model
needed. If 'right', 'behind' and 'left' all sit near zero for a block, that
block can be collapsed to a single "is it ahead?" feature.

Layout is read from callbacks.py, so this stays in step with the feature set as
long as the name lists below match. It refuses to print a model whose length
does not match the current layout rather than silently mislabelling columns.
"""
import pickle
import sys

import numpy as np

from .callbacks import (
    BOMB_OFFSET,
    FEATURE_DIM,
    MODEL_PATH,
    MOVE_OFFSET,
    USE_DANGER_AHEAD,
    USE_INTERACTIONS,
    WAIT_OFFSET,
)

SLOTS = ["ahead", "right", "behind", "left"]

# (name, first index) for the four-slot direction blocks, in layout order.
DIRECTION_BLOCKS = [
    ("can walk this way", MOVE_OFFSET + 1),
    ("BFS target this way", MOVE_OFFSET + 5),
    ("explosion this way", MOVE_OFFSET + 9),
    ("live bomb this way", MOVE_OFFSET + 13),
    ("escape route this way", MOVE_OFFSET + 17),
]

MOVE_SCALARS = [
    ("bias", MOVE_OFFSET + 0),
    ("coin visible", MOVE_OFFSET + 21),
    ("target distance", MOVE_OFFSET + 22),
    ("danger", MOVE_OFFSET + 23),
    ("crate_eff", MOVE_OFFSET + 24),
    ("trap", MOVE_OFFSET + 25),
]

INTERACTIONS = [
    ("danger x escape-ahead", MOVE_OFFSET + 26),
    ("danger x explosion-ahead", MOVE_OFFSET + 27),
    ("target dist x target-ahead", MOVE_OFFSET + 28),
]

WAIT_NAMES = ["bias", "n_valid/4", "coin visible", "target distance", "danger",
              "crate_eff", "trap", "escape exists", "n_explosion/4"]

BOMB_NAMES = ["bias", "n_valid/4", "danger", "crate_eff", "trap",
              "coin visible", "escape exists"]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH

    with open(path, "rb") as file:
        saved = pickle.load(file)

    w = saved["weights"]

    if w.shape != (FEATURE_DIM,):
        raise SystemExit(
            f"{path} holds weights of shape {w.shape}, but callbacks.py is "
            f"currently laid out for {(FEATURE_DIM,)}. Printing it would put the "
            "wrong names on the columns; load the matching callbacks.py instead."
        )

    print(f"{path}")
    print(f"episodes_trained={saved.get('episodes_trained')}  "
          f"epsilon={saved.get('epsilon'):.4f}  "
          f"FEATURE_DIM={FEATURE_DIM}  "
          f"USE_INTERACTIONS={USE_INTERACTIONS}  "
          f"USE_DANGER_AHEAD={USE_DANGER_AHEAD}\n")

    print("MOVE block  (shared by UP / RIGHT / DOWN / LEFT)")
    print(f"  {'':26s}" + "".join(f"{s:>9}" for s in SLOTS))
    for name, start in DIRECTION_BLOCKS:
        block = w[start:start + 4]
        tail = np.abs(block[1:]).max()
        flag = "   <- slots 1-3 near zero" if tail < 0.5 else ""
        print(f"  {name:26s}" + "".join(f"{v:9.2f}" for v in block) + flag)

    print()
    for name, index in MOVE_SCALARS:
        print(f"  {name:26s}{w[index]:9.2f}")

    if USE_INTERACTIONS:
        print()
        for name, index in INTERACTIONS:
            print(f"  {name:26s}{w[index]:9.2f}")

    print("\nWAIT block")
    for offset, name in enumerate(WAIT_NAMES):
        print(f"  {name:26s}{w[WAIT_OFFSET + offset]:9.2f}")

    print("\nBOMB block")
    for offset, name in enumerate(BOMB_NAMES):
        print(f"  {name:26s}{w[BOMB_OFFSET + offset]:9.2f}")


if __name__ == "__main__":
    main()
