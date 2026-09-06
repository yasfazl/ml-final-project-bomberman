"""
Print the trained weight vector as a readable table.

    python -m agent_code.q_learning_agent_task2_phi.show_weights
    python -m agent_code.q_learning_agent_task2_phi.show_weights path/to/q_model.pkl

The move block is one shared row, so a direction block prints as four numbers
[ahead, right, behind, left] instead of the 4x4 grid the per-action-row model
needed. If 'right', 'behind' and 'left' all sit near zero for a block, that
block can be collapsed to a single "is it ahead?" feature.
"""
import pickle
import sys

import numpy as np

from .callbacks import (
    BOMB_OFFSET,
    FEATURE_DIM,
    MODEL_PATH,
    USE_DANGER_AHEAD,
    USE_INTERACTIONS,
    WAIT_OFFSET,
)

SLOTS = ["ahead", "right", "behind", "left"]

# "can walk this way" is not here: its ahead slot was collinear with the move
# bias and has been removed, so the block is now three slots and prints below.
DIRECTION_BLOCKS = [
    ("BFS target this way", 4),
    ("explosion this way", 8),
    ("live bomb this way", 12),
    ("escape route this way", 16),
]

# right / behind / left only.
WALK_BLOCK = ("can walk this way", 1)

MOVE_SCALARS = [
    ("coin visible", 20),
    ("target distance", 21),
    ("danger", 22),
    ("crate_eff", 23),
    ("trap", 24),
]

INTERACTIONS = [
    ("danger x escape-ahead", 25),
    ("target dist x target-ahead", 26),
]

WAIT_NAMES = ["bias", "n_valid/4", "coin visible", "target distance", "danger",
              "crate_eff", "escape exists", "n_explosion/4"]

BOMB_NAMES = ["bias", "n_valid/4", "crate_eff", "trap", "coin visible"]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH

    with open(path, "rb") as file:
        saved = pickle.load(file)

    w = saved["weights"]

    if w.shape != (FEATURE_DIM,):
        raise SystemExit(
            f"{path} holds weights of shape {w.shape}, not {(FEATURE_DIM,)}. "
            "A (6, 22) file belongs to the previous per-action-row agent."
        )

    print(f"{path}")
    print(f"episodes_trained={saved.get('episodes_trained')}  "
          f"epsilon={saved.get('epsilon'):.4f}  "
          f"USE_INTERACTIONS={USE_INTERACTIONS}  "
          f"USE_DANGER_AHEAD={USE_DANGER_AHEAD}\n")

    print("MOVE block  (shared by UP / RIGHT / DOWN / LEFT)")
    print(f"  {'':26s}" + "".join(f"{s:>9}" for s in SLOTS))

    name, start = WALK_BLOCK
    block = w[start:start + 3]
    print(f"  {name:26s}{'':9s}" + "".join(f"{v:9.2f}" for v in block)
          + "   (ahead slot removed: collinear with bias)")

    for name, start in DIRECTION_BLOCKS:
        block = w[start:start + 4]
        tail = np.abs(block[1:]).max()
        flag = "   <- slots 1-3 near zero" if tail < 0.5 else ""
        print(f"  {name:26s}" + "".join(f"{v:9.2f}" for v in block) + flag)

    print(f"\n  {'bias':26s}{w[0]:9.2f}")
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
