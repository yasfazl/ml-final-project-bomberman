#!/usr/bin/env python3
"""Run Bomberman with deterministic experiment-level random seeds.

The framework's ``--seed`` controls the world generator, but the bundled
rule-based agents call ``numpy.random.seed()`` without an argument.  This
wrapper replaces those entropy reseeds with deterministic derived seeds and
also seeds Python, NumPy, PyTorch, and DQN replay sampling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np


UINT32_MODULUS = 2**32
SEED_STRIDE = 104_729


def configure_determinism(seed: int) -> None:
    """Seed experiment RNGs and control no-argument NumPy reseeding."""
    if seed < 0:
        raise ValueError("experiment seed must be non-negative")

    random.seed(seed)
    original_numpy_seed = np.random.seed
    original_numpy_seed(seed % UINT32_MODULUS)
    reseed_count = 0

    def deterministic_numpy_seed(value=None):
        nonlocal reseed_count
        if value is None:
            reseed_count += 1
            value = (seed + SEED_STRIDE * reseed_count) % UINT32_MODULUS
        return original_numpy_seed(value)

    np.random.seed = deterministic_numpy_seed
    os.environ.setdefault("BOMBERMAN_DQN_REPLAY_SEED", str(seed))

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main.py with fully controlled experiment RNGs."
    )
    parser.add_argument("--experiment-seed", required=True, type=int)
    parser.add_argument("game_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.game_arguments[:1] == ["--"]:
        args.game_arguments = args.game_arguments[1:]
    if not args.game_arguments:
        parser.error("Bomberman arguments are required after --")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_determinism(args.experiment_seed)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from main import main as bomberman_main

    bomberman_main(args.game_arguments)


if __name__ == "__main__":
    main()
