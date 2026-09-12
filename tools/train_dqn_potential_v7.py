#!/usr/bin/env python3
"""Train one controlled Potential-Shaping V7 candidate from V3-75.

The script refuses an unexpected active model, starts from the preserved
V3-75 checkpoint, trains once on the fixed empty-board protocol, verifies that
only adapter columns 45-51 changed, saves the candidate, and restores V3-75.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch


BASELINE_SHA256 = (
    "944377133b551419cd0a01af50ba9983a04e04366574ec50e32a2c905df27b69"
)
PREVIOUS_FEATURE_DIM = 45
DEFAULT_WORLD_SEED = 4242
DEFAULT_EXPERIMENT_SEED = 60_013
DEFAULT_ROUNDS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_adapter_only_update(
    baseline_checkpoint: dict,
    candidate_checkpoint: dict,
) -> dict[str, bool]:
    """Verify that only input columns 45-51 changed in network states."""
    input_weight_name = "network.0.weight"
    adapter_changed = False

    for state_name in ("policy_state", "target_state"):
        baseline_state = baseline_checkpoint[state_name]
        candidate_state = candidate_checkpoint[state_name]

        if baseline_state.keys() != candidate_state.keys():
            raise RuntimeError(f"{state_name} parameter names changed")

        for parameter_name, baseline_value in baseline_state.items():
            candidate_value = candidate_state[parameter_name]

            if parameter_name == input_weight_name:
                if not torch.equal(
                    baseline_value[:, :PREVIOUS_FEATURE_DIM],
                    candidate_value[:, :PREVIOUS_FEATURE_DIM],
                ):
                    raise RuntimeError(
                        f"Protected {state_name} input columns changed"
                    )
                if not torch.equal(
                    baseline_value[:, PREVIOUS_FEATURE_DIM:],
                    candidate_value[:, PREVIOUS_FEATURE_DIM:],
                ):
                    adapter_changed = True
            elif not torch.equal(baseline_value, candidate_value):
                raise RuntimeError(
                    f"Protected parameter changed: "
                    f"{state_name}.{parameter_name}"
                )

    if not adapter_changed:
        raise RuntimeError("Training did not change the safe-attack adapter")

    return {
        "features_0_44_unchanged": True,
        "later_layers_unchanged": True,
        "adapter_45_51_changed": True,
    }


def _write_json(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--world-seed",
        type=int,
        default=DEFAULT_WORLD_SEED,
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=DEFAULT_EXPERIMENT_SEED,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimization/potential_shaping_v7"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing V7 candidate in the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.world_seed < 0 or args.experiment_seed < 0:
        raise ValueError("Seeds must be non-negative")

    root = Path.cwd().resolve()
    baseline_path = (
        root / "checkpoints/dqn/selected_dqn_v3_scale075.pt"
    )
    model_path = root / "agent_code/dqn_agent/dqn_model.pt"
    wrapper_path = root / "tools/run_seeded_bomberman.py"
    output_dir = (root / args.output_dir).resolve()
    candidate_path = output_dir / "candidate.pt"
    training_stats_path = output_dir / "training.json"
    training_log_path = output_dir / "training.log"
    summary_path = output_dir / "summary.json"

    for path in (baseline_path, model_path, wrapper_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if _sha256(baseline_path) != BASELINE_SHA256:
        raise RuntimeError("The protected V3-75 checkpoint hash is wrong")
    if _sha256(model_path) != BASELINE_SHA256:
        raise RuntimeError(
            "The active model is not the protected V3-75 checkpoint"
        )
    if candidate_path.exists() and not args.force:
        raise FileExistsError(
            f"{candidate_path} already exists; use --force to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = _load_checkpoint(baseline_path)

    environment = os.environ.copy()
    environment["BOMBERMAN_DQN_DEVICE"] = "cpu"
    environment["PYTHONHASHSEED"] = str(args.experiment_seed)
    environment.pop("BOMBERMAN_TRACE", None)

    command = [
        sys.executable,
        str(wrapper_path),
        "--experiment-seed",
        str(args.experiment_seed),
        "--",
        "play",
        "--agents",
        "dqn_agent",
        "rule_based_agent",
        "rule_based_agent",
        "rule_based_agent",
        "--train",
        "1",
        "--scenario",
        "empty",
        "--seed",
        str(args.world_seed),
        "--n-rounds",
        str(args.rounds),
        "--no-gui",
        "--save-stats",
        str(training_stats_path),
    ]

    try:
        shutil.copy2(baseline_path, model_path)
        with training_log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

        candidate_checkpoint = _load_checkpoint(model_path)
        if (
            candidate_checkpoint.get("algorithm")
            != "potential_shaping_double_dqn_v7"
        ):
            raise RuntimeError("Training did not produce a V7 checkpoint")
        if int(candidate_checkpoint.get("feature_dim", -1)) != 52:
            raise RuntimeError("V7 checkpoint feature dimension changed")
        if not training_stats_path.is_file():
            raise RuntimeError("Training statistics were not created")
        verification = verify_adapter_only_update(
            baseline_checkpoint,
            candidate_checkpoint,
        )
        shutil.copy2(model_path, candidate_path)
        candidate_hash = _sha256(candidate_path)

        summary = {
            "algorithm": "potential_shaping_double_dqn_v7",
            "baseline": str(baseline_path),
            "baseline_sha256": BASELINE_SHA256,
            "candidate": str(candidate_path),
            "candidate_sha256": candidate_hash,
            "experiment_seed": args.experiment_seed,
            "rounds": args.rounds,
            "scenario": "empty",
            "verification": verification,
            "world_seed": args.world_seed,
        }
        _write_json(summary_path, summary)

        print("Potential-Shaping V7 training completed.")
        print(f"Candidate: {candidate_path}")
        print(f"SHA-256: {candidate_hash}")
        print(f"Summary: {summary_path}")
    finally:
        shutil.copy2(baseline_path, model_path)

    restored = _sha256(model_path) == BASELINE_SHA256
    print(f"V3-75 restored: {restored}")
    if not restored:
        raise RuntimeError("Failed to restore the protected V3-75 model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
