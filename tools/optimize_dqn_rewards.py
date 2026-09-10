#!/usr/bin/env python3
"""Bounded, resumable Bayesian reward search for the V3 safe-attack adapter.

Every candidate starts from the same preserved V3-75 checkpoint.  The script
trains sequentially, evaluates on a fixed development panel, records the
Optuna study in SQLite, and restores the user's working model on exit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping, Sequence


DEFAULT_PARAMS = {
    "progress_reward": 0.5,
    "wait_penalty": 1.0,
    "targeted_bomb_reward": 4.0,
}
SEARCH_BOUNDS = {
    "progress_reward": (0.1, 1.0),
    "wait_penalty": (0.25, 2.0),
    "targeted_bomb_reward": (0.5, 4.0),
}
REWARD_ENV_NAMES = {
    "progress_reward": "BOMBERMAN_DQN_SAFE_ATTACK_PROGRESS_REWARD",
    "wait_penalty": "BOMBERMAN_DQN_SAFE_ATTACK_WAIT_PENALTY",
    "targeted_bomb_reward": (
        "BOMBERMAN_DQN_TARGETED_OPPONENT_BOMB_REWARD"
    ),
}
FRESH_OPTIMIZER_ENV = "BOMBERMAN_DQN_FRESH_OPTIMIZER"
REPLAY_SEED_ENV = "BOMBERMAN_DQN_REPLAY_SEED"
DEFAULT_DEV_SEEDS = (1103, 2207, 3301, 4409, 5501)
TRACKED_METRICS = (
    "score",
    "kills",
    "suicides",
    "bombs",
    "invalid",
    "steps",
)


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_agent_run(path: Path) -> dict:
    data = _read_json(path)
    try:
        return data["by_agent"]["dqn_agent"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"No dqn_agent statistics in {path}."
        ) from error


def aggregate_runs(
    runs: Sequence[Mapping[str, float]],
    rounds_per_run: int,
) -> dict:
    """Aggregate raw statistics and normalize totals to 100 rounds."""
    if not runs:
        raise ValueError("At least one evaluation run is required.")
    if rounds_per_run <= 0:
        raise ValueError("rounds_per_run must be positive.")

    totals = {
        metric: float(sum(float(run.get(metric, 0) or 0) for run in runs))
        for metric in TRACKED_METRICS
    }
    total_rounds = rounds_per_run * len(runs)
    round_scale = 100.0 / total_rounds
    result = {
        f"{metric}_per_100_rounds": totals[metric] * round_scale
        for metric in TRACKED_METRICS
        if metric != "steps"
    }
    result["steps_total"] = totals["steps"]
    result["invalid_per_1000_steps"] = (
        1000.0 * totals["invalid"] / totals["steps"]
        if totals["steps"] > 0
        else 0.0
    )
    result["kills_per_100_bombs"] = (
        100.0 * totals["kills"] / totals["bombs"]
        if totals["bombs"] > 0
        else 0.0
    )
    result["total_rounds"] = total_rounds
    result["raw_totals"] = totals
    return result


def aggregate_run_files(
    paths: Iterable[Path],
    rounds_per_run: int,
) -> dict:
    return aggregate_runs(
        [load_agent_run(path) for path in paths],
        rounds_per_run,
    )


def constraint_values(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    suicide_tolerance: float,
    efficiency_retention: float,
) -> dict[str, float]:
    """Return values that are feasible exactly when all are <= zero."""
    return {
        "score_retention": (
            float(baseline["score_per_100_rounds"])
            - float(candidate["score_per_100_rounds"])
        ),
        "suicide_control": (
            float(candidate["suicides_per_100_rounds"])
            - float(baseline["suicides_per_100_rounds"])
            - suicide_tolerance
        ),
        "kill_efficiency": (
            efficiency_retention
            * float(baseline["kills_per_100_bombs"])
            - float(candidate["kills_per_100_bombs"])
        ),
    }


def constraints_are_feasible(values: Mapping[str, float]) -> bool:
    return bool(values) and all(float(value) <= 0.0 for value in values.values())


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in REWARD_ENV_NAMES.values():
        environment.pop(name, None)
    environment.pop(FRESH_OPTIMIZER_ENV, None)
    environment.pop(REPLAY_SEED_ENV, None)
    environment.pop("BOMBERMAN_TRACE", None)
    environment["BOMBERMAN_DQN_DEVICE"] = "cpu"
    return environment


def _seeded_command(
    python: str,
    wrapper: Path,
    experiment_seed: int,
    game_arguments: Sequence[str],
) -> list[str]:
    return [
        python,
        str(wrapper),
        "--experiment-seed",
        str(experiment_seed),
        "--",
        *game_arguments,
    ]


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    model_path: Path,
    repo_root: Path,
    python: str,
    wrapper: Path,
    destination: Path,
    seeds: Sequence[int],
    rounds: int,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, model_path)
    stats_paths = []
    environment = _base_environment()

    for index, world_seed in enumerate(seeds):
        stats_path = destination / f"seed{world_seed}.json"
        log_path = destination / f"seed{world_seed}.log"
        experiment_seed = 900_001 + index * 10_007
        environment["PYTHONHASHSEED"] = str(experiment_seed)
        command = _seeded_command(
            python,
            wrapper,
            experiment_seed,
            [
                "play",
                "--agents",
                "dqn_agent",
                "rule_based_agent",
                "rule_based_agent",
                "rule_based_agent",
                "--scenario",
                "empty",
                "--seed",
                str(world_seed),
                "--n-rounds",
                str(rounds),
                "--no-gui",
                "--save-stats",
                str(stats_path),
            ],
        )
        _run_command(
            command,
            cwd=repo_root,
            environment=environment,
            log_path=log_path,
        )
        stats_paths.append(stats_path)

    metrics = aggregate_run_files(stats_paths, rounds)
    _write_json(destination / "metrics.json", metrics)
    return metrics


class RewardObjective:
    def __init__(
        self,
        *,
        baseline_checkpoint: Path,
        baseline_metrics: Mapping[str, float],
        model_path: Path,
        repo_root: Path,
        python: str,
        wrapper: Path,
        output_dir: Path,
        train_rounds: int,
        eval_rounds: int,
        dev_seeds: Sequence[int],
        training_world_seed: int,
        training_experiment_seed: int,
        suicide_tolerance: float,
        efficiency_retention: float,
    ):
        self.baseline_checkpoint = baseline_checkpoint
        self.baseline_metrics = baseline_metrics
        self.model_path = model_path
        self.repo_root = repo_root
        self.python = python
        self.wrapper = wrapper
        self.output_dir = output_dir
        self.train_rounds = train_rounds
        self.eval_rounds = eval_rounds
        self.dev_seeds = tuple(dev_seeds)
        self.training_world_seed = training_world_seed
        self.training_experiment_seed = training_experiment_seed
        self.suicide_tolerance = suicide_tolerance
        self.efficiency_retention = efficiency_retention

    def __call__(self, trial) -> float:
        params = {
            name: trial.suggest_float(name, low, high)
            for name, (low, high) in SEARCH_BOUNDS.items()
        }
        trial_dir = self.output_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        _write_json(trial_dir / "params.json", params)

        shutil.copy2(self.baseline_checkpoint, self.model_path)
        environment = _base_environment()
        for name, value in params.items():
            environment[REWARD_ENV_NAMES[name]] = repr(float(value))
        environment[FRESH_OPTIMIZER_ENV] = "1"
        environment[REPLAY_SEED_ENV] = str(self.training_experiment_seed)
        environment["PYTHONHASHSEED"] = str(
            self.training_experiment_seed
        )

        training_stats = trial_dir / "training.json"
        training_command = _seeded_command(
            self.python,
            self.wrapper,
            self.training_experiment_seed,
            [
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
                str(self.training_world_seed),
                "--n-rounds",
                str(self.train_rounds),
                "--no-gui",
                "--save-stats",
                str(training_stats),
            ],
        )
        _run_command(
            training_command,
            cwd=self.repo_root,
            environment=environment,
            log_path=trial_dir / "training.log",
        )

        if not self.model_path.is_file():
            raise RuntimeError("Training did not produce dqn_model.pt.")
        candidate_checkpoint = trial_dir / "candidate.pt"
        shutil.copy2(self.model_path, candidate_checkpoint)

        metrics = evaluate_checkpoint(
            checkpoint=candidate_checkpoint,
            model_path=self.model_path,
            repo_root=self.repo_root,
            python=self.python,
            wrapper=self.wrapper,
            destination=trial_dir / "evaluation",
            seeds=self.dev_seeds,
            rounds=self.eval_rounds,
        )
        constraints = constraint_values(
            metrics,
            self.baseline_metrics,
            suicide_tolerance=self.suicide_tolerance,
            efficiency_retention=self.efficiency_retention,
        )
        for name, value in constraints.items():
            trial.set_constraint(name, float(value))

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("constraints", constraints)
        trial.set_user_attr("candidate_checkpoint", str(candidate_checkpoint))
        result = {
            "params": params,
            "metrics": metrics,
            "constraints": constraints,
            "feasible": constraints_are_feasible(constraints),
        }
        _write_json(trial_dir / "result.json", result)

        score = float(metrics["score_per_100_rounds"])
        print(
            f"Trial {trial.number}: score/100={score:.2f}, "
            f"suicides/100={metrics['suicides_per_100_rounds']:.2f}, "
            f"kills/100 bombs={metrics['kills_per_100_bombs']:.3f}, "
            f"feasible={result['feasible']}",
            flush=True,
        )
        return score


def _trial_constraints(trial) -> dict[str, float]:
    explicit = getattr(trial, "constraints", None)
    if explicit:
        return {name: float(value) for name, value in explicit.items()}
    stored = trial.user_attrs.get("constraints", {})
    return {name: float(value) for name, value in stored.items()}


def export_study(study, output_dir: Path, baseline_metrics: Mapping) -> dict:
    try:
        from optuna.trial import TrialState
    except ImportError as error:
        raise RuntimeError("Optuna is required to export the study.") from error

    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]
    feasible = [
        trial
        for trial in completed
        if constraints_are_feasible(_trial_constraints(trial))
    ]
    best = max(
        feasible,
        key=lambda trial: (
            float(trial.value),
            -float(
                trial.user_attrs.get("metrics", {}).get(
                    "suicides_per_100_rounds",
                    float("inf"),
                )
            ),
        ),
        default=None,
    )

    csv_path = output_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "trial",
                "state",
                "score_per_100_rounds",
                "feasible",
                *DEFAULT_PARAMS,
                "suicides_per_100_rounds",
                "kills_per_100_bombs",
                "invalid_per_1000_steps",
                "checkpoint",
            ]
        )
        for trial in study.trials:
            metrics = trial.user_attrs.get("metrics", {})
            constraints = _trial_constraints(trial)
            writer.writerow(
                [
                    trial.number,
                    trial.state.name,
                    "" if trial.value is None else trial.value,
                    constraints_are_feasible(constraints),
                    *[trial.params.get(name, "") for name in DEFAULT_PARAMS],
                    metrics.get("suicides_per_100_rounds", ""),
                    metrics.get("kills_per_100_bombs", ""),
                    metrics.get("invalid_per_1000_steps", ""),
                    trial.user_attrs.get("candidate_checkpoint", ""),
                ]
            )

    summary = {
        "baseline_metrics": dict(baseline_metrics),
        "completed_trials": len(completed),
        "feasible_trials": len(feasible),
        "best_trial": None,
        "decision": "keep_v3_75",
    }
    if best is not None:
        source = Path(best.user_attrs["candidate_checkpoint"])
        selected = output_dir / "best_feasible.pt"
        shutil.copy2(source, selected)
        summary["best_trial"] = {
            "number": best.number,
            "objective": best.value,
            "params": best.params,
            "metrics": best.user_attrs.get("metrics", {}),
            "constraints": _trial_constraints(best),
            "checkpoint": str(selected),
        }
        summary["decision"] = "validate_best_feasible_on_six_seeds"

    _write_json(output_dir / "study_summary.json", summary)
    return summary


def _resolve_from_repo(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _ensure_manifest(path: Path, expected: Mapping) -> None:
    if path.is_file():
        found = _read_json(path)
        if found != expected:
            raise RuntimeError(
                f"Existing study configuration differs from {path}. "
                "Use a new --output-dir instead of mixing experiments."
            )
        return
    _write_json(path, expected)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded Optuna TPE search over three V3 rewards."
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/dqn/selected_dqn_v3_scale075.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="optimization/reward_bayes_v6",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--train-rounds", type=int, default=100)
    parser.add_argument("--eval-rounds", type=int, default=30)
    parser.add_argument(
        "--dev-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_DEV_SEEDS),
    )
    parser.add_argument("--training-world-seed", type=int, default=4242)
    parser.add_argument(
        "--training-experiment-seed",
        type=int,
        default=60013,
    )
    parser.add_argument("--sampler-seed", type=int, default=20260909)
    parser.add_argument("--suicide-tolerance", type=float, default=0.5)
    parser.add_argument("--efficiency-retention", type=float, default=0.95)
    args = parser.parse_args(argv)

    if not 1 <= args.trials <= 30:
        parser.error("--trials must be between 1 and 30")
    if args.train_rounds <= 0 or args.eval_rounds <= 0:
        parser.error("training and evaluation rounds must be positive")
    if not args.dev_seeds or len(set(args.dev_seeds)) != len(args.dev_seeds):
        parser.error("--dev-seeds must contain distinct values")
    if args.suicide_tolerance < 0:
        parser.error("--suicide-tolerance must be non-negative")
    if not 0 < args.efficiency_retention <= 1:
        parser.error("--efficiency-retention must be in (0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as error:
        raise SystemExit(
            "Optuna 5 is required. Install requirements-reward-search.txt."
        ) from error

    if not hasattr(optuna.trial.Trial, "set_constraint"):
        raise SystemExit("This experiment requires Optuna 5 or newer.")

    repo_root = Path(args.repo).expanduser().resolve()
    baseline_checkpoint = _resolve_from_repo(
        repo_root,
        args.baseline_checkpoint,
    )
    output_dir = _resolve_from_repo(repo_root, args.output_dir)
    model_path = repo_root / "agent_code" / "dqn_agent" / "dqn_model.pt"
    wrapper = repo_root / "tools" / "run_seeded_bomberman.py"

    for required in (repo_root / "main.py", baseline_checkpoint, wrapper):
        if not required.is_file():
            raise SystemExit(f"Required file not found: {required}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_sha256": _sha256(baseline_checkpoint),
        "dev_seeds": list(args.dev_seeds),
        "eval_rounds": args.eval_rounds,
        "train_rounds": args.train_rounds,
        "training_world_seed": args.training_world_seed,
        "training_experiment_seed": args.training_experiment_seed,
        "sampler_seed": args.sampler_seed,
        "suicide_tolerance": args.suicide_tolerance,
        "efficiency_retention": args.efficiency_retention,
        "search_bounds": {
            name: list(bounds) for name, bounds in SEARCH_BOUNDS.items()
        },
        "default_params": DEFAULT_PARAMS,
    }
    _ensure_manifest(output_dir / "manifest.json", manifest)

    baseline_dir = output_dir / "baseline"
    baseline_metrics_path = baseline_dir / "metrics.json"

    with tempfile.TemporaryDirectory(prefix="bomberman_reward_search_") as temp:
        backup = Path(temp) / "working_model.pt"
        working_model_existed = model_path.is_file()
        if working_model_existed:
            shutil.copy2(model_path, backup)

        try:
            if baseline_metrics_path.is_file():
                baseline_metrics = _read_json(baseline_metrics_path)
            else:
                print("Evaluating untouched V3-75 baseline...", flush=True)
                baseline_metrics = evaluate_checkpoint(
                    checkpoint=baseline_checkpoint,
                    model_path=model_path,
                    repo_root=repo_root,
                    python=args.python,
                    wrapper=wrapper,
                    destination=baseline_dir,
                    seeds=args.dev_seeds,
                    rounds=args.eval_rounds,
                )

            database = output_dir / "study.sqlite3"
            sampler = optuna.samplers.TPESampler(
                seed=args.sampler_seed,
                n_startup_trials=5,
                multivariate=True,
            )
            study = optuna.create_study(
                study_name="dqn_reward_bayes_v6",
                direction="maximize",
                sampler=sampler,
                storage=f"sqlite:///{database}",
                load_if_exists=True,
            )
            if not study.trials:
                study.enqueue_trial(DEFAULT_PARAMS)

            completed_count = sum(
                trial.state == TrialState.COMPLETE
                for trial in study.trials
            )
            remaining = max(0, args.trials - completed_count)
            objective = RewardObjective(
                baseline_checkpoint=baseline_checkpoint,
                baseline_metrics=baseline_metrics,
                model_path=model_path,
                repo_root=repo_root,
                python=args.python,
                wrapper=wrapper,
                output_dir=output_dir,
                train_rounds=args.train_rounds,
                eval_rounds=args.eval_rounds,
                dev_seeds=args.dev_seeds,
                training_world_seed=args.training_world_seed,
                training_experiment_seed=(
                    args.training_experiment_seed
                ),
                suicide_tolerance=args.suicide_tolerance,
                efficiency_retention=args.efficiency_retention,
            )
            if remaining:
                print(
                    f"Running {remaining} sequential trial(s); "
                    f"target total={args.trials}.",
                    flush=True,
                )
                study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=1,
                    gc_after_trial=True,
                )
            else:
                print("Requested trial budget is already complete.")

            summary = export_study(study, output_dir, baseline_metrics)
        finally:
            if working_model_existed:
                shutil.copy2(backup, model_path)
            elif model_path.exists():
                model_path.unlink()

    print("\nSEARCH SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWorking model restored: {model_path}")
    print(f"Study results: {output_dir / 'study_summary.json'}")


if __name__ == "__main__":
    main()
