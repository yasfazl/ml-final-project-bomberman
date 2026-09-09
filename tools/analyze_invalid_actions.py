#!/usr/bin/env python3
"""Summarize JSONL files produced by invalid-action diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {error}"
                    ) from error
    return records


def print_counter(title: str, values: Counter) -> None:
    print(f"\n{title}")
    for name, count in values.most_common():
        print(f"{name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    records = load_records(args.files)
    print("INVALID ACTION DIAGNOSTICS")
    print("files:", len(args.files))
    print("recorded invalid actions:", len(records))
    if not records:
        return

    categories = Counter(record["category"] for record in records)
    actions = Counter(record["action"] for record in records)
    modes = Counter(record["mode"] for record in records)
    scenarios = Counter(str(record.get("scenario")) for record in records)
    print_counter("CATEGORIES", categories)
    print_counter("ACTIONS", actions)
    print_counter("MODES", modes)
    print_counter("SCENARIOS", scenarios)

    movement_records = [
        record
        for record in records
        if record["action"] in {"UP", "RIGHT", "DOWN", "LEFT"}
    ]
    collision_count = categories["simultaneous_agent_collision"]
    print("\nKEY RATIOS")
    print(
        "simultaneous collisions / all invalids:",
        f"{collision_count / len(records):.1%}",
    )
    if movement_records:
        print(
            "simultaneous collisions / movement invalids:",
            f"{collision_count / len(movement_records):.1%}",
        )
    endgame_count = modes["endgame"]
    print(
        "endgame invalids / all invalids:",
        f"{endgame_count / len(records):.1%}",
    )

    if args.details:
        print("\nFIRST 30 RECORDS")
        for record in records[:30]:
            print(
                f"seed={record.get('seed')} "
                f"round={record.get('round')} "
                f"step={record.get('step')} "
                f"mode={record.get('mode')} "
                f"action={record.get('action')} "
                f"category={record.get('category')} "
                f"destination={record.get('destination')} "
                f"occupants={record.get('execution_occupants')}"
            )


if __name__ == "__main__":
    main()
