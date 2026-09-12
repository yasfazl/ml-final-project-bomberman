#!/usr/bin/env python3
"""Summarize one or more DQN death-diagnostic JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load_deaths(paths: list[Path]) -> list[dict]:
    deaths = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if record.get("record_type") == "death":
                    deaths.append(record)
    return deaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    deaths = load_deaths(args.paths)
    categories = Counter(
        death["classification"]["category"]
        for death in deaths
    )
    self_kills = [
        death for death in deaths
        if death["classification"]["self_kill"]
    ]
    opponent_kills = [
        death for death in deaths
        if death["classification"]["opponent_kill"]
    ]
    overlapping = sum(
        bool(death["classification"]["overlapping_lethal_blasts"])
        for death in deaths
    )
    unknown = sum(
        death["classification"]["category"] == "unknown"
        for death in deaths
    )

    print("DQN DEATH DIAGNOSTICS")
    print("files:", len(args.paths))
    print("recorded deaths:", len(deaths))
    print("own-bomb deaths:", len(self_kills))
    print("opponent-bomb deaths:", len(opponent_kills))
    print("overlapping lethal blasts:", overlapping)
    print("unknown deaths:", unknown)

    print("\nCATEGORIES")
    for category, count in categories.most_common():
        print(f"{category}: {count}")

    print("\nOWN-BOMB FLAGS")
    flags = [
        "reentered_own_blast",
        "unsafe_post_bomb_action_seen",
        "later_opponent_threat_seen",
    ]
    for flag in flags:
        count = sum(
            bool(death["classification"].get(flag))
            for death in self_kills
        )
        print(f"{flag}: {count}")

    if args.details:
        print("\nOWN-BOMB DETAILS")
        for death in self_kills:
            classification = death["classification"]
            print(
                f"seed={death['seed']} round={death['round']} "
                f"step={death['step']} category={classification['category']} "
                f"placed_step={classification['own_bomb_placed_step']} "
                f"age={classification['own_bomb_age']} "
                f"last_action={death['last_action']}"
            )


if __name__ == "__main__":
    main()
