#!/usr/bin/env python3
"""Analyse a visual Manic Miner episode log (.jsonl)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_episodes(path: str, tail: int = 0) -> List[Dict[str, Any]]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if tail > 0:
        episodes = episodes[-tail:]
    return episodes


def fmt(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}"


def analyse(episodes: List[Dict[str, Any]]) -> None:
    if not episodes:
        print("No episodes found.")
        return

    n = len(episodes)
    scores = [e.get("score", 0) for e in episodes]
    rewards = [e.get("total_reward", 0.0) for e in episodes]
    steps = [e.get("steps", 0) for e in episodes]
    deaths = [e.get("deaths", 0) for e in episodes]

    first_ts = episodes[0].get("timestep", 0)
    last_ts = episodes[-1].get("timestep", 0)

    nonzero_score = [s for s in scores if s > 0]
    score_rate = len(nonzero_score) / n * 100

    print(f"Episodes analysed : {n}")
    print(f"Timestep range    : {first_ts:,} – {last_ts:,}")
    print()
    print(f"{'Metric':<26} {'Mean':>8}  {'Max':>8}  {'Min':>8}")
    print("─" * 55)

    def row(label, values, decimals=1):
        mean = sum(values) / len(values) if values else 0.0
        print(f"  {label:<24} {fmt(mean, decimals):>8}  {fmt(max(values), decimals):>8}  {fmt(min(values), decimals):>8}")

    row("Score", scores, decimals=0)
    row("Total reward", rewards, decimals=2)
    row("Steps", steps, decimals=0)
    row("Deaths per episode", deaths, decimals=2)

    print()
    print(f"  Episodes with score > 0 : {len(nonzero_score)} / {n}  ({fmt(score_rate)}%)")

    # Score buckets
    buckets = [(0, 0), (1, 99), (100, 199), (200, 499), (500, 999), (1000, 9999), (10000, 999999)]
    print()
    print("  Score distribution:")
    for lo, hi in buckets:
        count = sum(1 for s in scores if lo <= s <= hi)
        bar = "█" * min(40, int(count / n * 40))
        label = f"{lo}–{hi}" if lo > 0 else "0"
        print(f"    {label:>12} : {count:>5}  {bar}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse visual Manic Miner episode log")
    p.add_argument("log", nargs="?", default="episode_log.jsonl",
                   help="Path to the .jsonl episode log (default: episode_log.jsonl)")
    p.add_argument("--tail", type=int, default=0,
                   help="Analyse only the last N episodes (0 = all)")
    args = p.parse_args()

    if not Path(args.log).exists():
        print(f"Log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    episodes = load_episodes(args.log, tail=args.tail)
    analyse(episodes)


if __name__ == "__main__":
    main()
