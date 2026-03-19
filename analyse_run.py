#!/usr/bin/env python3
"""Analyse a per-episode JSONL log produced by train.py.

Usage:
    python analyse_run.py episode_log.jsonl
    python analyse_run.py episode_log.jsonl --tail 50   # focus on last N episodes
    python analyse_run.py episode_log.jsonl --compare other_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(path: str) -> List[Dict[str, Any]]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return episodes


def fmt(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}"


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def window_mean(values: List[float], n: int) -> Optional[float]:
    if not values:
        return None
    chunk = values[-n:]
    return mean(chunk)


def trend(values: List[float], window: int = 20) -> str:
    if len(values) < window * 2:
        return "insufficient data"
    early = mean(values[:window])
    late = mean(values[-window:])
    delta = late - early
    if abs(delta) < 0.01 * max(abs(early), abs(late), 1):
        return "flat"
    direction = "improving" if delta > 0 else "declining"
    return f"{direction} ({early:+.1f} → {late:+.1f})"


# ── section printers ─────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def section_overview(eps: List[Dict]) -> None:
    print_header("OVERVIEW")
    total = len(eps)
    timesteps = eps[-1].get("timestep", "?") if eps else "?"
    print(f"  Episodes    : {total}")
    print(f"  Timesteps   : {timesteps}")

    term_counts: Dict[str, int] = {}
    for e in eps:
        r = e.get("term_reason", "unknown")
        term_counts[r] = term_counts.get(r, 0) + 1
    print("  Term reasons:")
    for reason, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20}: {count:>5}  ({count / total * 100:.1f}%)")


def section_rewards(eps: List[Dict], window: int = 20) -> None:
    print_header("REWARDS")
    fields = [
        ("total_reward",    "Total"),
        ("objective_reward","Objective"),
        ("hazard_reward",   "Hazard"),
        ("pathing_reward",  "Pathing"),
        ("repeat_penalty",  "Repeat penalty"),
        ("walk_reward",     "Walk reward"),
    ]
    rewards_total = [e.get("total_reward", 0.0) for e in eps]
    for key, label in fields:
        vals = [e.get(key, 0.0) for e in eps]
        if not vals:
            continue
        recent_mean = window_mean(vals, window)
        t = trend(vals, window)
        print(f"  {label:<18}: mean={fmt(mean(vals)):>8}  "
              f"recent({window})={fmt(recent_mean):>8}  trend={t}")


def section_keys(eps: List[Dict], window: int = 20) -> None:
    print_header("KEY COLLECTION")
    collected = [e.get("keys_collected", 0) for e in eps]
    any_keys = [e for e in eps if e.get("keys_collected", 0) > 0]
    configured = [e.get("configured_keys", 0) for e in eps if e.get("configured_keys", 0) > 0]

    total_eps = len(eps)
    rate = len(any_keys) / total_eps if total_eps else 0.0
    recent_any = [1 if e.get("keys_collected", 0) > 0 else 0 for e in eps]
    recent_rate = window_mean(recent_any, window) or 0.0

    print(f"  Episodes with ≥1 key : {len(any_keys):>5}  ({pct(rate)} of all)")
    print(f"  Recent({window}) key rate : {pct(recent_rate)}")
    if configured:
        avg_cfg = mean(configured)
        avg_col = mean(collected)
        print(f"  Avg keys per episode : {fmt(avg_col, 2)} / {fmt(avg_cfg, 1)} configured")
    if any_keys:
        max_col = max(e.get("keys_collected", 0) for e in eps)
        print(f"  Max keys in one ep   : {max_col}")

    # trend
    t = trend([float(v) for v in collected], window)
    print(f"  Key collection trend : {t}")


def section_deaths(eps: List[Dict], window: int = 20) -> None:
    print_header("DEATHS & SAFETY")
    lives_lost = [e.get("life_losses", 0) for e in eps]
    safety = [e.get("safety_triggers", 0) for e in eps]
    jump_gate = [e.get("jump_gate_blocks", 0) for e in eps]
    steps = [e.get("steps", 1) for e in eps]

    total_deaths = sum(lives_lost)
    deaths_per_ep = mean(lives_lost) if lives_lost else 0.0
    recent_deaths = window_mean(lives_lost, window) or 0.0

    safety_per_step = [s / max(st, 1) for s, st in zip(safety, steps)]
    recent_safety_rate = window_mean(safety_per_step, window) or 0.0

    print(f"  Total deaths          : {total_deaths}")
    print(f"  Deaths/episode (mean) : {fmt(deaths_per_ep, 2)}")
    print(f"  Deaths/ep  recent({window}): {fmt(recent_deaths, 2)}")
    print(f"  Shield triggers/step  : {fmt(mean(safety_per_step) * 100, 1)}%  "
          f"recent({window})={fmt(recent_safety_rate * 100, 1)}%")
    t = trend([float(v) for v in lives_lost], window)
    print(f"  Death count trend     : {t}")


def section_exploration(eps: List[Dict], window: int = 20) -> None:
    print_header("EXPLORATION")
    coverage = [e.get("coverage_ratio", 0.0) for e in eps]
    ep_steps = [e.get("steps", 0) for e in eps]

    if coverage:
        recent_cov = window_mean(coverage, window) or 0.0
        t = trend(coverage, window)
        print(f"  Coverage ratio (mean)   : {pct(mean(coverage))}")
        print(f"  Coverage recent({window})    : {pct(recent_cov)}")
        print(f"  Coverage trend          : {t}")
    if ep_steps:
        print(f"  Episode length (mean)   : {fmt(mean(ep_steps), 0)} steps")
        print(f"  Episode length recent({window}): {fmt(window_mean(ep_steps, window) or 0, 0)} steps")


def section_diagnosis(eps: List[Dict], window: int = 20) -> None:
    """Identify the most likely current problem and suggest a fix."""
    print_header("DIAGNOSIS & RECOMMENDATIONS")

    if len(eps) < 10:
        print("  Too few episodes for diagnosis.")
        return

    recent = eps[-window:]
    key_rate = mean(1 if e.get("keys_collected", 0) > 0 else 0 for e in recent)
    deaths_per_ep = mean(e.get("life_losses", 0) for e in recent)
    coverage = mean(e.get("coverage_ratio", 0.0) for e in recent)
    safety_per_step = mean(
        e.get("safety_triggers", 0) / max(e.get("steps", 1), 1) for e in recent
    )
    steps = mean(e.get("steps", 1) for e in recent)
    repeat_pen = mean(abs(e.get("repeat_penalty", 0.0)) for e in recent)
    pathing = mean(e.get("pathing_reward", 0.0) for e in recent)

    issues = []

    if key_rate == 0.0:
        issues.append((10, "No keys collected in recent episodes — agent is not reaching any key."))
    elif key_rate < 0.05:
        issues.append((8, f"Key collection rare ({pct(key_rate)} of recent episodes)."))

    if deaths_per_ep > 3:
        issues.append((7, f"High death rate ({fmt(deaths_per_ep, 1)}/ep) — safety rules may be insufficient or agent is stuck in dangerous areas."))

    if safety_per_step > 0.25:
        issues.append((5, f"Safety shield triggering on {pct(safety_per_step)} of steps — agent is repeatedly attempting blocked actions."))
    elif safety_per_step < 0.005 and deaths_per_ep > 1:
        issues.append((4, "Shield rarely triggers but deaths are high — shield may not be covering the dangerous actions."))

    if coverage < 0.15:
        issues.append((6, f"Low screen coverage ({pct(coverage)}) — agent is not exploring; consider raising --pathing-reward."))
    elif coverage > 0.6 and key_rate < 0.1:
        issues.append((3, "Good coverage but no keys — agent is exploring but not targeting keys. Check key cell features in obs."))

    if repeat_pen > 5.0:
        issues.append((5, f"High repeat-transition penalty ({fmt(repeat_pen, 1)}/ep) — agent is oscillating. Consider raising REPEAT_TRANSITION_PENALTY or its max."))

    if steps < 100:
        issues.append((9, f"Very short episodes ({fmt(steps, 0)} steps) — agent dies almost immediately. Focus on survival first."))

    if not issues:
        if key_rate > 0.5:
            print("  Progress looks healthy — key collection rate is reasonable.")
            print("  Consider: increase --timesteps, check if agent is reaching the exit.")
        else:
            print(f"  No critical issues detected. Key rate {pct(key_rate)}, deaths {fmt(deaths_per_ep, 1)}/ep.")
        return

    issues.sort(key=lambda x: -x[0])
    print(f"  Most likely issues (last {window} episodes):\n")
    for priority, msg in issues:
        print(f"  [{priority}] {msg}")

    print("\n  Suggested next steps:")
    top = issues[0][1]
    if "No keys" in top or "Key collection rare" in top:
        print("   • Verify key cell features reach the observation (debug obs vector).")
        print("   • Increase --pathing-reward to push exploration towards keys.")
        print("   • Check safety shield is not blocking the jump onto or near the key.")
    if "death rate" in top.lower():
        print("   • Review recent shield trigger reasons in the log (safety_reason field).")
        print("   • Check that landing-position checks are active for fixed lethals.")
    if "oscillating" in top.lower() or "repeat" in top.lower():
        print("   • Raise REPEAT_TRANSITION_PENALTY_MAX in manic_play.py.")
        print("   • Check jump gate is not trapping the agent between two walls.")
    if "coverage" in top.lower() and "low" in top.lower():
        print("   • Raise --pathing-reward (currently drives exploration).")
        print("   • Verify visited_cells is cleared on episode reset.")


# ── main ─────────────────────────────────────────────────────────────────────

def analyse(eps: List[Dict], label: str, tail: Optional[int] = None, window: int = 20) -> None:
    if tail and tail < len(eps):
        eps = eps[-tail:]
        label = f"{label} (last {tail} episodes)"
    print(f"\n{'═' * 60}")
    print(f"  ANALYSIS: {label}")
    print(f"{'═' * 60}")
    if not eps:
        print("  No episodes found.")
        return
    section_overview(eps)
    section_rewards(eps, window)
    section_keys(eps, window)
    section_deaths(eps, window)
    section_exploration(eps, window)
    section_diagnosis(eps, window)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Manic Miner episode logs")
    parser.add_argument("log", help="Path to episode_log.jsonl")
    parser.add_argument("--tail", type=int, default=0,
                        help="Only analyse the last N episodes")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling window size for recent-trend stats (default 20)")
    parser.add_argument("--compare", default="",
                        help="Path to a second log file to compare against")
    args = parser.parse_args()

    eps = load_log(args.log)
    if not eps:
        print(f"No episodes found in {args.log}")
        sys.exit(1)

    tail = args.tail if args.tail > 0 else None
    analyse(eps, args.log, tail=tail, window=args.window)

    if args.compare:
        eps2 = load_log(args.compare)
        if eps2:
            analyse(eps2, args.compare, tail=tail, window=args.window)
        else:
            print(f"No episodes found in {args.compare}")


if __name__ == "__main__":
    main()
