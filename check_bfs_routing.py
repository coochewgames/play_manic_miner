#!/usr/bin/env python3
"""Diagnostic: verify BFS routing through all 5 keys.

For each stage (all 5 keys present, then 1 collected, 2 collected, …) prints:
  - Willy's start cell
  - Which keys remain and their BFS distance from Willy's start
  - A composite ASCII map showing:
      #  solid platform
      ~  conveyor tile (floor of conveyor surface)
      X  nasty
      K  key position
      >  BFS waypoint (next step)
      W  Willy start
      .  free space

Also prints the conveyor tile positions detected from ROM so we can verify
they match the visible green belt on screen.

Usage:
    python check_bfs_routing.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

from manic_env_semantic import ManicMinerSemanticEnv, GAME_CELLS_H, GAME_CELLS_W
from pathfinder import find_waypoint, find_bfs_distance

FUSE_BIN    = "/Users/roddy/Dev/fuse/fuse/fuse"
FUSE_SOCKET = "/tmp/fuse-ml.sock"
FUSE_SPEED  = 800

W = GAME_CELLS_W
H = GAME_CELLS_H


# ── emulator helpers (same as check_semantic_obs.py) ──────────────────────────

def _readline_sock(sock: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("socket closed")
        buf += chunk
    return buf.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()


def launch_fuse() -> subprocess.Popen:
    if os.path.exists(FUSE_SOCKET):
        os.unlink(FUSE_SOCKET)
    env = os.environ.copy()
    env["FUSE_ML_MODE"]           = "1"
    env["FUSE_ML_SOCKET"]         = FUSE_SOCKET
    env["FUSE_ML_RESET_SNAPSHOT"] = "/Users/roddy/Dev/fuse/play_manic_miner/manicminer.szx"
    proc = subprocess.Popen(
        [FUSE_BIN, f"--speed={FUSE_SPEED}"],
        env=env,
        cwd=str(Path(FUSE_BIN).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            if not os.path.exists(FUSE_SOCKET):
                raise FileNotFoundError
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(FUSE_SOCKET)
            _readline_sock(s)
            s.sendall(b"PING\n")
            _readline_sock(s)
            s.close()
            print("Emulator ready.")
            return proc
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Emulator did not become ready in time.")


# ── display ────────────────────────────────────────────────────────────────────

def print_map(
    solid, nasty, conveyor,
    keys: list,
    willy: tuple,
    waypoint,
    title: str,
) -> None:
    print(f"\n  {title}")
    print(f"  # solid  ~ conveyor-tile  X nasty  K key  > waypoint  W willy  . free")
    print("  +" + "-" * W + "+")
    for row in range(H):
        line = []
        for col in range(W):
            ch = "."
            if solid[row, col]:    ch = "#"
            if conveyor[row, col]: ch = "~"
            if nasty[row, col]:    ch = "X"
            if (col, row) in keys: ch = "K"
            if waypoint and (col, row) == waypoint:   ch = ">"
            if (col, row) == willy:                   ch = "W"
            line.append(ch)
        print(f"  |{''.join(line)}|  y={row}")
    print("  +" + "-" * W + "+")
    print("   " + "".join(str(x % 10) for x in range(W)))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Launching emulator…")
    fuse_proc = launch_fuse()

    try:
        env = ManicMinerSemanticEnv(
            socket_path=FUSE_SOCKET,
            headless=False,
            warmup_steps=10,
        )
        _, _ = env.reset()

        solid    = env._solid_grid
        nasty    = env._nasty_grid
        conveyor = env._conveyor_tile_grid
        graph    = env._movement_graph
        keys     = list(env._active_keys)   # all 5, original order
        willy    = env._read_willy_cell()

        # ── 1. Conveyor tile summary ───────────────────────────────────────────
        conv_cells = [(x, y)
                      for y in range(H) for x in range(W)
                      if conveyor[y, x]]
        print("\n" + "=" * 66)
        print("  CONVEYOR TILES DETECTED FROM ROM")
        print("=" * 66)
        if conv_cells:
            ys = sorted(set(c[1] for c in conv_cells))
            for row in ys:
                xs = sorted(c[0] for c in conv_cells if c[1] == row)
                print(f"  y={row}: x={xs[0]}..{xs[-1]}  ({len(xs)} cells)")
        else:
            print("  NONE — conveyor attribute byte may not be matching any cell")

        # ── 2. Key reachability matrix ────────────────────────────────────────
        # Rows = source (Willy start + each key), Cols = destination keys.
        # Shows BFS distance; "---" = unreachable.
        all_positions = [("Willy", willy)] + [(f"K{i+1}({kx},{ky})", (kx, ky))
                                               for i, (kx, ky) in enumerate(keys)]
        print("\n" + "=" * 66)
        print("  REACHABILITY MATRIX  (BFS distances; --- = unreachable)")
        print("=" * 66)
        col_labels = [f"K{i+1}({kx},{ky})" for i, (kx, ky) in enumerate(keys)]
        header = f"  {'From/To':20s}" + "".join(f"{lbl:14s}" for lbl in col_labels)
        print(header)
        print("  " + "-" * (20 + 14 * len(keys)))
        for src_label, src_pos in all_positions:
            row = f"  {src_label:20s}"
            for kx, ky in keys:
                d = find_bfs_distance(graph, src_pos, [(kx, ky)])
                cell = str(int(d)) if d < 1e9 else "---"
                row += f"{cell:14s}"
            print(row)

        # ── 3. Reverse BFS: which cells can reach each unreachable key ─────────
        print("\n" + "=" * 66)
        print("  PREDECESSORS OF UNREACHABLE KEYS")
        print("  (cells that have a direct edge INTO the key cell)")
        print("=" * 66)
        for kx, ky in keys:
            d_from_willy = find_bfs_distance(graph, willy, [(kx, ky)])
            if d_from_willy < 1e9:
                continue  # reachable — skip
            # Find all cells with a direct edge to (kx, ky)
            predecessors = [(cx, cy) for (cx, cy), nbrs in graph.items()
                            if (kx, ky) in nbrs]
            reachable_preds = [(cx, cy) for (cx, cy) in predecessors
                               if find_bfs_distance(graph, willy, [(cx, cy)]) < 1e9]
            print(f"\n  Key ({kx},{ky}) is UNREACHABLE from Willy {willy}")
            print(f"  Total predecessors in graph : {len(predecessors)}")
            print(f"  Predecessors reachable from Willy: {len(reachable_preds)}")
            if predecessors:
                print(f"  All predecessors : {sorted(predecessors)}")
            if reachable_preds:
                print(f"  Reachable ones   : {sorted(reachable_preds)}")

        # ── 4. Per-stage waypoint maps ─────────────────────────────────────────
        # Use the known correct collection order as the fallback sequence so all
        # 6 maps are always printed even when some keys are BFS-unreachable.
        def priority_keys(active_keys):
            """Return highest-reachability-score key(s) from active_keys.
            Mirrors ManicMinerSemanticEnv._update_key_priorities logic."""
            if not active_keys:
                return []
            scores = {}
            for k in active_keys:
                if k not in graph:
                    scores[k] = 0
                    continue
                scores[k] = sum(
                    1 for other in active_keys
                    if other != k and find_bfs_distance(graph, k, [other]) < 1e9
                )
            max_score = max(scores.values(), default=0)
            return [k for k, s in scores.items() if s == max_score]

        def approach_targets(active_keys):
            """Mirror of ManicMinerSemanticEnv._key_approach_targets (with priority)."""
            pkeys = priority_keys(active_keys)
            key_list = pkeys if pkeys else active_keys
            targets = []
            for kx, ky in key_list:
                if (kx, ky) in graph:
                    targets.append((kx, ky))
                else:
                    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                        nb = (kx + dx, ky + dy)
                        if nb in graph:
                            targets.append(nb)
            # Fall back to all keys if priority subset yields nothing
            if not targets:
                for kx, ky in active_keys:
                    if (kx, ky) in graph:
                        targets.append((kx, ky))
                    else:
                        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                            nb = (kx + dx, ky + dy)
                            if nb in graph:
                                targets.append(nb)
            return targets

        def nearest_priority_key(pos, active_keys):
            """Return the highest-priority key nearest to pos by BFS distance."""
            pkeys = priority_keys(active_keys)
            candidates = pkeys if pkeys else active_keys
            best, best_dist = None, float("inf")
            for k in candidates:
                d = find_bfs_distance(graph, pos, [k])
                if d < best_dist:
                    best, best_dist = k, d
            return best, best_dist

        current_pos = willy
        remaining = list(keys)
        collection_order = []

        for stage in range(len(keys) + 1):
            collected = len(keys) - len(remaining)
            targets = approach_targets(remaining) if remaining else []

            waypoint = find_waypoint(graph, current_pos, targets) if remaining else None

            if remaining:
                pkeys = priority_keys(remaining)
                nk, nk_dist = nearest_priority_key(current_pos, remaining)
                title = (f"Stage {stage}: {collected} key(s) collected  |  "
                         f"willy_pos={current_pos}  |  "
                         f"priority={pkeys}  "
                         f"waypoint={waypoint}  "
                         f"targeting={nk}(d={int(nk_dist) if nk_dist < 1e9 else 'UNREACH'})")
            else:
                title = f"Stage 5: all keys collected (priority order: {collection_order})"

            print_map(solid, nasty, conveyor, remaining, current_pos, waypoint, title)

            if remaining:
                nk, nk_dist = nearest_priority_key(current_pos, remaining)
                if nk is None or nk_dist >= 1e9:
                    print(f"  [WARNING: nearest priority key unreachable from {current_pos}]")
                    break
                collection_order.append(nk)
                remaining.remove(nk)
                current_pos = nk

        env.close()

    finally:
        fuse_proc.terminate()
        print("\nEmulator stopped.")


if __name__ == "__main__":
    main()
