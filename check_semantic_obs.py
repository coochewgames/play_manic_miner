#!/usr/bin/env python3
"""Diagnostic: print the 5-channel semantic observation as ASCII art.

Launches the emulator, reads one observation, prints each channel, then quits.

Usage:
    python check_semantic_obs.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

from manic_env_semantic import ManicMinerSemanticEnv, GAME_CELLS_H, GAME_CELLS_W

# ── emulator config (mirrors run_loop.py) ─────────────────────────────────────

FUSE_BIN      = "/Users/roddy/Dev/fuse/fuse/fuse"
FUSE_SOCKET   = "/tmp/fuse-ml.sock"
FUSE_SNAPSHOT = "/Users/roddy/Dev/fuse/play_manic_miner/manicminer.szx"
FUSE_SPEED    = 800   # slower than training so the window is visible

CHANNEL_NAMES  = ["SOLID", "NASTY", "WILLY", "GUARDIAN", "KEY", "PORTAL", "WAYPOINT"]
CHANNEL_GLYPHS = [
    (" ", "#"),   # SOLID
    (" ", "X"),   # NASTY
    (" ", "W"),   # WILLY
    (" ", "G"),   # GUARDIAN
    (" ", "K"),   # KEY
    (" ", "P"),   # PORTAL
    (" ", ">"),   # WAYPOINT
]

# ── emulator helpers ──────────────────────────────────────────────────────────

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
    env["FUSE_ML_RESET_SNAPSHOT"] = FUSE_SNAPSHOT
    proc = subprocess.Popen(
        [FUSE_BIN, f"--speed={FUSE_SPEED}"],
        env=env,
        cwd=str(Path(FUSE_BIN).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for the socket to be ready
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            if not os.path.exists(FUSE_SOCKET):
                raise FileNotFoundError
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(FUSE_SOCKET)
            banner = _readline_sock(s)
            s.sendall(b"PING\n")
            _readline_sock(s)
            s.close()
            print("Emulator ready.")
            return proc
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Emulator did not become ready in time.")

# ── display helpers ───────────────────────────────────────────────────────────

def print_channel(name: str, grid, glyph_off: str, glyph_on: str) -> None:
    print(f"\n  {name}")
    print("  +" + "-" * GAME_CELLS_W + "+")
    for row in range(GAME_CELLS_H):
        line = "".join(glyph_on if grid[row, col] else glyph_off
                       for col in range(GAME_CELLS_W))
        print(f"  |{line}|")
    print("  +" + "-" * GAME_CELLS_W + "+")


def print_composite(obs) -> None:
    """All channels overlaid; later channels (higher index) take priority."""
    print("\n  COMPOSITE  (# solid, X nasty, W willy, G guardian, K key, P portal, > waypoint, . empty)")
    print("  +" + "-" * GAME_CELLS_W + "+")
    for row in range(GAME_CELLS_H):
        line = []
        for col in range(GAME_CELLS_W):
            glyph = "."
            for ch, (_, g_on) in enumerate(CHANNEL_GLYPHS):
                if obs[row, col, ch]:
                    glyph = g_on
            line.append(glyph)
        print(f"  |{''.join(line)}|")
    print("  +" + "-" * GAME_CELLS_W + "+")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Launching emulator…")
    fuse_proc = launch_fuse()

    try:
        print("Building semantic observation…")
        env = ManicMinerSemanticEnv(
            socket_path=FUSE_SOCKET,
            headless=False,
            warmup_steps=10,
        )
        obs, _ = env.reset()

        # Channel summary
        print("\nChannel summary (cells with value 255):")
        for ch, name in enumerate(CHANNEL_NAMES):
            count = int((obs[:, :, ch] > 0).sum())
            print(f"  Ch {ch} {name:8s}: {count:3d} cells")

        # Per-channel grids
        for ch, (name, (g_off, g_on)) in enumerate(zip(CHANNEL_NAMES, CHANNEL_GLYPHS)):
            print_channel(name, obs[:, :, ch], g_off, g_on)

        # Composite
        print_composite(obs)

        # Key positions
        print(f"\nActive key positions: {env._active_keys}")

        # Guardian: take 5 no-op steps and report position each time
        print("\nGuardian position over 5 no-op steps:")
        current_obs = obs
        for i in range(5):
            g = (current_obs[:, :, 3] > 0).nonzero()
            if len(g[0]):
                cols = sorted(set(g[1].tolist()))
                rows = sorted(set(g[0].tolist()))
                print(f"  Step {i}: cols={cols}  rows={rows}  ({len(g[0])} cells)")
            else:
                print(f"  Step {i}: not visible")
            current_obs, _, _, _, _ = env.step(0)

        env.close()

    finally:
        fuse_proc.terminate()
        print("\nEmulator stopped.")


if __name__ == "__main__":
    main()
