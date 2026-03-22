#!/usr/bin/env python3
"""Orchestrate one training iteration.

Responsibilities:
  1. Kill any existing Fuse process holding the socket.
  2. Launch Fuse in the background.
  3. Poll the ML socket with PING until Fuse is ready.
  4. Run train.py (blocking).  train.py sends QUIT to Fuse when it finishes.
  5. Run analyse_run.py and print the report.

Usage (single run):
    python run_loop.py

Usage (N iterations, continuing from previous model):
    python run_loop.py --iterations 5 --timesteps 50000

All train.py flags can be passed after --train-args:
    python run_loop.py --train-args "--pathing-reward 3.0 --ent-coef 0.03"
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────

FUSE_BIN         = "/Users/roddy/Dev/fuse/fuse/fuse"
FUSE_SOCKET      = "/tmp/fuse-ml.sock"      # used when --num-envs 1
FUSE_SOCKET_BASE = "/tmp/fuse-ml"           # used when --num-envs > 1  (→ /tmp/fuse-ml-0.sock …)
FUSE_SNAPSHOT    = "/Users/roddy/Dev/fuse/play_manic_miner/manicminer.szx"
FUSE_SPEED       = 3000        # emulation speed % (headless; no rendering overhead)
FUSE_READY_TIMEOUT_S = 30      # max seconds to wait for Fuse to accept connections
FUSE_READY_POLL_S    = 0.25    # seconds between readiness checks

PYTHON          = str(Path(sys.executable))  # same venv python that runs this script
TRAIN_SCRIPT    = str(Path(__file__).parent / "train.py")
ANALYSE_SCRIPT  = str(Path(__file__).parent / "analyse_run.py")
MODEL_PATH      = "ppo_manic_miner"
EPISODE_LOG     = "episode_log.jsonl"


# ── Fuse process management ───────────────────────────────────────────────────

def kill_existing_fuse(socket_path: str) -> None:
    """Send QUIT via socket if a Fuse instance is already running."""
    if not os.path.exists(socket_path):
        return
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect(socket_path)
        # drain the banner
        _readline_sock(sock)
        sock.sendall(b"QUIT\n")
        try:
            _readline_sock(sock)
        except Exception:
            pass
        sock.close()
        print("[loop] Sent QUIT to existing Fuse instance.")
        time.sleep(1.0)
    except Exception:
        pass
    # Remove stale socket file if Fuse didn't clean it up.
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass


def launch_fuse(socket_path: str, snapshot: str, speed: int, visual: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["FUSE_ML_MODE"] = "1"
    env["FUSE_ML_SOCKET"] = socket_path
    env["FUSE_ML_RESET_SNAPSHOT"] = snapshot
    if not visual:
        env["FUSE_ML_VISUAL"] = "0"

    cmd = [FUSE_BIN, f"--speed={speed}"]
    print(f"[loop] Launching Fuse: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(Path(FUSE_BIN).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _readline_sock(sock: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("socket closed")
        buf += chunk
    line, _ = buf.split(b"\n", 1)
    return line.decode("utf-8", errors="replace").strip()


def wait_for_fuse_ready(socket_path: str, timeout_s: float, poll_s: float) -> bool:
    """Return True when Fuse accepts a PING, False on timeout."""
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            if not os.path.exists(socket_path):
                raise FileNotFoundError("socket not yet present")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(socket_path)
            banner = _readline_sock(sock)
            if not banner.startswith("OK READY"):
                sock.close()
                raise RuntimeError(f"unexpected banner: {banner!r}")
            sock.sendall(b"PING\n")
            pong = _readline_sock(sock)
            sock.close()
            if "PONG" in pong or pong.startswith("OK"):
                print(f"[loop] Fuse ready after {attempt} poll(s).")
                return True
        except Exception as exc:
            pass
        time.sleep(poll_s)
    return False


# ── subprocess helpers ────────────────────────────────────────────────────────

def socket_paths_for(num_envs: int) -> list[str]:
    """Return the list of socket paths for the given number of envs."""
    if num_envs <= 1:
        return [FUSE_SOCKET]
    return [f"{FUSE_SOCKET_BASE}-{i}.sock" for i in range(num_envs)]


def run_training(
    timesteps: int,
    model_out: str,
    episode_log: str,
    load_model: str,
    extra_args: list[str],
    visual: bool,
    num_envs: int,
) -> int:
    env = os.environ.copy()
    env["FUSE_ML_RESET_SNAPSHOT"] = FUSE_SNAPSHOT

    sockets = socket_paths_for(num_envs)
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--timesteps", str(timesteps),
        "--model-out", model_out,
        "--episode-log", episode_log,
    ]
    if num_envs <= 1:
        env["FUSE_ML_SOCKET"] = FUSE_SOCKET
        cmd += ["--socket", FUSE_SOCKET]
    else:
        cmd += ["--num-envs", str(num_envs), "--socket-base", FUSE_SOCKET_BASE]
    if load_model and Path(load_model + ".zip").exists():
        cmd += ["--load-model", load_model]
        print(f"[loop] Continuing from model: {load_model}.zip")
    else:
        print("[loop] Starting fresh model.")
    if visual:
        cmd.append("--visual")
    cmd += extra_args
    print(f"[loop] Training command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_analysis(episode_log: str, tail: int = 0) -> None:
    if not Path(episode_log).exists():
        print("[loop] No episode log found — skipping analysis.")
        return
    cmd = [PYTHON, ANALYSE_SCRIPT, episode_log]
    if tail > 0:
        cmd += ["--tail", str(tail)]
    print("\n" + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)
    subprocess.run(cmd)


# ── main loop ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse + training orchestration loop")
    p.add_argument("--iterations", type=int, default=1,
                   help="Number of training iterations to run (default 1)")
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="Timesteps per iteration")
    p.add_argument("--model-out", default=MODEL_PATH,
                   help="Model path prefix (no .zip)")
    p.add_argument("--episode-log", default=EPISODE_LOG,
                   help="Path for per-episode JSONL log")
    p.add_argument("--continue-model", action="store_true",
                   help="Load the model from --model-out if it exists and continue training")
    p.add_argument("--visual", action="store_true",
                   help="Run Fuse in visual mode")
    p.add_argument("--analysis-tail", type=int, default=100,
                   help="Show analysis of last N episodes after each run (0 = all)")
    p.add_argument("--train-args", default="",
                   help="Extra arguments forwarded verbatim to train.py (quote the whole string)")
    p.add_argument("--snapshot", default=FUSE_SNAPSHOT,
                   help="Path to the .szx reset snapshot")
    p.add_argument("--num-envs", type=int, default=1,
                   help="Number of parallel Fuse instances / training environments (default 1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extra_train_args = args.train_args.split() if args.train_args.strip() else []
    num_envs = max(1, args.num_envs)
    fuse_procs: list[subprocess.Popen] = []

    def shutdown(signum=None, frame=None):
        print("\n[loop] Interrupted — shutting down.")
        for p in fuse_procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        sys.exit(1)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    sockets = socket_paths_for(num_envs)

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'━' * 60}")
        print(f"  ITERATION {iteration} / {args.iterations}  ({num_envs} env(s))")
        print(f"{'━' * 60}\n")

        # 1. Clean up any previous Fuse instances.
        for sp in sockets:
            kill_existing_fuse(sp)

        # 2. Launch one Fuse per env.
        fuse_procs = []
        for sp in sockets:
            proc = launch_fuse(sp, args.snapshot, FUSE_SPEED, args.visual)
            fuse_procs.append(proc)

        # 3. Wait for all Fuse instances to be ready.
        print(f"[loop] Waiting for {num_envs} Fuse socket(s) (up to {FUSE_READY_TIMEOUT_S}s each)…")
        for i, sp in enumerate(sockets):
            ready = wait_for_fuse_ready(sp, FUSE_READY_TIMEOUT_S, FUSE_READY_POLL_S)
            if not ready:
                print(f"[loop] ERROR: Fuse env {i} ({sp}) did not become ready in time.")
                for p in fuse_procs:
                    p.terminate()
                sys.exit(1)

        # 4. Run training.  train.py sends QUIT to Fuse instances when it finishes.
        load_model = args.model_out if args.continue_model else ""
        rc = run_training(
            timesteps=args.timesteps,
            model_out=args.model_out,
            episode_log=args.episode_log,
            load_model=load_model,
            extra_args=extra_train_args,
            visual=args.visual,
            num_envs=num_envs,
        )
        if rc != 0:
            print(f"[loop] WARNING: train.py exited with code {rc}")

        # 5. Ensure all Fuse instances are gone.
        for i, (proc, sp) in enumerate(zip(fuse_procs, sockets)):
            if proc.poll() is None:
                print(f"[loop] Fuse env {i} still running — sending QUIT.")
                kill_existing_fuse(sp)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
        fuse_procs = []

        # 6. Analyse.
        run_analysis(args.episode_log, tail=args.analysis_tail)

        if iteration < args.iterations:
            print(f"\n[loop] Iteration {iteration} complete. Starting next in 2s…")
            time.sleep(2.0)
            # From the second iteration onwards always continue from saved model.
            args.continue_model = True

    print("\n[loop] All iterations complete.")


if __name__ == "__main__":
    main()
