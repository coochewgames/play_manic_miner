#!/usr/bin/env python3
"""Train an RL agent against Manic Miner in Fuse ML mode."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from typing import Any, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from logging_utils import configure_logging
from manic_env import ManicMinerEnv

logger = logging.getLogger(__name__)


class StatusWindow:
    """Fixed terminal status panel with in-place updates (no GUI backend)."""

    def __init__(self, title: str):
        self._title = title
        self._fields = [
            ("Timestep", "timestep"),
            ("Key Chord", "action_key_chord"),
            ("Local Jump", "local_jump_forced"),
            ("Local Blocked", "local_jump_blocked"),
            ("Local Reason", "local_jump_reason"),
            ("Repeat Count", "repeat_count"),
            ("Repeat Pen", "repeat_penalty"),
            ("Repeat Why", "repeat_reason"),
            ("Walk Reward", "walk_reward"),
            ("Walk Why", "walk_reason"),
            ("Under Lethal", "under_lethal_reward"),
            ("Under Why", "under_lethal_reason"),
            ("Walk Hold", "walk_hysteresis_applied"),
            ("Walk Hold Why", "walk_hysteresis_reason"),
            ("Jump Gate", "jump_gate_blocked"),
            ("Safety Blocked", "safety_blocked"),
            ("Safety Level", "safety_level"),
            ("Lethal Cells", "known_lethal_cells"),
            ("Lethal Attrs", "known_lethal_attrs"),
            ("Willy X (px)", "willy_x_px"),
            ("Willy Y (px)", "willy_y_px"),
            ("Level", "level"),
            ("Lives", "lives"),
            ("Score", "score"),
            ("Keys Remaining", "keys_remaining"),
            ("Portal Open", "portal_open"),
            ("Step Reward", "step_reward"),
        ]
        self._enabled = sys.stdout.isatty()
        self._ansi_ok = self._enabled and os.environ.get("TERM", "dumb") != "dumb"
        if not self._enabled:
            logger.warning("Status window disabled because stdout is not a TTY")
        elif not self._ansi_ok:
            logger.warning("Status window disabled because terminal does not support ANSI control")

    def update(self, values: Dict[str, Any]) -> None:
        if not self._enabled or not self._ansi_ok:
            return
        lines = [f"{self._title}"]
        for heading, key in self._fields:
            value = values.get(key, "-")
            text = f"{value:.3f}" if isinstance(value, float) else str(value)
            lines.append(f"{heading:<16}: {text}")
        # Preserve the active cursor used by other terminal output.
        sys.stdout.write("\x1b7")
        for row, line in enumerate(lines, start=1):
            sys.stdout.write(f"\x1b[{row};1H\x1b[2K{line}")
        sys.stdout.write("\x1b8")
        sys.stdout.flush()

    def close(self) -> None:
        if self._enabled and self._ansi_ok:
            sys.stdout.write("\x1b7")
            total_rows = 1 + len(self._fields)
            for row in range(1, total_rows + 1):
                sys.stdout.write(f"\x1b[{row};1H\x1b[2K")
            sys.stdout.write("\x1b8")
            sys.stdout.flush()


class SwitchToHeadlessCallback(BaseCallback):
    """Switch env runtime mode from visual to headless after N timesteps."""

    def __init__(self, switch_at_timesteps: int):
        super().__init__()
        self.switch_at_timesteps = max(1, int(switch_at_timesteps))
        self.switched = False

    def _on_step(self) -> bool:
        if not self.switched and self.num_timesteps >= self.switch_at_timesteps:
            self.training_env.env_method("set_runtime_mode", True, 0)
            self.switched = True
            logger.info("Switched to headless mode at timestep %s", self.num_timesteps)
        return True


class StatusWindowCallback(BaseCallback):
    """Keep a small fixed status window updated during training."""

    def __init__(self, title: str, refresh_steps: int):
        super().__init__()
        self.title = str(title)
        self.refresh_steps = max(1, int(refresh_steps))
        self.window: StatusWindow | None = None

    def _on_training_start(self) -> None:
        try:
            self.window = StatusWindow(self.title)
        except Exception as exc:
            self.window = None
            logger.error("Failed to create status window: %s", exc)

    def _on_step(self) -> bool:
        if self.window is None:
            return True
        if self.num_timesteps % self.refresh_steps != 0:
            return True

        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        if not infos:
            return True

        info = infos[0] if isinstance(infos[0], dict) else {}
        state = info.get("state", {}) if isinstance(info.get("state"), dict) else {}
        reward = rewards[0] if len(rewards) > 0 else "-"
        values = {
            "timestep": self.num_timesteps,
            "action_key_chord": info.get("action_key_chord", "-"),
            "local_jump_forced": info.get("local_jump_forced", "-"),
            "local_jump_blocked": info.get("local_jump_blocked", "-"),
            "local_jump_reason": info.get("local_jump_reason", "-"),
            "repeat_count": info.get("repeat_count", "-"),
            "repeat_penalty": info.get("repeat_penalty", "-"),
            "repeat_reason": info.get("repeat_reason", "-"),
            "walk_reward": info.get("walk_reward", "-"),
            "walk_reason": info.get("walk_reason", "-"),
            "under_lethal_reward": info.get("under_lethal_detected", "-"),
            "under_lethal_reason": info.get("under_lethal_reason", "-"),
            "walk_hysteresis_applied": info.get("walk_hysteresis_applied", "-"),
            "walk_hysteresis_reason": info.get("walk_hysteresis_reason", "-"),
            "jump_gate_blocked": info.get("jump_gate_blocked", "-"),
            "safety_blocked": info.get("safety_blocked", "-"),
            "safety_level": info.get("safety_level", "-"),
            "known_lethal_cells": info.get("known_lethal_cells", "-"),
            "known_lethal_attrs": info.get("known_lethal_attrs", "-"),
            "willy_x_px": state.get("willy_x_px", "-"),
            "willy_y_px": state.get("willy_y_px", "-"),
            "level": state.get("level", "-"),
            "lives": state.get("lives", "-"),
            "score": state.get("score", "-"),
            "keys_remaining": state.get("keys_remaining", "-"),
            "portal_open": state.get("portal_open", "-"),
            "step_reward": reward,
        }
        try:
            self.window.update(values)
        except Exception:
            logger.warning("Status window closed or unavailable; disabling status window updates")
            self.window = None
        return True

    def _on_training_end(self) -> None:
        if self.window is not None:
            self.window.close()
            self.window = None


class RunSummaryCallback(BaseCallback):
    """Collect and log run-level summary metrics."""

    def __init__(self):
        super().__init__()
        self.max_level_seen = -1
        self.configured_keys_by_level: Dict[int, int] = {}
        self.min_keys_remaining_by_level: Dict[int, int] = {}
        self.max_score_seen = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        info = infos[0] if isinstance(infos[0], dict) else {}
        state = info.get("state", {}) if isinstance(info.get("state"), dict) else {}
        level = state.get("level")
        keys_remaining = state.get("keys_remaining")
        score = state.get("score")
        configured_keys = info.get("configured_keys_for_level")

        if not isinstance(level, int) or not isinstance(keys_remaining, int):
            return True

        self.max_level_seen = max(self.max_level_seen, level)
        if isinstance(score, int):
            self.max_score_seen = max(self.max_score_seen, score)
        if isinstance(configured_keys, int) and configured_keys >= 0:
            prev_cfg = self.configured_keys_by_level.get(level, 0)
            if configured_keys > prev_cfg:
                self.configured_keys_by_level[level] = configured_keys
            prev_min_remaining = self.min_keys_remaining_by_level.get(level, keys_remaining)
            if keys_remaining < prev_min_remaining:
                self.min_keys_remaining_by_level[level] = keys_remaining
            else:
                self.min_keys_remaining_by_level[level] = prev_min_remaining
        return True

    def _on_training_end(self) -> None:
        if self.max_level_seen < 0:
            logger.info("Run summary: no environment state samples captured")
            return

        reached_level = self.max_level_seen
        configured_keys = self.configured_keys_by_level.get(reached_level)
        min_keys_remaining = self.min_keys_remaining_by_level.get(reached_level)
        cavern_number = reached_level + 1
        if configured_keys is None or min_keys_remaining is None:
            logger.info(
                "Run summary: cavern reached in play testing=%s (level=%s), max keys collected in cavern reached=%s, highest score=%s",
                cavern_number,
                reached_level,
                0,
                self.max_score_seen,
            )
            return
        bounded_remaining = max(0, min(configured_keys, min_keys_remaining))
        max_keys_collected = max(0, configured_keys - bounded_remaining)
        logger.info(
            "Run summary: cavern reached in play testing=%s (level=%s), max keys collected in cavern reached=%s/%s, highest score=%s",
            cavern_number,
            reached_level,
            max_keys_collected,
            configured_keys,
            self.max_score_seen,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Fuse Manic Miner")
    parser.add_argument("--socket", default="/tmp/fuse-ml.sock", help="UNIX socket path")
    parser.add_argument(
        "--socket-timeout-s",
        type=float,
        default=30.0,
        help="Timeout in seconds for each ML bridge command",
    )
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--frames-per-action", type=int, default=2, help="Frames to step per action")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max env steps per episode")
    parser.add_argument(
        "--random-action-prob",
        type=float,
        default=0.0,
        help="Probability of replacing chosen action with a random action during training",
    )
    parser.add_argument(
        "--reset-random-action-steps",
        type=int,
        default=8,
        help="Maximum random warmup actions after RESET (uniformly sampled 0..N)",
    )
    parser.add_argument("--model-out", default="ppo_manic_miner", help="Output model path prefix")
    parser.add_argument(
        "--load-model",
        default="",
        help="Path to an existing .zip model to continue training from",
    )
    parser.add_argument("--tensorboard-log", default="./tb_logs", help="TensorBoard log dir")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.02,
        help="PPO entropy coefficient (higher encourages more exploration)",
    )
    parser.add_argument("--visual", action="store_true", help="Run Fuse in visual mode")
    parser.add_argument("--visual-pace-ms", type=int, default=0, help="Visual pace per frame in ms")
    parser.add_argument(
        "--visual-steps",
        type=int,
        default=0,
        help="If > 0 and --visual is set, switch to headless after this many training timesteps",
    )
    parser.add_argument(
        "--include-bridge-reward",
        action="store_true",
        help="Add bridge reward field to client-side reward shaping",
    )
    parser.add_argument(
        "--pathing-reward",
        type=float,
        default=2.0,
        help="Reward per newly visited screen cell",
    )
    parser.add_argument(
        "--first-key-mode",
        action="store_true",
        help="Stage-1 curriculum: end episode when first key is collected",
    )
    parser.add_argument(
        "--first-key-success-bonus",
        type=float,
        default=120.0,
        help="Bonus reward when first key is collected in --first-key-mode",
    )
    parser.add_argument(
        "--disable-safety-shield",
        action="store_true",
        help="Disable learned action blocking for previously observed lethal cells",
    )
    parser.add_argument(
        "--status-window",
        action="store_true",
        help="Show a fixed live status window (input/actions/Willy position) during training",
    )
    parser.add_argument(
        "--status-window-title",
        default="Manic Miner Live Status",
        help="Title text for the status window",
    )
    parser.add_argument(
        "--status-refresh-steps",
        type=int,
        default=1,
        help="Update frequency for status window in training timesteps",
    )
    return parser.parse_args()


def check_fuse_reset_snapshot_env() -> None:
    snapshot_path = os.environ.get("FUSE_ML_RESET_SNAPSHOT", "").strip()
    if not snapshot_path:
        logger.warning(
            "FUSE_ML_RESET_SNAPSHOT is not set; Fuse RESET may fail or reset to an unexpected state"
        )
        return
    if not os.path.exists(snapshot_path):
        logger.warning("FUSE_ML_RESET_SNAPSHOT does not exist: %s", snapshot_path)
        return
    if not os.path.isfile(snapshot_path):
        logger.warning("FUSE_ML_RESET_SNAPSHOT is not a file: %s", snapshot_path)
        return
    if not os.access(snapshot_path, os.R_OK):
        logger.warning("FUSE_ML_RESET_SNAPSHOT is not readable: %s", snapshot_path)
        return
    logger.info("Fuse reset snapshot: %s", snapshot_path)


def main() -> None:
    args = parse_args()
    configure_logging(level=getattr(logging, args.log_level))
    check_fuse_reset_snapshot_env()

    def make_env():
        env = ManicMinerEnv(
            socket_path=args.socket,
            socket_timeout_s=args.socket_timeout_s,
            frames_per_action=args.frames_per_action,
            max_steps=args.max_steps,
            headless=not args.visual,
            visual_pace_ms=args.visual_pace_ms,
            auto_reset_on_done=False,
            include_bridge_reward=args.include_bridge_reward,
            random_action_prob=args.random_action_prob,
            reset_random_action_steps=args.reset_random_action_steps,
            first_key_mode=args.first_key_mode,
            first_key_success_bonus=args.first_key_success_bonus,
            safety_shield=not args.disable_safety_shield,
            pathing_new_cell_reward=args.pathing_reward,
        )
        return env

    vec_env = VecMonitor(DummyVecEnv([make_env]))

    if args.load_model:
        model = PPO.load(
            args.load_model,
            env=vec_env,
            tensorboard_log=args.tensorboard_log,
        )
        logger.info("Loaded model from %s", args.load_model)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=512,
            batch_size=128,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=args.ent_coef,
            tensorboard_log=args.tensorboard_log,
        )

    callbacks = []
    callbacks.append(RunSummaryCallback())
    if args.visual and args.visual_steps > 0:
        callbacks.append(SwitchToHeadlessCallback(args.visual_steps))
    if args.status_window:
        callbacks.append(StatusWindowCallback(args.status_window_title, args.status_refresh_steps))
    callback = CallbackList(callbacks) if callbacks else None

    learn_completed = False
    timed_out = False
    try:
        try:
            model.learn(
                total_timesteps=args.timesteps,
                progress_bar=not args.status_window,
                callback=callback,
                reset_num_timesteps=not bool(args.load_model),
            )
            learn_completed = True
        except socket.timeout as exc:
            timed_out = True
            logger.warning(
                "Fuse ML socket timed out during training; ending run early and saving partial model: %s",
                exc,
            )
    finally:
        try:
            model.save(args.model_out)
            if learn_completed:
                logger.info("Saved model to %s.zip", args.model_out)
            elif timed_out:
                logger.info("Saved partial model to %s.zip after timeout", args.model_out)
            else:
                logger.info("Saved model to %s.zip after interrupted training", args.model_out)
        except Exception as exc:
            logger.error("Failed to save model %s.zip: %s", args.model_out, exc)
        try:
            vec_env.env_method("quit_emulator")
            logger.info("Sent QUIT to Fuse emulator")
        except Exception as exc:
            logger.warning("Could not send QUIT to Fuse emulator: %s", exc)
        vec_env.close()


if __name__ == "__main__":
    main()
