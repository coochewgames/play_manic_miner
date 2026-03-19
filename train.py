#!/usr/bin/env python3
"""Train an RL agent against Manic Miner in Fuse ML mode."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from typing import Any, Dict

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from logging_utils import configure_logging
from manic_env import ManicMinerEnv

logger = logging.getLogger(__name__)


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


class EpisodeLoggerCallback(BaseCallback):
    """Write one JSON line per episode to a .jsonl log file."""

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self._episode_count = 0
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._ep_steps = 0
        self._ep_total_reward = 0.0
        self._ep_objective = 0.0
        self._ep_hazard = 0.0
        self._ep_pathing = 0.0
        self._ep_repeat_penalty = 0.0
        self._ep_walk_reward = 0.0
        self._ep_under_lethal = 0.0
        self._ep_safety_triggers = 0
        self._ep_jump_gate_blocks = 0
        self._ep_life_losses = 0
        self._ep_level = 0
        self._ep_coverage_ratio = 0.0
        self._ep_keys_remaining = 0
        self._ep_configured_keys = 0

    def _on_training_start(self) -> None:
        logger.info("Episode log: %s", self.log_path)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if not infos:
            return True

        info = infos[0] if isinstance(infos[0], dict) else {}
        reward = float(rewards[0]) if rewards else 0.0
        done = bool(dones[0]) if dones else False

        state = info.get("state", {})
        if not isinstance(state, dict):
            state = {}

        self._ep_steps += 1
        self._ep_total_reward += reward
        self._ep_objective += float(info.get("objective_reward", 0.0))
        self._ep_hazard += float(info.get("hazard_reward", 0.0))
        self._ep_pathing += float(info.get("pathing_reward", 0.0))
        self._ep_repeat_penalty += float(info.get("repeat_penalty", 0.0))
        self._ep_walk_reward += float(info.get("walk_reward", 0.0))
        self._ep_under_lethal += float(info.get("under_lethal_reward", 0.0))
        if info.get("safety_blocked"):
            self._ep_safety_triggers += 1
        if info.get("jump_gate_blocked"):
            self._ep_jump_gate_blocks += 1
        if info.get("life_lost_this_step"):
            self._ep_life_losses += 1

        level = state.get("level", 0)
        if isinstance(level, int):
            self._ep_level = max(self._ep_level, level)

        coverage = state.get("coverage_ratio", 0.0)
        if isinstance(coverage, (int, float)):
            self._ep_coverage_ratio = float(coverage)

        keys_remaining = state.get("keys_remaining")
        configured_keys = info.get("configured_keys_for_level")
        if isinstance(keys_remaining, int):
            self._ep_keys_remaining = keys_remaining
        if isinstance(configured_keys, int):
            self._ep_configured_keys = configured_keys

        if done:
            if info.get("first_key_terminated"):
                term_reason = "first_key"
            elif state.get("lives", 1) == 0:
                term_reason = "lives_exhausted"
            elif info.get("life_lost_this_step"):
                term_reason = "life_lost"
            elif info.get("bridge_done"):
                term_reason = "bridge_done"
            else:
                term_reason = "truncated"

            keys_collected = max(0, self._ep_configured_keys - self._ep_keys_remaining)
            self._episode_count += 1
            record: Dict[str, Any] = {
                "episode": self._episode_count,
                "timestep": self.num_timesteps,
                "steps": self._ep_steps,
                "total_reward": round(self._ep_total_reward, 3),
                "objective_reward": round(self._ep_objective, 3),
                "hazard_reward": round(self._ep_hazard, 3),
                "pathing_reward": round(self._ep_pathing, 3),
                "repeat_penalty": round(self._ep_repeat_penalty, 3),
                "walk_reward": round(self._ep_walk_reward, 3),
                "under_lethal_reward": round(self._ep_under_lethal, 3),
                "keys_collected": keys_collected,
                "configured_keys": self._ep_configured_keys,
                "keys_remaining": self._ep_keys_remaining,
                "life_losses": self._ep_life_losses,
                "safety_triggers": self._ep_safety_triggers,
                "jump_gate_blocks": self._ep_jump_gate_blocks,
                "level": self._ep_level,
                "coverage_ratio": round(self._ep_coverage_ratio, 4),
                "term_reason": term_reason,
            }
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as exc:
                logger.warning("Failed to write episode log: %s", exc)

            self._reset_accumulators()

        return True


class RunSummaryCallback(BaseCallback):
    """Log a run-level summary at the end of training."""

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
            prev_min = self.min_keys_remaining_by_level.get(level, keys_remaining)
            self.min_keys_remaining_by_level[level] = min(prev_min, keys_remaining)
        return True

    def _on_training_end(self) -> None:
        if self.max_level_seen < 0:
            logger.info("Run summary: no environment state samples captured")
            return
        level = self.max_level_seen
        configured = self.configured_keys_by_level.get(level)
        min_remaining = self.min_keys_remaining_by_level.get(level)
        cavern = level + 1
        if configured is None or min_remaining is None:
            logger.info(
                "Run summary: cavern=%s level=%s keys_collected=0 score=%s",
                cavern, level, self.max_score_seen,
            )
            return
        collected = max(0, configured - max(0, min(configured, min_remaining)))
        logger.info(
            "Run summary: cavern=%s level=%s keys_collected=%s/%s score=%s",
            cavern, level, collected, configured, self.max_score_seen,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Fuse Manic Miner")
    parser.add_argument("--socket", default="/tmp/fuse-ml.sock", help="UNIX socket path")
    parser.add_argument("--socket-timeout-s", type=float, default=30.0,
                        help="Timeout in seconds for each ML bridge command")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--frames-per-action", type=int, default=2, help="Frames to step per action")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max env steps per episode")
    parser.add_argument("--random-action-prob", type=float, default=0.0,
                        help="Probability of replacing chosen action with a random action")
    parser.add_argument("--reset-random-action-steps", type=int, default=8,
                        help="Maximum random warmup actions after RESET")
    parser.add_argument("--model-out", default="ppo_manic_miner", help="Output model path prefix")
    parser.add_argument("--load-model", default="",
                        help="Path to an existing .zip model to continue training from")
    parser.add_argument("--tensorboard-log", default="./tb_logs", help="TensorBoard log dir")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--ent-coef", type=float, default=0.08,
                        help="PPO entropy coefficient")
    parser.add_argument("--gamma", type=float, default=0.999, help="PPO discount factor")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout buffer size")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="PPO minibatch size")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes for the policy/value network")
    parser.add_argument("--visual", action="store_true", help="Run Fuse in visual mode")
    parser.add_argument("--visual-pace-ms", type=int, default=0,
                        help="Visual pace per frame in ms")
    parser.add_argument("--visual-steps", type=int, default=0,
                        help="Switch to headless after this many timesteps (requires --visual)")
    parser.add_argument("--include-bridge-reward", action="store_true",
                        help="Add bridge reward field to shaped reward")
    parser.add_argument("--pathing-reward", type=float, default=3.0,
                        help="Reward per newly visited screen cell")
    parser.add_argument("--first-key-mode", action="store_true",
                        help="Curriculum: end episode on first key collected")
    parser.add_argument("--first-key-success-bonus", type=float, default=120.0,
                        help="Bonus reward for first key in --first-key-mode")
    parser.add_argument("--disable-safety-shield", action="store_true",
                        help="Disable lethal-action blocking")
    parser.add_argument("--infinite-air", action="store_true",
                        help="Poke air supply to max each step")
    parser.add_argument("--episode-log", default="episode_log.jsonl",
                        help="Path for per-episode JSONL log file")
    return parser.parse_args()


def check_fuse_reset_snapshot_env() -> None:
    snapshot_path = os.environ.get("FUSE_ML_RESET_SNAPSHOT", "").strip()
    if not snapshot_path:
        logger.warning("FUSE_ML_RESET_SNAPSHOT is not set")
        return
    if not os.path.isfile(snapshot_path):
        logger.warning("FUSE_ML_RESET_SNAPSHOT does not exist or is not a file: %s", snapshot_path)
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
        return ManicMinerEnv(
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
            infinite_air=args.infinite_air,
        )

    vec_env = VecMonitor(DummyVecEnv([make_env]))

    if args.load_model:
        model = MaskablePPO.load(args.load_model, env=vec_env, tensorboard_log=args.tensorboard_log)
        logger.info("Loaded model from %s", args.load_model)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=3e-4,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            policy_kwargs={"net_arch": args.net_arch},
            tensorboard_log=args.tensorboard_log,
        )

    callbacks: list = [
        RunSummaryCallback(),
        EpisodeLoggerCallback(args.episode_log),
    ]
    if args.visual and args.visual_steps > 0:
        callbacks.append(SwitchToHeadlessCallback(args.visual_steps))

    learn_completed = False
    timed_out = False
    try:
        try:
            # Always reset the timestep counter so --timesteps means exactly
            # "train for this many additional steps".  Policy weights are
            # preserved regardless of reset_num_timesteps.
            model.learn(
                total_timesteps=args.timesteps,
                progress_bar=False,
                callback=CallbackList(callbacks),
                reset_num_timesteps=True,
                use_masking=True,
            )
            learn_completed = True
        except socket.timeout as exc:
            timed_out = True
            logger.warning("Socket timed out; saving partial model: %s", exc)
    finally:
        try:
            model.save(args.model_out)
            label = "partial (timeout)" if timed_out else ("partial (interrupted)" if not learn_completed else "")
            suffix = f" [{label}]" if label else ""
            logger.info("Saved model to %s.zip%s", args.model_out, suffix)
        except Exception as exc:
            logger.error("Failed to save model: %s", exc)
        try:
            vec_env.env_method("quit_emulator")
        except Exception as exc:
            logger.warning("Could not send QUIT to Fuse: %s", exc)
        vec_env.close()


if __name__ == "__main__":
    main()
