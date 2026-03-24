#!/usr/bin/env python3
"""Train PPO with a custom CNN policy on the semantic Manic Miner environment."""

from __future__ import annotations

import argparse
import json
import logging
import socket
from typing import Dict, Any

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from logging_utils import configure_logging
from manic_env_semantic import ManicMinerSemanticEnv

logger = logging.getLogger(__name__)


# ── Custom CNN feature extractor ──────────────────────────────────────────────
#
# SB3's NatureCNN is designed for 84×84 Atari frames and its three
# convolutional layers reduce a 16×32 input to zero height.  This extractor
# uses smaller kernels suited to the 16×32 semantic grid.
#
# Input after VecTransposeImage: (B, C, 16, 32),
#   C = 5 channels (solid, nasty, willy, guardian, key) × n_stack (default 4) = 20.
#
# Spatial sizes after each layer:
#   Conv(3, s=1, p=1)  →  (32, 16, 32)
#   Conv(3, s=2)       →  (64,  7, 15)
#   Conv(3, s=1)       →  (64,  5, 13)
#   Flatten            →  64 × 5 × 13 = 4160
#   Linear             →  features_dim


class AttrGridCNN(BaseFeaturesExtractor):
    """Small CNN suited to the 24×32 ZX Spectrum attribute grid."""

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # SB3 transposes (H, W, C) → (C, H, W) before passing to the network.
        # observation_space.shape is (C, H, W) at this point.
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 normalises uint8 observations to [0, 1] before calling forward.
        return self.linear(self.cnn(observations))


# ── Callbacks ─────────────────────────────────────────────────────────────────


class EpisodeLoggerCallback(BaseCallback):
    """Write one JSON line per episode to a .jsonl log file."""

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self._ep_count = 0
        self._reset()

    def _reset(self) -> None:
        self._ep_steps = 0
        self._ep_reward = 0.0
        self._ep_score = 0
        self._ep_deaths = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if not infos:
            return True

        info = infos[0] if isinstance(infos[0], dict) else {}
        reward = float(rewards[0]) if rewards else 0.0
        done = bool(dones[0]) if dones else False

        self._ep_steps += 1
        self._ep_reward += reward
        if info.get("life_lost"):
            self._ep_deaths += 1
        score = info.get("score")
        if isinstance(score, int):
            self._ep_score = score

        if done:
            self._ep_count += 1
            record: Dict[str, Any] = {
                "episode": self._ep_count,
                "timestep": self.num_timesteps,
                "steps": self._ep_steps,
                "total_reward": round(self._ep_reward, 3),
                "score": self._ep_score,
                "deaths": self._ep_deaths,
            }
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as exc:
                logger.warning("Failed to write episode log: %s", exc)
            self._reset()

        return True


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO (CnnPolicy) on visual Manic Miner")
    p.add_argument("--socket", default="/tmp/fuse-ml.sock", help="UNIX socket path")
    p.add_argument("--socket-timeout-s", type=float, default=30.0)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--frames-per-action", type=int, default=3,
                   help="Emulator frames advanced per agent action")
    p.add_argument("--max-steps", type=int, default=4000,
                   help="Max env steps per episode before truncation")
    p.add_argument("--model-out", default="manic_semantic", help="Output model path prefix (no .zip)")
    p.add_argument("--load-model", default="", help="Path to an existing .zip to continue from")
    p.add_argument("--tensorboard-log", default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    p.add_argument("--ent-coef", type=float, default=0.01, help="PPO entropy coefficient")
    p.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor")
    p.add_argument("--n-steps", type=int, default=2048, help="PPO rollout buffer size")
    p.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-stack", type=int, default=4, help="Number of frames to stack")
    p.add_argument("--features-dim", type=int, default=512,
                   help="Output dimension of the CNN feature extractor")
    p.add_argument("--death-penalty", type=float, default=1.0,
                   help="Reward penalty applied when a life is lost")
    p.add_argument("--score-scale", type=float, default=100.0,
                   help="Divide raw score delta by this to produce reward")
    p.add_argument("--warmup-steps", type=int, default=10,
                   help="No-op steps after reset to let the snapshot finish loading")
    p.add_argument("--no-infinite-air", action="store_true",
                   help="Let the air supply deplete normally (default: infinite)")
    p.add_argument("--pathing-reward", type=float, default=0.01,
                   help="Reward per newly visited screen cell (0=off). "
                        "Keep small (e.g. 0.01) so key reward (+1.0) stays dominant.")
    p.add_argument("--visual", action="store_true", help="Run Fuse in visual mode")
    p.add_argument("--visual-pace-ms", type=int, default=0)
    p.add_argument("--episode-log", default="episode_log.jsonl")
    p.add_argument("--device", default="auto",
                   help="PyTorch device: auto, cpu, cuda, mps (default: auto)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────


def resolve_device(requested: str) -> str:
    """Return the best available PyTorch device string."""
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    configure_logging(level=getattr(logging, args.log_level))
    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    def make_env():
        return ManicMinerSemanticEnv(
            socket_path=args.socket,
            socket_timeout_s=args.socket_timeout_s,
            frames_per_action=args.frames_per_action,
            max_steps=args.max_steps,
            headless=not args.visual,
            visual_pace_ms=args.visual_pace_ms,
            death_penalty=args.death_penalty,
            score_scale=args.score_scale,
            warmup_steps=args.warmup_steps,
            infinite_air=not args.no_infinite_air,
            pathing_reward=args.pathing_reward,
        )

    # Stack 4 frames; SB3 will apply VecTransposeImage automatically for CnnPolicy.
    vec_env = VecMonitor(
        VecFrameStack(DummyVecEnv([make_env]), n_stack=args.n_stack)
    )

    policy_kwargs = {
        "features_extractor_class": AttrGridCNN,
        "features_extractor_kwargs": {"features_dim": args.features_dim},
    }

    if args.load_model:
        model = PPO.load(
            args.load_model,
            env=vec_env,
            device=device,
            tensorboard_log=args.tensorboard_log,
        )
        logger.info("Loaded model from %s", args.load_model)
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            device=device,
            tensorboard_log=args.tensorboard_log,
        )

    learn_completed = False
    timed_out = False
    try:
        try:
            model.learn(
                total_timesteps=args.timesteps,
                progress_bar=False,
                callback=EpisodeLoggerCallback(args.episode_log),
                reset_num_timesteps=True,
            )
            learn_completed = True
        except socket.timeout as exc:
            timed_out = True
            logger.warning("Socket timed out; saving partial model: %s", exc)
    finally:
        try:
            model.save(args.model_out)
            label = (
                "partial (timeout)" if timed_out
                else ("partial (interrupted)" if not learn_completed else "")
            )
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
