#!/usr/bin/env python3
"""Gymnasium environment for Manic Miner running inside Fuse ML mode."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym as gym  # type: ignore

from manic_data import ManicDataMixin, ManicState, WILLY_AIRBORNE_ADDR, AIR_SUPPLY_ADDR, AIR_SUPPLY_MAX
from ml_client import FuseMLClient

PLAY_MODULE_NAME = os.environ.get("MANIC_PLAY_MODULE", "manic_play")
_play_module = importlib.import_module(PLAY_MODULE_NAME)
ManicPlayMixin = _play_module.ManicPlayMixin
PATHING_NEW_CELL_REWARD = _play_module.PATHING_NEW_CELL_REWARD
logger = logging.getLogger(__name__)


ACTION_KEY_CHORDS: Dict[int, str] = {
    0: "-",
    1: "q",
    2: "w",
    3: "space",
    4: "q+space",
    5: "w+space",
}

KEY_SCORE_INCREMENT = 100


class ManicMinerEnv(ManicDataMixin, ManicPlayMixin, gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        socket_path: str = "/tmp/fuse-ml.sock",
        socket_timeout_s: float = 30.0,
        frames_per_action: int = 4,
        max_steps: int = 4000,
        headless: bool = True,
        visual_pace_ms: int = 0,
        auto_reset_on_done: bool = False,
        include_bridge_reward: bool = False,
        random_action_prob: float = 0.0,
        reset_random_action_steps: int = 8,
        first_key_mode: bool = False,
        safety_shield: bool = True,
        pathing_new_cell_reward: float = PATHING_NEW_CELL_REWARD,
        infinite_air: bool = False,
    ):
        super().__init__()
        self.socket_path = socket_path
        self.socket_timeout_s = max(1.0, float(socket_timeout_s))
        self.frames_per_action = frames_per_action
        self.max_steps = max_steps
        self.headless = headless
        self.visual_pace_ms = visual_pace_ms
        self.auto_reset_on_done = auto_reset_on_done
        self.include_bridge_reward = include_bridge_reward
        self.random_action_prob = float(np.clip(random_action_prob, 0.0, 1.0))
        self.reset_random_action_steps = max(0, int(reset_random_action_steps))
        self.first_key_mode = bool(first_key_mode)
        self.safety_shield = bool(safety_shield)
        self.pathing_new_cell_reward = float(pathing_new_cell_reward)
        self.infinite_air = bool(infinite_air)

        # 0=no-op, 1=left(q), 2=right(w), 3=jump(space), 4=jump-left, 5=jump-right.
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(13, dtype=np.float32),
            high=np.ones(13, dtype=np.float32),
            dtype=np.float32,
        )

        self.client = FuseMLClient(socket_path=socket_path, timeout_s=self.socket_timeout_s)
        self._configure_mode()

        self.step_count = 0
        self.prev_state: Optional[ManicState] = None
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.configured_lethal_attrs_by_level: Dict[int, Set[int]] = {}
        self.configured_lethal_attr_groups_by_level: Dict[int, Tuple[Set[int], Set[int]]] = {}
        self._h_guardian_bounds_by_level: Dict[int, list] = {}
        self._last_dynamic_lethal_cells_count = 0
        self._supports_episode_step_keys: Optional[bool] = None
        self._last_action_key_chord = "-"
        self._tracked_key_level: Optional[int] = None
        self._tracked_keys_remaining = 0
        self._tracked_key_score_anchor = 0

    def _rng_random(self) -> float:
        if hasattr(self, "np_random") and self.np_random is not None:
            return float(self.np_random.random())
        return float(np.random.random())

    def _rng_integers(self, low: int, high: int) -> int:
        if hasattr(self, "np_random") and self.np_random is not None:
            return int(self.np_random.integers(low, high))
        return int(np.random.randint(low, high))

    def _configure_mode(self) -> None:
        if self.headless:
            self.client.mode_headless()
        else:
            self.client.mode_visual(self.visual_pace_ms)

    def set_runtime_mode(self, headless: bool, visual_pace_ms: Optional[int] = None) -> None:
        self.headless = bool(headless)
        if visual_pace_ms is not None:
            self.visual_pace_ms = max(0, int(visual_pace_ms))
        self._configure_mode()

    def _read_u8(self, address: int) -> int:
        return self.client.read_bytes(address, 1)[0]

    def _read_u16_be(self, address: int) -> int:
        hi, lo = self.client.read_bytes(address, 2)
        return (hi << 8) | lo

    def _read_u16_le(self, address: int) -> int:
        lo, hi = self.client.read_bytes(address, 2)
        return lo | (hi << 8)

    def _action_to_key_chord(self, action: int) -> str:
        return ACTION_KEY_CHORDS.get(int(action), "-")

    def _episode_step_with_action(self, action: int, frames: int) -> Dict[str, int]:
        auto_reset = 1 if self.auto_reset_on_done else 0
        key_chord = self._action_to_key_chord(action)

        if self._supports_episode_step_keys is not False:
            try:
                episode = self.client.episode_step_keys(key_chord, frames, auto_reset=auto_reset)
                self._supports_episode_step_keys = True
                self._last_action_key_chord = key_chord
                return episode
            except RuntimeError as exc:
                text = str(exc)
                if "unknown command" not in text and "EPISODE_STEP_KEYS" not in text:
                    raise
                self._supports_episode_step_keys = False

        episode = self.client.episode_step(int(action), frames, auto_reset=auto_reset)
        self._last_action_key_chord = key_chord
        return episode

    def _stabilize_keys_remaining(self, state: ManicState, force_reset: bool = False) -> None:
        configured = self._count_configured_items_for_level(state.level)
        life_lost = self.prev_state is not None and state.lives < self.prev_state.lives
        level_changed = self._tracked_key_level != state.level

        if force_reset or level_changed or life_lost:
            self._tracked_key_level = state.level
            self._tracked_keys_remaining = configured
            self._tracked_key_score_anchor = state.score
        else:
            score_delta = int(state.score) - int(self._tracked_key_score_anchor)
            if score_delta >= KEY_SCORE_INCREMENT and self._tracked_keys_remaining > 0:
                collected = min(self._tracked_keys_remaining, score_delta // KEY_SCORE_INCREMENT, 1)
                self._tracked_keys_remaining -= int(collected)
                self._tracked_key_score_anchor += int(collected) * KEY_SCORE_INCREMENT
            elif score_delta < 0:
                self._tracked_key_score_anchor = state.score

        state.keys_remaining = max(0, min(configured, int(self._tracked_keys_remaining)))
        state.portal_open = 1 if state.keys_remaining == 0 else 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._configure_mode()
        self.client.reset()

        warmup_steps = self._rng_integers(0, self.reset_random_action_steps + 1)
        for _ in range(warmup_steps):
            random_action = self._rng_integers(0, self.action_space.n)
            airborne_status = self._read_u8(WILLY_AIRBORNE_ADDR)
            input_suppressed_for_airborne = self._is_airborne_status_active(airborne_status)
            action_used = 0 if input_suppressed_for_airborne else int(random_action)
            frames_used = 1 if (action_used in (3, 4, 5) or input_suppressed_for_airborne) else self.frames_per_action
            self._episode_step_with_action(action_used, frames_used)

        state = self._read_state()
        self._stabilize_keys_remaining(state, force_reset=True)
        info = self.client.get_info()

        self.visited_cells.clear()
        self.visited_cells.update(self._covered_cells(state.willy_x_px, state.willy_y_px))
        state.coverage_ratio = self._coverage_ratio()

        self.prev_state = state
        self.step_count = 0
        self._tracked_key_level = state.level
        self._tracked_keys_remaining = state.keys_remaining
        self._tracked_key_score_anchor = state.score

        info["state"] = state.__dict__.copy()
        info["first_key_mode"] = self.first_key_mode
        return self._normalize_observation(state.to_observation()), info

    def step(self, action: int):
        self.step_count += 1

        action_used = int(action)
        frames_used = self.frames_per_action
        random_action_applied = False
        state_before = self.prev_state if self.prev_state is not None else self._read_state()
        airborne_status_before = self._read_u8(WILLY_AIRBORNE_ADDR)
        input_suppressed_for_airborne = self._is_airborne_status_active(airborne_status_before)
        attr_buffer_before = self._read_attr_buffer()
        safety_level = state_before.level
        safety_blocked = False
        safety_reason = "-"

        if input_suppressed_for_airborne:
            action_used = 0
            safety_reason = f"airborne_input_suppressed(status=0x{airborne_status_before:02X})"
        else:
            if self.random_action_prob > 0.0 and self._rng_random() < self.random_action_prob:
                action_used = self._rng_integers(0, self.action_space.n)
                random_action_applied = True

            action_used, safety_blocked, safety_reason = self._select_safe_action(
                state_before, action_used, attr_buffer_before
            )

        # Jump is a one-shot input: send for one frame only, then suppress
        # inputs while Willy is airborne.
        if action_used in (3, 4, 5) or input_suppressed_for_airborne:
            frames_used = 1

        ep = self._episode_step_with_action(action_used, frames_used)
        if self.infinite_air:
            self.client.write_bytes(AIR_SUPPLY_ADDR, bytes([AIR_SUPPLY_MAX]))
        state = self._read_state()
        self._stabilize_keys_remaining(state)
        airborne_status_after = self._read_u8(WILLY_AIRBORNE_ADDR)
        is_airborne = self._is_airborne_status_active(airborne_status_after)
        life_lost_this_step = self.prev_state is not None and state.lives < self.prev_state.lives

        total, painting, frontier, key_reward, level_reward, life_penalty, key_approach = self._compute_reward(
            state, self.prev_state, action_used, is_airborne=is_airborne
        )
        if self.include_bridge_reward:
            total += float(ep["reward"])

        first_key_achieved_now = (
            self.prev_state is not None
            and state.keys_remaining < self.prev_state.keys_remaining
        )
        bridge_done = bool(ep["done"])
        first_key_terminated = self.first_key_mode and first_key_achieved_now
        terminated = bridge_done or state.lives == 0 or first_key_terminated
        truncated = self.step_count >= self.max_steps

        info: Dict[str, Any] = {
            "frame_count": ep["frame_count"],
            "tstates": ep["tstates"],
            "screen_width": ep["width"],
            "screen_height": ep["height"],
            "bridge_reward": ep["reward"],
            "bridge_done": ep["done"],
            "bridge_reset": ep["reset"],
            "state": state.__dict__.copy(),
            "action_key_chord": self._last_action_key_chord,
            "frames_used": frames_used,
            "random_action_applied": random_action_applied,
            "airborne_status_before": airborne_status_before,
            "airborne_status_after": airborne_status_after,
            "input_suppressed_for_airborne": input_suppressed_for_airborne,
            "safety_shield": self.safety_shield,
            "safety_level": safety_level,
            "safety_blocked": safety_blocked,
            "safety_reason": safety_reason,
            "known_lethal_cells": self._last_dynamic_lethal_cells_count,
            "known_lethal_attrs": len(self._configured_lethal_attrs_for_level(safety_level)),
            "life_lost_this_step": life_lost_this_step,
            "configured_keys_for_level": self._count_configured_items_for_level(state.level),
            "first_key_mode": self.first_key_mode,
            "first_key_achieved": first_key_achieved_now,
            "first_key_terminated": first_key_terminated,
            "painting_reward": painting,
            "frontier_reward": frontier,
            "key_reward": key_reward,
            "key_approach_reward": key_approach,
            "level_reward": level_reward,
            "life_penalty": life_penalty,
            "visited_cells": len(self.visited_cells),
            "exit_distance_px": self._distance_to_exit_px(state),
        }

        self.prev_state = state
        return self._normalize_observation(state.to_observation()), total, terminated, truncated, info

    def close(self) -> None:
        self.client.close()

    def quit_emulator(self) -> None:
        try:
            self.client.quit()
        except Exception as exc:
            logger.warning("Failed to send QUIT to Fuse ML bridge: %s", exc)
