#!/usr/bin/env python3
"""Gymnasium environment for Manic Miner running inside Fuse ML mode."""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym as gym  # type: ignore

from manic_data import ManicDataMixin, ManicState, WILLY_AIRBORNE_ADDR
from ml_client import FuseMLClient

PLAY_MODULE_NAME = os.environ.get("MANIC_PLAY_MODULE", "manic_play")
_play_module = importlib.import_module(PLAY_MODULE_NAME)
ManicPlayMixin = _play_module.ManicPlayMixin
PATHING_NEW_CELL_REWARD = _play_module.PATHING_NEW_CELL_REWARD

ACTION_KEY_CHORDS: Dict[int, str] = {
    0: "-",
    1: "q",
    2: "w",
    3: "space",
    4: "q+space",
    5: "w+space",
}

WALK_FLIP_HYSTERESIS_STEPS = 2


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
        first_key_success_bonus: float = 120.0,
        safety_shield: bool = True,
        pathing_new_cell_reward: float = PATHING_NEW_CELL_REWARD,
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
        self.first_key_success_bonus = float(first_key_success_bonus)
        self.safety_shield = bool(safety_shield)
        self.pathing_new_cell_reward = float(pathing_new_cell_reward)

        # 0=no-op, 1=left(q), 2=right(w), 3=jump(space), 4=jump-left, 5=jump-right.
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(10, dtype=np.float32),
            high=np.ones(10, dtype=np.float32),
            dtype=np.float32,
        )

        self.client = FuseMLClient(socket_path=socket_path, timeout_s=self.socket_timeout_s)
        self._configure_mode()

        self.step_count = 0
        self.prev_state: Optional[ManicState] = None
        self.keys_at_reset = 0
        self.first_key_achieved = False
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.configured_lethal_attrs_by_level: Dict[int, Set[int]] = {}
        self.configured_lethal_attr_groups_by_level: Dict[int, Tuple[Set[int], Set[int]]] = {}
        self._rewarded_under_lethal_cells: Set[Tuple[int, int, int]] = set()
        self._last_dynamic_lethal_cells_count = 0
        self._last_repeat_signature: Optional[Tuple[int, ...]] = None
        self._repeat_signature_count = 0
        self._supports_episode_step_keys: Optional[bool] = None
        self._last_action_key_chord = "-"
        self._last_action_used: Optional[int] = None
        self._walk_flip_request_count = 0

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

    def _apply_walk_direction_hysteresis(self, action: int) -> Tuple[int, bool, str]:
        if action not in (1, 2):
            self._walk_flip_request_count = 0
            return action, False, "-"
        prev_action = self._last_action_used
        if prev_action not in (1, 2):
            self._walk_flip_request_count = 0
            return action, False, "-"
        if action == prev_action:
            self._walk_flip_request_count = 0
            return action, False, "-"

        self._walk_flip_request_count += 1
        if self._walk_flip_request_count < WALK_FLIP_HYSTERESIS_STEPS:
            reason = (
                f"walk_hysteresis_hold(prev={prev_action},requested={action},"
                f"count={self._walk_flip_request_count})"
            )
            return int(prev_action), True, reason

        self._walk_flip_request_count = 0
        reason = f"walk_hysteresis_allow_flip(prev={prev_action},requested={action})"
        return action, False, reason

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
        info = self.client.get_info()

        self.visited_cells.clear()
        self.visited_cells.update(self._covered_cells(state.willy_x_px, state.willy_y_px))
        state.coverage_ratio = self._coverage_ratio()

        self.prev_state = state
        self.keys_at_reset = state.keys_remaining
        self.first_key_achieved = False
        self.step_count = 0
        self._last_repeat_signature = None
        self._repeat_signature_count = 0
        self._last_action_used = None
        self._walk_flip_request_count = 0
        self._rewarded_under_lethal_cells.clear()

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
        _, fixed_attrs = self._configured_lethal_attr_groups_for_level(safety_level)
        fixed_lethal_cells_before = self._dynamic_cells_for_attrs(attr_buffer_before, fixed_attrs)
        dynamic_lethal_cells_before = self._dynamic_lethal_cells_for_level(safety_level, attr_buffer_before)
        key_cells_before = self._active_item_cells_for_level(safety_level)
        self._last_dynamic_lethal_cells_count = len(dynamic_lethal_cells_before)
        jump_above_blocked = False
        jump_above_reason = "-"
        local_jump_blocked = False
        local_jump_forced = False
        local_jump_reason = "-"
        jump_gate_blocked = False
        jump_gate_reason = "-"
        walk_hysteresis_applied = False
        walk_hysteresis_reason = "-"
        safety_blocked = False
        safety_reason = "-"
        safety_overridden_by_jump_gate = False
        if input_suppressed_for_airborne:
            action_used = 0
            safety_reason = f"airborne_input_suppressed(status=0x{airborne_status_before:02X})"
        else:
            if self.random_action_prob > 0.0 and self._rng_random() < self.random_action_prob:
                action_used = self._rng_integers(0, self.action_space.n)
                random_action_applied = True
            action_used, local_jump_blocked, local_jump_forced, local_jump_reason = self._apply_local_jump_rules(
                state_before,
                action_used,
                fixed_lethal_cells_before,
                dynamic_lethal_cells_before,
                key_cells_before,
            )
            jump_above_blocked = "block_jump_up_fixed_above" in local_jump_reason
            jump_above_reason = local_jump_reason
            if not local_jump_forced:
                action_used, jump_gate_blocked, jump_gate_reason = self._apply_directional_jump_gate(
                    state_before, action_used, attr_buffer_before
                )
            else:
                jump_gate_reason = "jump_gate_skipped_for_key_forced_jump"

            action_used, walk_hysteresis_applied, walk_hysteresis_reason = self._apply_walk_direction_hysteresis(
                action_used
            )
            action_used, safety_blocked, safety_reason = self._apply_safety_shield(
                state_before, action_used, attr_buffer_before
            )

        # Jump is a one-shot input: send it for one frame only, then suppress inputs while airborne.
        if action_used in (3, 4, 5) or input_suppressed_for_airborne:
            frames_used = 1

        ep = self._episode_step_with_action(action_used, frames_used)
        state = self._read_state()
        attr_buffer_after = self._read_attr_buffer()
        dynamic_lethal_cells_after = self._dynamic_lethal_cells_for_level(state.level, attr_buffer_after)
        airborne_status_after = self._read_u8(WILLY_AIRBORNE_ADDR)
        life_lost_this_step = self.prev_state is not None and state.lives < self.prev_state.lives

        first_key_achieved_now = state.keys_remaining < self.keys_at_reset
        (
            reward,
            objective_reward,
            hazard_reward,
            pathing_reward,
            repeat_penalty,
            walk_reward,
            under_lethal_reward,
            repeat_detected,
            repeat_count,
            repeat_reason,
            walk_reason,
            under_lethal_reason,
        ) = self._compute_reward(
            state,
            ep["reward"],
            first_key_achieved_now,
            action_used,
            self._last_action_used,
            dynamic_lethal_cells_after,
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
            "jump_above_blocked": jump_above_blocked,
            "jump_above_reason": jump_above_reason,
            "local_jump_blocked": local_jump_blocked,
            "local_jump_forced": local_jump_forced,
            "local_jump_reason": local_jump_reason,
            "jump_gate_blocked": jump_gate_blocked,
            "jump_gate_reason": jump_gate_reason,
            "walk_hysteresis_applied": walk_hysteresis_applied,
            "walk_hysteresis_reason": walk_hysteresis_reason,
            "safety_blocked": safety_blocked,
            "safety_reason": safety_reason,
            "safety_overridden_by_jump_gate": safety_overridden_by_jump_gate,
            "known_lethal_cells": self._last_dynamic_lethal_cells_count,
            "known_lethal_attrs": len(self._configured_lethal_attrs_for_level(safety_level)),
            "fixed_lethal_cells": len(fixed_lethal_cells_before),
            "visible_key_cells": len(key_cells_before),
            "life_lost_this_step": life_lost_this_step,
            "first_key_mode": self.first_key_mode,
            "first_key_achieved": first_key_achieved_now,
            "first_key_terminated": first_key_terminated,
            "objective_reward": objective_reward,
            "hazard_reward": hazard_reward,
            "pathing_reward": pathing_reward,
            "repeat_penalty": repeat_penalty,
            "walk_reward": walk_reward,
            "walk_reason": walk_reason,
            "under_lethal_reward": under_lethal_reward,
            "under_lethal_reason": under_lethal_reason,
            "repeat_detected": repeat_detected,
            "repeat_count": repeat_count,
            "repeat_reason": repeat_reason,
            "visited_cells": len(self.visited_cells),
            "exit_distance_px": self._distance_to_exit_px(state),
        }

        if first_key_achieved_now:
            self.first_key_achieved = True

        self._last_action_used = int(action_used)
        self.prev_state = state
        return self._normalize_observation(state.to_observation()), reward, terminated, truncated, info

    def close(self) -> None:
        self.client.close()
