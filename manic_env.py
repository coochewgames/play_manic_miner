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

from manic_data import (
    ManicDataMixin, ManicState, WILLY_AIRBORNE_ADDR, AIR_SUPPLY_ADDR, AIR_SUPPLY_MAX,
    CELL_SIZE_PX, WILLY_SIZE_PX, SCREEN_CELLS_W, SCREEN_CELLS_H,
)
from ml_client import FuseMLClient

PLAY_MODULE_NAME = os.environ.get("MANIC_PLAY_MODULE", "manic_play")
_play_module = importlib.import_module(PLAY_MODULE_NAME)
ManicPlayMixin = _play_module.ManicPlayMixin
PATHING_NEW_CELL_REWARD = _play_module.PATHING_NEW_CELL_REWARD
FRONTIER_APPROACH_COEF = _play_module.FRONTIER_APPROACH_COEF
logger = logging.getLogger(__name__)

ACTION_KEY_CHORDS: Dict[int, str] = {
    0: "-",
    1: "q",
    2: "w",
    3: "space",
    4: "q+space",
    5: "w+space",
}

WALK_FLIP_HYSTERESIS_STEPS = 2
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
        first_key_success_bonus: float = 120.0,
        num_keys_mode: int = 0,
        safety_shield: bool = True,
        pathing_new_cell_reward: float = PATHING_NEW_CELL_REWARD,
        key_collect_reward: float = 80.0,
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
        # num_keys_mode generalises first_key_mode: episode terminates after N
        # keys are collected and a success bonus fires at the Nth key.
        # first_key_mode=True is treated as num_keys_mode=1 for backward compat.
        _nkm = int(num_keys_mode)
        if first_key_mode and _nkm == 0:
            _nkm = 1
        self.num_keys_mode = _nkm
        self.first_key_mode = _nkm == 1  # kept for info/log compat
        self.first_key_success_bonus = float(first_key_success_bonus)
        self.infinite_air = bool(infinite_air)
        self.safety_shield = bool(safety_shield)
        self.pathing_new_cell_reward = float(pathing_new_cell_reward)
        self.key_collect_reward = float(key_collect_reward)

        # 0=no-op, 1=left(q), 2=right(w), 3=jump(space), 4=jump-left, 5=jump-right.
        self.action_space = gym.spaces.Discrete(6)
        # 10 base features + 12 entity features + 3 exploration features + 1 airborne flag:
        #   guardian (dx, dy, dist), nasty (dx, dy, dist), exit_portal (dx, dy, dist),
        #   lethal_overhead, lethal_left, lethal_right,
        #   nearest_unvisited (dx, dy, dist), is_airborne
        self.observation_space = gym.spaces.Box(
            low=np.zeros(26, dtype=np.float32),
            high=np.ones(26, dtype=np.float32),
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
        self._solid_platform_cells_by_level: Dict[int, Set[Tuple[int, int]]] = {}
        self._last_dynamic_lethal_cells_count = 0
        self._last_repeat_signature: Optional[Tuple[int, ...]] = None
        self._repeat_signature_count = 0
        self._supports_episode_step_keys: Optional[bool] = None
        self._last_action_key_chord = "-"
        self._last_action_used: Optional[int] = None
        self._walk_flip_request_count = 0
        self._tracked_key_level: Optional[int] = None
        self._tracked_keys_remaining = 0
        self._tracked_key_score_anchor = 0
        self._action_mask: np.ndarray = np.ones(6, dtype=bool)
        self._pending_key_reward: float = 0.0
        self._episode_min_keys_remaining: int = 0

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
                # At most one item can realistically be collected per env step.
                collected = min(self._tracked_keys_remaining, score_delta // KEY_SCORE_INCREMENT, 1)
                self._tracked_keys_remaining -= int(collected)
                self._tracked_key_score_anchor += int(collected) * KEY_SCORE_INCREMENT
            elif score_delta < 0:
                # Score should not normally go backwards; re-anchor if it does.
                self._tracked_key_score_anchor = state.score

        state.keys_remaining = max(0, min(configured, int(self._tracked_keys_remaining)))
        state.portal_open = 1 if state.keys_remaining == 0 else 0

    def _nearest_entity_features(
        self, willy_x_px: int, willy_y_px: int, entity_cells: Set[Tuple[int, int]]
    ) -> Tuple[float, float, float]:
        """Return (dx_norm, dy_norm, dist_norm) to the nearest cell in entity_cells.

        All values are normalised to [0, 1].  dx/dy are centred at 0.5
        (0.5 = same position, <0.5 = entity is to the left/above).
        dist is 0.0 when on top of the entity, 1.0 at maximum screen distance.
        If entity_cells is empty, returns (0.5, 0.5, 1.0) — no information.
        """
        if not entity_cells:
            return 0.5, 0.5, 1.0
        willy_cx = willy_x_px // CELL_SIZE_PX
        willy_cy = willy_y_px // CELL_SIZE_PX
        best_dist = float("inf")
        best_dx = 0.0
        best_dy = 0.0
        for ex, ey in entity_cells:
            dx = ex - willy_cx
            dy = ey - willy_cy
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best_dist = dist
                best_dx = float(dx)
                best_dy = float(dy)
        dx_norm = float(np.clip((best_dx + SCREEN_CELLS_W) / (2.0 * SCREEN_CELLS_W), 0.0, 1.0))
        dy_norm = float(np.clip((best_dy + SCREEN_CELLS_H) / (2.0 * SCREEN_CELLS_H), 0.0, 1.0))
        dist_norm = float(np.clip(best_dist / (SCREEN_CELLS_W + SCREEN_CELLS_H), 0.0, 1.0))
        return dx_norm, dy_norm, dist_norm

    def _nearest_unvisited_dist(self, willy_x_px: int, willy_y_px: int) -> float:
        """Return the raw Manhattan distance (in cells) to the nearest unvisited cell.
        Returns 0.0 if there are no unvisited cells."""
        willy_cx = willy_x_px // CELL_SIZE_PX
        willy_cy = willy_y_px // CELL_SIZE_PX
        best_dist = float("inf")
        for cx in range(SCREEN_CELLS_W):
            for cy in range(SCREEN_CELLS_H):
                if (cx, cy) not in self.visited_cells:
                    dist = abs(cx - willy_cx) + abs(cy - willy_cy)
                    if dist < best_dist:
                        best_dist = dist
        return 0.0 if best_dist == float("inf") else best_dist

    def _nearest_unvisited_features(self, willy_x_px: int, willy_y_px: int) -> Tuple[float, float, float]:
        """Return (dx_norm, dy_norm, dist_norm) to the nearest unvisited cell.

        Uses the same normalisation convention as _nearest_entity_features.
        If the screen is fully visited, returns (0.5, 0.5, 0.0).
        """
        willy_cx = willy_x_px // CELL_SIZE_PX
        willy_cy = willy_y_px // CELL_SIZE_PX
        best_dist = float("inf")
        best_dx = 0.0
        best_dy = 0.0
        for cx in range(SCREEN_CELLS_W):
            for cy in range(SCREEN_CELLS_H):
                if (cx, cy) not in self.visited_cells:
                    dx = cx - willy_cx
                    dy = cy - willy_cy
                    dist = abs(dx) + abs(dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_dx = float(dx)
                        best_dy = float(dy)
        if best_dist == float("inf"):
            return 0.5, 0.5, 0.0  # fully explored
        dx_norm = float(np.clip((best_dx + SCREEN_CELLS_W) / (2.0 * SCREEN_CELLS_W), 0.0, 1.0))
        dy_norm = float(np.clip((best_dy + SCREEN_CELLS_H) / (2.0 * SCREEN_CELLS_H), 0.0, 1.0))
        dist_norm = float(np.clip(best_dist / (SCREEN_CELLS_W + SCREEN_CELLS_H), 0.0, 1.0))
        return dx_norm, dy_norm, dist_norm

    def _exit_direction_features(self, state: ManicState) -> Tuple[float, float, float]:
        """Return (dx_norm, dy_norm, dist_norm) from Willy to the exit portal.

        Always present in the observation regardless of whether the portal is
        open — the agent can use exit location as a landmark for route planning
        even while collecting keys.  Returns (0.5, 0.5, 1.0) if the portal
        position cannot be read from memory.
        """
        portal_xy = self._read_portal_xy_for_level(state.level)
        if portal_xy is None:
            return 0.5, 0.5, 1.0
        willy_cx = state.willy_x_px // CELL_SIZE_PX
        willy_cy = state.willy_y_px // CELL_SIZE_PX
        exit_cx = portal_xy[0] // CELL_SIZE_PX
        exit_cy = portal_xy[1] // CELL_SIZE_PX
        dx = exit_cx - willy_cx
        dy = exit_cy - willy_cy
        dist = abs(dx) + abs(dy)
        dx_norm = float(np.clip((dx + SCREEN_CELLS_W) / (2.0 * SCREEN_CELLS_W), 0.0, 1.0))
        dy_norm = float(np.clip((dy + SCREEN_CELLS_H) / (2.0 * SCREEN_CELLS_H), 0.0, 1.0))
        dist_norm = float(np.clip(dist / (SCREEN_CELLS_W + SCREEN_CELLS_H), 0.0, 1.0))
        return dx_norm, dy_norm, dist_norm

    def action_masks(self) -> np.ndarray:
        """Return the cached action mask for use with MaskablePPO."""
        return self._action_mask.copy()

    def _compute_action_mask(
        self, state: ManicState, attr_buffer: bytes, airborne_status: int
    ) -> np.ndarray:
        """Compute which actions are valid in the current state.

        During airborne, only no-op (0) is valid — all inputs are suppressed.
        On the ground, actions blocked by either local jump rules or the safety
        shield are masked out so MaskablePPO never attributes rewards to actions
        that would be silently replaced.  At least no-op (0) is always unmasked.
        """
        mask = np.ones(6, dtype=bool)
        if self._is_airborne_status_active(airborne_status):
            mask[1:] = False
            return mask
        if not self.safety_shield:
            return mask

        # Mirrors the cell-set computation in step() so both paths agree.
        safety_level = state.level
        moving_attrs, fixed_attrs = self._configured_lethal_attr_groups_for_level(safety_level)
        fixed_lethal = self._dynamic_cells_for_attrs(attr_buffer, fixed_attrs)
        moving_lethal = self._dynamic_cells_for_attrs(attr_buffer, moving_attrs)
        any_lethal = moving_lethal | self._dynamic_cells_for_attrs(attr_buffer, fixed_attrs)
        key_cells = self._active_item_cells_for_level(safety_level, attr_buffer)

        for action in range(6):
            # Local jump rules (lethal proximity + arc projection checks).
            _, blocked, _, _ = self._apply_local_jump_rules(
                state, action, fixed_lethal, any_lethal, key_cells, moving_lethal
            )
            if blocked:
                mask[action] = False
                continue
            # Safety shield (arc/walk projection against all dynamic lethals).
            _, shielded, _ = self._apply_safety_shield(state, action, attr_buffer)
            if shielded:
                mask[action] = False

        if not mask.any():
            mask[0] = True
        return mask

    def _build_observation(
        self, state: ManicState, attr_buffer: bytes, airborne_status: int = 0
    ) -> np.ndarray:
        """Build the full 26-feature observation for the current state."""
        base_obs = self._normalize_observation(state.to_observation())

        moving_attrs, fixed_attrs = self._configured_lethal_attr_groups_for_level(state.level)
        guardian_cells = self._dynamic_cells_for_attrs(attr_buffer, moving_attrs)
        nasty_cells = self._dynamic_cells_for_attrs(attr_buffer, fixed_attrs)
        all_lethal_cells = guardian_cells | nasty_cells

        wx, wy = state.willy_x_px, state.willy_y_px
        g_dx, g_dy, g_dist = self._nearest_entity_features(wx, wy, guardian_cells)
        n_dx, n_dy, n_dist = self._nearest_entity_features(wx, wy, nasty_cells)
        e_dx, e_dy, e_dist = self._exit_direction_features(state)

        lethal_overhead = 1.0 if self._is_under_lethal(state, all_lethal_cells) else 0.0
        lethal_left = 1.0 if bool(self._front_cells_for_direction(state, -1) & all_lethal_cells) else 0.0
        lethal_right = 1.0 if bool(self._front_cells_for_direction(state, 1) & all_lethal_cells) else 0.0

        uv_dx, uv_dy, uv_dist = self._nearest_unvisited_features(state.willy_x_px, state.willy_y_px)
        is_airborne = 1.0 if self._is_airborne_status_active(airborne_status) else 0.0

        entity_obs = np.array(
            [g_dx, g_dy, g_dist, n_dx, n_dy, n_dist, e_dx, e_dy, e_dist,
             lethal_overhead, lethal_left, lethal_right,
             uv_dx, uv_dy, uv_dist, is_airborne],
            dtype=np.float32,
        )
        return np.concatenate([base_obs, entity_obs])

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
        attr_buffer = self._read_attr_buffer()
        airborne_status = self._read_u8(WILLY_AIRBORNE_ADDR)

        # Reset visited cells every episode so there is always dense pathing
        # signal.  Combined with frontier approach shaping, the agent earns
        # pathing reward for the known lower cells then is guided toward the
        # frontier by the approach reward.  Clear on level transition too.
        prev_level = self.prev_state.level if self.prev_state is not None else None
        if prev_level is not None and state.level == prev_level:
            pass  # same level: reset per episode (done unconditionally below)
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
        self._tracked_key_level = state.level
        self._tracked_keys_remaining = state.keys_remaining
        self._tracked_key_score_anchor = state.score
        self._pending_key_reward = 0.0
        self._episode_min_keys_remaining = state.keys_remaining
        self._action_mask = self._compute_action_mask(state, attr_buffer, airborne_status)

        info["state"] = state.__dict__.copy()
        info["first_key_mode"] = self.first_key_mode
        return self._build_observation(state, attr_buffer, airborne_status), info

    def step(self, action: int):
        self.step_count += 1

        action_used = int(action)
        frames_used = self.frames_per_action
        random_action_applied = False
        state_before = self.prev_state if self.prev_state is not None else self._read_state()
        _prev_uv_dist = self._nearest_unvisited_dist(state_before.willy_x_px, state_before.willy_y_px)
        airborne_status_before = self._read_u8(WILLY_AIRBORNE_ADDR)
        input_suppressed_for_airborne = self._is_airborne_status_active(airborne_status_before)
        attr_buffer_before = self._read_attr_buffer()
        safety_level = state_before.level
        moving_attrs, fixed_attrs = self._configured_lethal_attr_groups_for_level(safety_level)
        fixed_lethal_cells_before = self._dynamic_cells_for_attrs(attr_buffer_before, fixed_attrs)
        moving_lethal_cells_before = self._dynamic_cells_for_attrs(attr_buffer_before, moving_attrs)
        dynamic_lethal_cells_before = moving_lethal_cells_before | self._dynamic_cells_for_attrs(attr_buffer_before, fixed_attrs)
        key_cells_before = self._active_item_cells_for_level(safety_level, attr_buffer_before)
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
                moving_lethal_cells_before,
            )
            jump_above_blocked = "block_jump_up_fixed_above" in local_jump_reason
            jump_above_reason = local_jump_reason

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
        if self.infinite_air:
            self.client.write_bytes(AIR_SUPPLY_ADDR, bytes([AIR_SUPPLY_MAX]))
        state = self._read_state()
        self._stabilize_keys_remaining(state)
        attr_buffer_after = self._read_attr_buffer()
        airborne_status_after = self._read_u8(WILLY_AIRBORNE_ADDR)
        life_lost_this_step = self.prev_state is not None and state.lives < self.prev_state.lives

        first_key_achieved_now = state.keys_remaining < self.keys_at_reset
        (
            reward,
            objective_reward,
            hazard_reward,
            pathing_reward,
            repeat_penalty,
            repeat_detected,
            repeat_count,
            repeat_reason,
        ) = self._compute_reward(
            state,
            ep["reward"],
            first_key_achieved_now,
            action_used,
        )

        # Frontier approach shaping: reward reducing Manhattan distance to the
        # nearest unvisited cell.  Guides the agent toward unvisited territory
        # when there is no pathing reward (all reachable cells already visited).
        if FRONTIER_APPROACH_COEF > 0.0:
            _curr_uv_dist = self._nearest_unvisited_dist(state.willy_x_px, state.willy_y_px)
            _approach = (_prev_uv_dist - _curr_uv_dist) * FRONTIER_APPROACH_COEF
            reward += _approach

        bridge_done = bool(ep["done"])
        keys_collected_ep = self.keys_at_reset - state.keys_remaining
        num_keys_terminated = (self.num_keys_mode > 0 and keys_collected_ep >= self.num_keys_mode)
        first_key_terminated = num_keys_terminated and self.num_keys_mode == 1  # compat
        terminated = bridge_done or state.lives == 0 or num_keys_terminated
        truncated = self.step_count >= self.max_steps

        # Deferred key reward: keys collected mid-air only credited on safe
        # landing.  Prevents the agent learning suicidal airborne key-grabs
        # (e.g. jumping into a key from the wrong direction then dying on landing).
        was_airborne = self._is_airborne_status_active(airborne_status_before)
        is_still_airborne = self._is_airborne_status_active(airborne_status_after)
        # Use monotone floor (same logic as _compute_reward) so airborne
        # deferral doesn't double-count respawned keys after a death.
        if self.prev_state is not None:
            _floor = min(self.prev_state.keys_remaining, self._episode_min_keys_remaining)
            keys_collected_this_step = max(0, _floor - state.keys_remaining)
        else:
            keys_collected_this_step = 0
        # Release or cancel pending reward from a previous airborne collection.
        if self._pending_key_reward != 0.0:
            if life_lost_this_step:
                self._pending_key_reward = 0.0  # died before landing — cancel
            elif not is_still_airborne:
                reward += self._pending_key_reward  # landed safely — credit
                objective_reward += self._pending_key_reward
                self._pending_key_reward = 0.0
            # else: still mid-jump — hold pending
        # Defer key reward for keys collected THIS step while airborne,
        # unless the episode terminates on this key (then credit immediately).
        if was_airborne and keys_collected_this_step > 0 and not num_keys_terminated:
            defer_amount = self.key_collect_reward * float(keys_collected_this_step)
            _nkm_target = self.num_keys_mode if self.num_keys_mode > 0 else 1
            _distinct_now = self.keys_at_reset - self._episode_min_keys_remaining
            _distinct_prev = _distinct_now - keys_collected_this_step
            if _distinct_prev < _nkm_target <= _distinct_now:
                defer_amount += self.first_key_success_bonus
            reward -= defer_amount
            objective_reward -= defer_amount
            if not life_lost_this_step:
                self._pending_key_reward += defer_amount
            # life_lost same step: reward already reduced, pending stays 0

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
            "configured_keys_for_level": self._count_configured_items_for_level(state.level),
            "first_key_mode": self.first_key_mode,
            "first_key_achieved": first_key_achieved_now,
            "first_key_terminated": first_key_terminated,
            "objective_reward": objective_reward,
            "hazard_reward": hazard_reward,
            "pathing_reward": pathing_reward,
            "repeat_penalty": repeat_penalty,
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
        self._action_mask = self._compute_action_mask(state, attr_buffer_after, airborne_status_after)
        return self._build_observation(state, attr_buffer_after, airborne_status_after), reward, terminated, truncated, info

    def close(self) -> None:
        self.client.close()

    def quit_emulator(self) -> None:
        try:
            self.client.quit()
        except Exception as exc:
            logger.warning("Failed to send QUIT to Fuse ML bridge: %s", exc)
