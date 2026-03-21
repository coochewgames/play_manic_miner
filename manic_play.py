#!/usr/bin/env python3
"""Gameplay policy and reward-shaping logic for Manic Miner."""

from __future__ import annotations

from typing import Optional, Set, Tuple

from manic_data import CELL_SIZE_PX, SCREEN_CELLS_H, SCREEN_CELLS_W, WILLY_SIZE_PX, ManicState


# Objectives (positive)
KEY_COLLECT_REWARD = 80.0
LEVEL_COMPLETE_REWARD = 1000.0
ALL_KEYS_CLEARED_REWARD = 250.0
EXIT_APPROACH_REWARD_PER_PX = 0.08
KEY_APPROACH_REWARD_PER_PX = 0.2   # reward per px closer to nearest uncollected key
KEY_APPROACH_LETHAL_MARGIN_CELLS = 4  # skip keys within this many cells of a fixed nasty

# Hazards (negative)
LIFE_LOSS_PENALTY = 80.0
AIR_DECAY_PENALTY_COEF = 0.0002

# Pathing / anti-loop
PATHING_NEW_CELL_REWARD = 8.0
REPEAT_TRANSITION_TOLERANCE = 1
REPEAT_TRANSITION_PENALTY = 0.05
REPEAT_TRANSITION_PENALTY_MAX = 0.3
FRONTIER_APPROACH_COEF = 0.8   # reward per cell closer to nearest unvisited cell
WALK_PROGRESS_REWARD_PER_PX = 0.0
WALK_STREAK_BONUS = 0.0
WALK_DIRECTION_FLIP_PENALTY = 0.0
WALK_NO_PROGRESS_PENALTY = 0.0
WALK_UNDER_LETHAL_REWARD_PER_CELL = 0.0
JUMP_UNDER_LETHAL_PENALTY = 0.0
WALK_UNDER_LETHAL_VERTICAL_LOOKAHEAD_CELLS = 2
WALK_UNDER_LETHAL_SIDE_MARGIN_CELLS = 1

# Safety projection thresholds
SAFETY_BLOCK_INTERSECTION_MIN_CELLS = 1
SAFETY_JUMP_BLOCK_INTERSECTION_MIN_CELLS = 4
SAFETY_WALK_LOOKAHEAD_PX = 2
JUMP_PHASE_HORIZONTAL_PX = 2
# Cumulative rise (px, upward positive) at each of the 18 game frames of the
# arc.  Derived from ZX Spectrum assembly at $8ABB: each frame n (0–17) applies
# delta_units = (n & 0xFE) - 8; delta_px = delta_units / 2 (negative = upward).
# Peak rise = 20 px (2.5 attribute cells), confirmed against emulator.
JUMP_PHASES_RISE_Y_PX = (4, 8, 11, 14, 16, 18, 19, 20)
JUMP_ARC_Y_FROM_START_PX = (
    4,
    8,
    11,
    14,
    16,
    18,
    19,
    20,
    20,
    20,
    19,
    18,
    16,
    14,
    11,
    8,
    4,
    0,
)


class ManicPlayMixin:
    """Methods that decide actions and calculate rewards."""

    def _jump_arc_offsets(self, horizontal_sign: int) -> Tuple[Tuple[int, int], ...]:
        offsets = []
        for phase_idx, rise_px in enumerate(JUMP_ARC_Y_FROM_START_PX, start=1):
            x_px = horizontal_sign * (phase_idx * JUMP_PHASE_HORIZONTAL_PX)
            y_px = -rise_px
            offsets.append((x_px, y_px))
        return tuple(offsets)

    def _sample_offsets_for_action(self, action: int) -> Tuple[Tuple[int, int], ...]:
        if action == 1:
            return ((-SAFETY_WALK_LOOKAHEAD_PX, 0),)
        if action == 2:
            return ((SAFETY_WALK_LOOKAHEAD_PX, 0),)
        if action == 3:
            return tuple((0, -rise_px) for rise_px in JUMP_ARC_Y_FROM_START_PX)
        if action == 4:
            return self._jump_arc_offsets(-1)
        if action == 5:
            return self._jump_arc_offsets(1)
        return ((0, 0),)

    def _projected_cells_for_action(
        self,
        state: ManicState,
        action: int,
        solid_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Set[Tuple[int, int]]:
        """Project all cells Willy occupies along the action arc.

        When solid_cells is provided (the precomputed solid-platform cell set
        for the level), ascending arc phases are checked against it.  If the
        top of Willy's projected sprite enters a solid platform cell the game
        engine would cut the jump short there, so lethals beyond that phase
        are unreachable and must not block the jump.
        """
        cells: Set[Tuple[int, int]] = set()
        prev_rise = 0
        for dx_px, dy_px in self._sample_offsets_for_action(action):
            x_px = max(0, min(255, state.willy_x_px + dx_px))
            y_px = max(0, min(191, state.willy_y_px + dy_px))
            cells.update(self._covered_cells(x_px, y_px))
            if solid_cells is not None and dy_px < 0:
                rise = -dy_px
                if rise > prev_rise:  # still ascending
                    prev_rise = rise
                    top_cy = y_px // CELL_SIZE_PX
                    cx0 = x_px // CELL_SIZE_PX
                    cx1 = (x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX
                    for cx in range(cx0, cx1 + 1):
                        if (cx, top_cy) in solid_cells:
                            return cells  # solid platform ceiling truncates arc
        return cells

    def _apply_local_jump_rules(
        self,
        state: ManicState,
        action: int,
        fixed_lethal_cells: Set[Tuple[int, int]],
        any_lethal_cells: Set[Tuple[int, int]],
        key_cells: Set[Tuple[int, int]],
        moving_lethal_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[int, bool, bool, str]:
        blocked_actions: Set[int] = set()
        reasons = []

        # Only block jump-up when a fixed nasty is in the cell directly above —
        # clearly unreachable any other way.  All arc-based blocking for
        # directional jumps is handled by the safety shield (threshold = 2 cells)
        # so the policy can still attempt jumps that only graze a lethal.
        fixed_above = bool(self._cells_above_willy(state) & fixed_lethal_cells)
        if fixed_above:
            blocked_actions.add(3)
            reasons.append("block_jump_up_fixed_above")

        key_above_right = bool(self._cells_above_right_willy(state) & key_cells)
        key_above_left = bool(self._cells_above_left_willy(state) & key_cells)
        key_above = bool(self._cells_above_willy(state) & key_cells)

        if key_above_right and 5 not in blocked_actions:
            return 5, False, True, "force_jump_right_key_above_right"
        if key_above_left and 4 not in blocked_actions:
            return 4, False, True, "force_jump_left_key_above_left"
        if key_above and 3 not in blocked_actions:
            return 3, False, True, "force_jump_up_key_above"

        if action in blocked_actions:
            fallback = 0
            if action == 4:
                fallback = 1
            elif action == 5:
                fallback = 2
            reason = ",".join(reasons) if reasons else "jump_block_local_rule"
            return fallback, True, False, f"{reason}(action={action},fallback={fallback})"

        return action, False, False, ""

    def _apply_directional_jump_gate(
        self, state: ManicState, action: int, attr_buffer: bytes
    ) -> Tuple[int, bool, str]:
        if action not in (4, 5):
            return action, False, ""

        direction = -1 if action == 4 else 1
        moving_attrs, fixed_attrs = self._configured_lethal_attr_groups_for_level(state.level)
        moving_cells = self._dynamic_cells_for_attrs(attr_buffer, moving_attrs)
        fixed_cells = self._dynamic_cells_for_attrs(attr_buffer, fixed_attrs)
        front_cells = self._front_cells_for_direction(state, direction)
        fixed_ahead = bool(front_cells & fixed_cells)
        moving_ahead = bool(front_cells & moving_cells)
        solid_cells = self._solid_platform_cells_for_level(state.level)
        explores_unvisited = bool(self._projected_cells_for_action(state, action, solid_cells) - self.visited_cells)

        if fixed_ahead:
            return action, False, "jump_gate_allow_fixed_ahead"
        if moving_ahead:
            return action, False, "jump_gate_allow_moving_ahead"
        if explores_unvisited:
            return action, False, "jump_gate_allow_explore_unvisited"

        fallback = 1 if action == 4 else 2
        return fallback, True, f"jump_gate_block(action={action},fallback={fallback})"

    def _arc_landing_cells(
        self,
        state: ManicState,
        action: int,
        solid_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Set[Tuple[int, int]]:
        """Return cells occupied at the landing position of a jump arc.

        Returns an empty set if:
        - action is not a jump
        - a solid ceiling truncates the arc before Willy lands

        This lets callers apply a stricter (threshold=1) landing check
        independently of the arc-body threshold.
        """
        offsets = self._sample_offsets_for_action(action)
        if not offsets:
            return set()
        prev_rise = 0
        for i, (dx_px, dy_px) in enumerate(offsets):
            x_px = max(0, min(255, state.willy_x_px + dx_px))
            y_px = max(0, min(191, state.willy_y_px + dy_px))
            if solid_cells is not None and dy_px < 0:
                rise = -dy_px
                if rise > prev_rise:
                    prev_rise = rise
                    top_cy = y_px // CELL_SIZE_PX
                    cx0 = x_px // CELL_SIZE_PX
                    cx1 = (x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX
                    for cx in range(cx0, cx1 + 1):
                        if (cx, top_cy) in solid_cells:
                            return set()  # arc truncated before landing
            if i == len(offsets) - 1:
                return self._covered_cells(x_px, y_px)
        return set()

    def _action_hits_dynamic_lethal(
        self,
        state: ManicState,
        action: int,
        dynamic_lethal_cells: Set[Tuple[int, int]],
        solid_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> bool:
        if action == 0:
            return False
        if not dynamic_lethal_cells:
            return False
        if action == 1:
            overlap = self._front_cells_for_direction(state, -1) & dynamic_lethal_cells
            return len(overlap) >= SAFETY_BLOCK_INTERSECTION_MIN_CELLS
        if action == 2:
            overlap = self._front_cells_for_direction(state, 1) & dynamic_lethal_cells
            return len(overlap) >= SAFETY_BLOCK_INTERSECTION_MIN_CELLS
        projected_cells = self._projected_cells_for_action(state, action, solid_cells)
        overlap = projected_cells & dynamic_lethal_cells
        if len(overlap) >= SAFETY_JUMP_BLOCK_INTERSECTION_MIN_CELLS:
            return True
        # Landing-position check: if Willy would land directly on a lethal
        # that is 1 cell overlap, block it — landing is deterministic death.
        # Only applies when the arc completes (not ceiling-truncated).
        landing_cells = self._arc_landing_cells(state, action, solid_cells)
        if landing_cells & dynamic_lethal_cells:
            return True
        return False

    def _safe_fallback_action(
        self, state: ManicState, blocked_action: int, dynamic_lethal_cells: Set[Tuple[int, int]]
    ) -> int:
        if blocked_action == 1:
            candidates = [4, 3, 2, 0, 5, 1]
        elif blocked_action == 2:
            candidates = [5, 3, 1, 0, 4, 2]
        elif blocked_action in (3, 4, 5):
            candidates = [1, 2, 0]
        else:
            candidates = [0, 1, 2]
        for candidate in candidates:
            if not self._action_hits_dynamic_lethal(state, candidate, dynamic_lethal_cells):
                return candidate
        return 0

    def _apply_safety_shield(
        self, state: ManicState, action: int, attr_buffer: Optional[bytes] = None
    ) -> Tuple[int, bool, str]:
        if not self.safety_shield:
            self._last_dynamic_lethal_cells_count = 0
            return action, False, ""
        if attr_buffer is None:
            attr_buffer = self._read_attr_buffer()

        dynamic_lethal_cells = self._dynamic_lethal_cells_for_level(state.level, attr_buffer)
        self._last_dynamic_lethal_cells_count = len(dynamic_lethal_cells)
        solid_cells = self._solid_platform_cells_for_level(state.level)

        if not self._action_hits_dynamic_lethal(state, action, dynamic_lethal_cells, solid_cells):
            return action, False, ""
        fallback = self._safe_fallback_action(state, action, dynamic_lethal_cells)
        return fallback, True, f"configured_lethal_block(action={action},fallback={fallback})"

    def _key_approach_targets(
        self,
        key_cells: Set[Tuple[int, int]],
        fixed_lethal_cells: Set[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        """Return the subset of key_cells safe to use as approach targets.

        Filters out key cells where any fixed nasty is within
        KEY_APPROACH_LETHAL_MARGIN_CELLS (Manhattan distance in cells).
        This prevents the approach reward from guiding the agent toward
        keys that are dangerous to reach (e.g. a key near a conveyor belt
        that pushes Willy into a nasty).

        Falls back to the full key_cells set if none pass the filter so
        there is always a valid approach target.
        """
        if not fixed_lethal_cells or KEY_APPROACH_LETHAL_MARGIN_CELLS <= 0:
            return key_cells
        safe: Set[Tuple[int, int]] = set()
        for kx, ky in key_cells:
            near_lethal = any(
                abs(kx - lx) + abs(ky - ly) <= KEY_APPROACH_LETHAL_MARGIN_CELLS
                for lx, ly in fixed_lethal_cells
            )
            if not near_lethal:
                safe.add((kx, ky))
        return safe if safe else key_cells

    def _objective_reward(
        self,
        state: ManicState,
        prev_state: ManicState,
        keys_collected: int,
        level_delta: int,
        first_key_achieved: bool,
    ) -> float:
        reward = 0.0
        if keys_collected > 0:
            reward += self.key_collect_reward * float(keys_collected)
        if level_delta > 0:
            reward += LEVEL_COMPLETE_REWARD * float(level_delta)
        if prev_state.keys_remaining > 0 and state.keys_remaining == 0:
            reward += ALL_KEYS_CLEARED_REWARD
        if prev_state.keys_remaining == 0 and state.keys_remaining == 0:
            prev_distance = self._distance_to_exit_px(prev_state)
            current_distance = self._distance_to_exit_px(state)
            if prev_distance is not None and current_distance is not None:
                distance_delta = prev_distance - current_distance
                if distance_delta > 0:
                    reward += EXIT_APPROACH_REWARD_PER_PX * distance_delta
        # Give the success bonus when the Nth key is collected (where N =
        # num_keys_mode).  Falls back to first-key behaviour when num_keys_mode=1.
        # With num_keys_mode=0 (no curriculum) the bonus still fires on the first
        # key as a soft incentive (decoupled from episode termination).
        _nkm = getattr(self, "num_keys_mode", 1 if getattr(self, "first_key_mode", False) else 0)
        _target = _nkm if _nkm > 0 else 1
        # Use monotone episode min so the bonus only fires once per distinct key.
        _ep_min = getattr(self, '_episode_min_keys_remaining', state.keys_remaining)
        _distinct_now = self.keys_at_reset - _ep_min
        _distinct_prev = _distinct_now - keys_collected
        if _distinct_prev < _target <= _distinct_now:
            reward += self.first_key_success_bonus
        return reward

    def _hazard_reward(self, lives_delta: int, air_delta: int) -> float:
        reward = 0.0
        if lives_delta < 0:
            reward += LIFE_LOSS_PENALTY * float(lives_delta)
        if air_delta < 0:
            reward += AIR_DECAY_PENALTY_COEF * float(air_delta)
        return reward

    def _pathing_reward(self, state: ManicState) -> float:
        covered = self._covered_cells(state.willy_x_px, state.willy_y_px)
        new_cells = covered - self.visited_cells
        if new_cells:
            self.visited_cells.update(new_cells)
        return self.pathing_new_cell_reward * float(len(new_cells))

    def _repeat_transition_penalty(
        self,
        prev_state: Optional[ManicState],
        state: ManicState,
        action_used: int,
        pathing_reward: float,
    ) -> Tuple[float, bool, int, str]:
        if prev_state is None:
            self._last_repeat_signature = None
            self._repeat_signature_count = 0
            return 0.0, False, 0, "-"

        prev_cell = (prev_state.willy_x_px // CELL_SIZE_PX, prev_state.willy_y_px // CELL_SIZE_PX)
        curr_cell = (state.willy_x_px // CELL_SIZE_PX, state.willy_y_px // CELL_SIZE_PX)
        no_outcome_change = (
            state.level == prev_state.level
            and state.lives == prev_state.lives
            and state.keys_remaining == prev_state.keys_remaining
            and state.score == prev_state.score
        )
        no_progress = (prev_cell == curr_cell) and no_outcome_change and (pathing_reward <= 0.0)
        if not no_progress:
            self._last_repeat_signature = None
            self._repeat_signature_count = 0
            return 0.0, False, 0, "-"

        signature = (
            prev_cell[0],
            prev_cell[1],
            int(action_used),
            curr_cell[0],
            curr_cell[1],
            state.level,
            state.lives,
            state.keys_remaining,
            state.score,
        )
        if signature == self._last_repeat_signature:
            self._repeat_signature_count += 1
        else:
            self._last_repeat_signature = signature
            self._repeat_signature_count = 1

        repeats_over_tolerance = max(0, self._repeat_signature_count - REPEAT_TRANSITION_TOLERANCE)
        if repeats_over_tolerance <= 0:
            return 0.0, True, self._repeat_signature_count, "repeat_transition_warmup"

        penalty = -min(
            REPEAT_TRANSITION_PENALTY * float(repeats_over_tolerance),
            REPEAT_TRANSITION_PENALTY_MAX,
        )
        reason = f"repeat_transition(count={self._repeat_signature_count},action={action_used})"
        return penalty, True, self._repeat_signature_count, reason

    def _walk_behavior_reward(
        self,
        prev_state: Optional[ManicState],
        state: ManicState,
        action_used: int,
        prev_action_used: Optional[int],
    ) -> Tuple[float, str]:
        if prev_state is None:
            return 0.0, "-"
        if action_used not in (1, 2):
            return 0.0, "-"

        x_delta = state.willy_x_px - prev_state.willy_x_px
        expected_sign = -1 if action_used == 1 else 1
        moved_in_expected_direction = x_delta * expected_sign > 0

        reward = 0.0
        reasons = []
        if moved_in_expected_direction:
            progress_px = abs(x_delta)
            reward += WALK_PROGRESS_REWARD_PER_PX * float(progress_px)
            reasons.append(f"walk_progress_px={progress_px}")
            if prev_action_used == action_used:
                reward += WALK_STREAK_BONUS
                reasons.append("walk_streak")
        else:
            reward -= WALK_NO_PROGRESS_PENALTY
            reasons.append("walk_no_progress")

        if prev_action_used in (1, 2) and prev_action_used != action_used:
            reward -= WALK_DIRECTION_FLIP_PENALTY
            reasons.append("walk_direction_flip")

        if not reasons:
            return reward, "-"
        return reward, ",".join(reasons)

    def _under_lethal_overlap_cells(
        self, state: ManicState, dynamic_lethal_cells: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        if not dynamic_lethal_cells:
            return set()
        x0 = max(0, min(SCREEN_CELLS_W - 1, state.willy_x_px // CELL_SIZE_PX))
        x1 = max(0, min(SCREEN_CELLS_W - 1, (state.willy_x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        y_top = max(0, state.willy_y_px // CELL_SIZE_PX)
        left_x = max(0, x0 - WALK_UNDER_LETHAL_SIDE_MARGIN_CELLS)
        right_x = min(SCREEN_CELLS_W - 1, x1 + WALK_UNDER_LETHAL_SIDE_MARGIN_CELLS)
        overlap: Set[Tuple[int, int]] = set()
        for x_cell in range(left_x, right_x + 1):
            for dy in range(0, WALK_UNDER_LETHAL_VERTICAL_LOOKAHEAD_CELLS + 1):
                y_cell = y_top - dy
                if y_cell < 0:
                    break
                cell = (x_cell, y_cell)
                if cell in dynamic_lethal_cells:
                    overlap.add(cell)
        return overlap

    def _is_under_lethal(self, state: ManicState, dynamic_lethal_cells: Set[Tuple[int, int]]) -> bool:
        return bool(self._under_lethal_overlap_cells(state, dynamic_lethal_cells))

    def _walk_under_lethal_reward(
        self,
        prev_state: Optional[ManicState],
        state: ManicState,
        action_used: int,
        dynamic_lethal_cells: Set[Tuple[int, int]],
    ) -> Tuple[float, str]:
        if prev_state is None:
            return 0.0, "-"
        if action_used not in (1, 2):
            return 0.0, "-"
        if not dynamic_lethal_cells:
            return 0.0, "-"

        x_delta = state.willy_x_px - prev_state.willy_x_px
        expected_sign = -1 if action_used == 1 else 1
        if x_delta * expected_sign <= 0:
            return 0.0, "-"

        overhead = self._under_lethal_overlap_cells(state, dynamic_lethal_cells)
        if not overhead:
            return 0.0, "-"

        newly_rewarded = 0
        for x_cell, y_cell in overhead:
            key = (state.level, x_cell, y_cell)
            if key in self._rewarded_under_lethal_cells:
                continue
            self._rewarded_under_lethal_cells.add(key)
            newly_rewarded += 1

        if newly_rewarded <= 0:
            return 0.0, "-"
        reward = WALK_UNDER_LETHAL_REWARD_PER_CELL * float(newly_rewarded)
        reason = f"walk_under_lethal_new_cells={newly_rewarded},overhead_cells={len(overhead)}"
        return reward, reason

    def _jump_under_lethal_penalty(
        self,
        state: ManicState,
        action_used: int,
        dynamic_lethal_cells: Set[Tuple[int, int]],
    ) -> Tuple[float, str]:
        if action_used not in (3, 4, 5):
            return 0.0, "-"
        overhead = self._under_lethal_overlap_cells(state, dynamic_lethal_cells)
        if not overhead:
            return 0.0, "-"
        return -JUMP_UNDER_LETHAL_PENALTY, f"jump_under_lethal_penalty,overhead_cells={len(overhead)}"

    def _compute_reward(
        self,
        state: ManicState,
        bridge_reward: int,
        first_key_achieved: bool,
        action_used: int,
        prev_action_used: Optional[int],
        dynamic_lethal_cells: Set[Tuple[int, int]],
    ) -> Tuple[float, float, float, float, float, float, float, bool, int, str, str, str]:
        if self.prev_state is None:
            objective = 0.0
            hazards = 0.0
        else:
            lives_delta = state.lives - self.prev_state.lives
            level_delta = state.level - self.prev_state.level
            air_delta = state.air - self.prev_state.air
            # Monotone floor: use the episode minimum keys_remaining as the
            # baseline so that keys respawning after a death never trigger a
            # second reward for the same key.
            _ep_min = getattr(self, '_episode_min_keys_remaining', self.prev_state.keys_remaining)
            _floor = min(self.prev_state.keys_remaining, _ep_min)
            keys_collected = max(0, _floor - state.keys_remaining)
            if keys_collected > 0:
                self._episode_min_keys_remaining = state.keys_remaining

            # 1) Objectives: positive rewards for game goals.
            objective = self._objective_reward(
                state,
                self.prev_state,
                keys_collected,
                level_delta,
                first_key_achieved,
            )

            # 2) Hazards: negative rewards for dangerous outcomes.
            hazards = self._hazard_reward(lives_delta, air_delta)

        # 3) Pathing: positive reward for unique area coverage.
        pathing = self._pathing_reward(state)

        repeat_penalty, repeat_detected, repeat_count, repeat_reason = self._repeat_transition_penalty(
            self.prev_state,
            state,
            action_used,
            pathing,
        )
        walk_reward, walk_reason = self._walk_behavior_reward(
            self.prev_state,
            state,
            action_used,
            prev_action_used,
        )
        under_lethal_reward, under_lethal_reason = self._walk_under_lethal_reward(
            self.prev_state,
            state,
            action_used,
            dynamic_lethal_cells,
        )
        jump_lethal_penalty, jump_lethal_reason = self._jump_under_lethal_penalty(
            state,
            action_used,
            dynamic_lethal_cells,
        )
        under_lethal_reward += jump_lethal_penalty
        if jump_lethal_reason != "-":
            under_lethal_reason = (
                jump_lethal_reason if under_lethal_reason == "-"
                else f"{under_lethal_reason},{jump_lethal_reason}"
            )

        reward = objective + hazards + pathing + repeat_penalty + walk_reward + under_lethal_reward
        if self.include_bridge_reward:
            reward += float(bridge_reward)

        state.coverage_ratio = self._coverage_ratio()
        return (
            reward,
            objective,
            hazards,
            pathing,
            repeat_penalty,
            walk_reward,
            under_lethal_reward,
            repeat_detected,
            repeat_count,
            repeat_reason,
            walk_reason,
            under_lethal_reason,
        )
