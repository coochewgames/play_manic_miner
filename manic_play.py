#!/usr/bin/env python3
"""Gameplay policy and reward shaping for Manic Miner.

Strategy
--------
* **Screen painting** — reward Willy for entering attribute cells not yet visited.
* **Objectives** — reward key collection and level completion; penalise life loss.
* **Safety blocking** — before executing any action, compute every attribute cell
  that action will enter (walk: the cell immediately ahead; jump: the full arc)
  and block the action if any of those cells currently hold a lethal attribute.
  Fixed nasties and moving guardians are both detected via the attribute buffer.
* **Guardian prediction** — for jump actions, moving guardian cells are dilated
  horizontally by GUARDIAN_PREDICT_CELLS in each direction.  This accounts for
  the guardian moving during the jump arc so the safety check reflects where the
  guardian *will be*, not just where it *is* at the moment the jump is chosen.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

from manic_data import CELL_SIZE_PX, SCREEN_CELLS_H, SCREEN_CELLS_W, WILLY_SIZE_PX, ManicState

# Rename exported name so manic_env.py can import it by the legacy alias.
PATHING_NEW_CELL_REWARD = 2.0   # reward per new 8×8 attribute cell entered
CELL_VISIT_REWARD = PATHING_NEW_CELL_REWARD

KEY_COLLECT_REWARD = 80.0       # reward per key collected
LEVEL_COMPLETE_REWARD = 1000.0  # reward on completing a level
# In first-key-mode the episode terminates on death, so missing the key reward
# (80) is already the implicit cost.  An explicit penalty on top tips the
# exploration-vs-safety balance against exploring and the agent learns to stand
# still.  Set to 0 for first-key-mode; increase for full-game training where
# the agent must learn to value its remaining lives.
LIFE_LOSS_PENALTY = 0.0

# Set to 0 to disable; non-zero values add a small per-step cost.
TIME_STEP_COST = 0.0

# Set to 0 to disable the airborne step penalty.
AIRBORNE_STEP_PENALTY = 0.0

# Dense reward for moving closer to the nearest uncollected key.
# Applied each grounded step where the Manhattan-pixel distance to the
# nearest key decreases and no key was collected that step.
KEY_APPROACH_REWARD_PER_PX = 0.2

# Frontier bonus: reward per unvisited orthogonal neighbour of a newly entered
# cell.  Creates a gradient toward unexplored areas — cells deep in unexplored
# territory have more unvisited neighbours and therefore attract more reward.
FRONTIER_REWARD_PER_NEIGHBOUR = 0.5

# Jump arc: cumulative Y rise (px, upward positive) at each game frame of the arc.
#
# Derived from the assembly at 8ABB.  The y-coordinate at 0x8068 stores twice
# the pixel y-position (1 unit = 0.5 px).  Each frame n (0–17) applies:
#   delta_units = (n & 0xFE) - 8
#   delta_px    = delta_units / 2   (negative = upward)
# Cumulative rise per frame: 4,8,11,14,16,18,19,20 (peak),20,20,19,18,16,14,11,8,4,0
# Peak = 20 px (2.5 attribute cells).  Confirmed by the disassembly annotations:
#   frame 13 → 16 px above start (2 cell-heights) ✓
#   frame 16 →  8 px above start (1 cell-height)  ✓
#   frame 17 →  0 px            (landed)           ✓
#
# Horizontal movement: for directional jumps (4/5) Willy continues walking at
# his normal rate throughout the arc — the direction key suppression in $806A
# prevents *new* input, but existing lateral movement persists.  A value of 2
# (px per game frame) matches Willy's walk speed and accurately sweeps the
# cells he passes through.  Standing jump (action 3) uses h_sign=0 so this
# constant has no effect on that arc.
JUMP_PHASE_HORIZONTAL_PX = 2
JUMP_ARC_Y_FROM_START_PX = (4, 8, 11, 14, 16, 18, 19, 20, 20, 20, 19, 18, 16, 14, 11, 8, 4, 0)

# Minimum lethal-cell overlap count to block an action.
WALK_LETHAL_BLOCK_MIN = 1
JUMP_LETHAL_BLOCK_MIN = 1

# Number of cells to expand moving guardian cells horizontally when checking
# directional jump safety (actions 4/5).  Dilation is asymmetric: extended in
# the guardian's current travel direction, with a 1-cell margin the other way.
# A jump takes ~18 game frames; at ~2 px/frame a guardian moves ~4 attribute
# cells.  4 gives a comfortable safety margin.
GUARDIAN_PREDICT_CELLS = 4

# For a standing jump (action 3) Willy's x-position is fixed for the full arc,
# so the guardian can approach from *either* side and direction may reverse on a
# boundary bounce.  Use a larger symmetric window.
GUARDIAN_STANDING_JUMP_PREDICT_CELLS = 5


class ManicPlayMixin:
    """Action selection (safety blocking) and reward computation."""

    # ------------------------------------------------------------------ #
    # Jump arc geometry                                                    #
    # ------------------------------------------------------------------ #

    def _jump_arc_cells(self, state: ManicState, h_sign: int) -> Set[Tuple[int, int]]:
        """All attribute cells Willy's sprite will occupy during a jump arc.

        h_sign: -1 = jump-left, 0 = stand jump, +1 = jump-right.
        """
        cells: Set[Tuple[int, int]] = set()
        for phase, rise_px in enumerate(JUMP_ARC_Y_FROM_START_PX, start=1):
            x = max(0, min(255, state.willy_x_px + h_sign * phase * JUMP_PHASE_HORIZONTAL_PX))
            y = max(0, min(191, state.willy_y_px - rise_px))
            cells.update(self._covered_cells(x, y))
        return cells

    # ------------------------------------------------------------------ #
    # Safety blocking                                                      #
    # ------------------------------------------------------------------ #

    def _dilate_cells_h(
        self,
        cells: Set[Tuple[int, int]],
        n_left: int,
        n_right: int,
    ) -> Set[Tuple[int, int]]:
        """Expand *cells* by *n_left* columns leftward and *n_right* rightward."""
        dilated: Set[Tuple[int, int]] = set()
        for cx, cy in cells:
            for dx in range(-n_left, n_right + 1):
                nx = cx + dx
                if 0 <= nx < SCREEN_CELLS_W:
                    dilated.add((nx, cy))
        return dilated

    def _cells_above_willy(self, state: ManicState, rows: int = 2) -> Set[Tuple[int, int]]:
        """Attribute cells in the N rows directly above Willy's current position.

        Checked explicitly before the arc for all jump actions, as a
        belt-and-suspenders guard against arc sampling gaps.
        """
        x0 = max(0, state.willy_x_px // CELL_SIZE_PX)
        x1 = min(SCREEN_CELLS_W - 1, (state.willy_x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX)
        y_top = state.willy_y_px // CELL_SIZE_PX
        cells: Set[Tuple[int, int]] = set()
        for dy in range(1, rows + 1):
            y = y_top - dy
            if y < 0:
                break
            for x in range(x0, x1 + 1):
                cells.add((x, y))
        return cells

    def _filter_ceiling_protected(
        self,
        fixed_lethal: Set[Tuple[int, int]],
        attr_buffer: bytes,
        all_lethal_attrs: Set[int],
        willy_bottom_cell: int,
    ) -> Set[Tuple[int, int]]:
        """Remove fixed lethal cells that are on a platform ceiling above Willy.

        A nasty at (cx, cy) is ceiling-protected only when BOTH conditions hold:
          1. The cell at (cx, cy+1) is a solid non-lethal platform tile, AND
          2. Willy's bottom cell is strictly below that platform
             (willy_bottom_cell > cy+1), so the platform physically lies between
             Willy and the nasty and the jump arc is stopped before reaching it.

        If Willy is already at or above the platform (e.g. standing on it), the
        nasty is directly reachable and must not be filtered out.
        """
        accessible: Set[Tuple[int, int]] = set()
        for cx, cy in fixed_lethal:
            below_y = cy + 1
            if below_y >= SCREEN_CELLS_H:
                accessible.add((cx, cy))
                continue
            idx = below_y * SCREEN_CELLS_W + cx
            if idx >= len(attr_buffer):
                accessible.add((cx, cy))
                continue
            below_attr = int(attr_buffer[idx]) & 0x7F
            if (below_attr != 0
                    and below_attr not in all_lethal_attrs
                    and willy_bottom_cell > below_y):
                continue  # ceiling-protected — platform lies between Willy and nasty
            accessible.add((cx, cy))
        return accessible

    def _action_is_safe(
        self,
        state: ManicState,
        action: int,
        lethal_cells: Set[Tuple[int, int]],
    ) -> bool:
        """Return True if *action* will not move Willy into a lethal cell."""
        if not lethal_cells:
            return True
        if action == 1:
            front = self._front_cells_for_direction(state, -1)
            return len(front & lethal_cells) < WALK_LETHAL_BLOCK_MIN
        if action == 2:
            front = self._front_cells_for_direction(state, 1)
            return len(front & lethal_cells) < WALK_LETHAL_BLOCK_MIN
        if action in (3, 4, 5):
            h_sign = 0 if action == 3 else (-1 if action == 4 else 1)
            return len(self._jump_arc_cells(state, h_sign) & lethal_cells) < JUMP_LETHAL_BLOCK_MIN
        return True  # no-op (0) is always safe

    def _guardian_sweep_cells(self, level: int) -> Set[Tuple[int, int]]:
        """All attribute cells within each horizontal guardian's full patrol range.

        Uses the guardian's left/right movement bounds from cavern ROM data —
        exact, level-specific, and independent of the guardian's current position.
        """
        cells: Set[Tuple[int, int]] = set()
        for g_y, g_x_left, g_x_right in self._h_guardian_bounds_for_level(level):
            for x in range(g_x_left, g_x_right + 1):
                cells.add((x, g_y))
        return cells

    def _lethal_cells_for_action(
        self,
        action: int,
        fixed_lethal: Set[Tuple[int, int]],
        moving_lethal: Set[Tuple[int, int]],
        level: Optional[int] = None,
    ) -> Set[Tuple[int, int]]:
        """Return the lethal cell set appropriate for *action*.

        Standing jump (3): symmetric dilation with GUARDIAN_STANDING_JUMP_PREDICT_CELLS.
          Willy's x is fixed for the full arc, the guardian can approach from
          either side, and may reverse on a boundary bounce — so direction is
          irrelevant and a larger window is needed.

        Directional jumps (4/5): asymmetric dilation with GUARDIAN_PREDICT_CELLS,
          extended in the guardian's current travel direction with a 1-cell margin
          the other way to catch boundary bounces.

        Walk/no-op: actual current guardian positions only.
        """
        if action == 3 and moving_lethal:
            n = GUARDIAN_STANDING_JUMP_PREDICT_CELLS
            dilated = self._dilate_cells_h(moving_lethal, n, n)
            return fixed_lethal | dilated

        if action in (4, 5) and moving_lethal:
            directions = self._read_h_guardian_directions()
            any_right = any(d == "right" for d in directions)
            any_left  = any(d == "left"  for d in directions)
            if not directions:
                n_left = n_right = GUARDIAN_PREDICT_CELLS
            else:
                n_right = GUARDIAN_PREDICT_CELLS if any_right else 1
                n_left  = GUARDIAN_PREDICT_CELLS if any_left  else 1
            dilated = self._dilate_cells_h(moving_lethal, n_left, n_right)
            return fixed_lethal | dilated

        return fixed_lethal | moving_lethal

    def _safe_fallback(
        self,
        state: ManicState,
        blocked: int,
        fixed_lethal: Set[Tuple[int, int]],
        moving_lethal: Set[Tuple[int, int]],
    ) -> int:
        """Find the best safe fallback when *blocked* action cannot be taken."""
        if blocked == 4:
            candidates = [1, 0]       # jump-left blocked → try walk-left, then no-op
        elif blocked == 5:
            candidates = [2, 0]       # jump-right blocked → try walk-right, then no-op
        elif blocked == 3:
            candidates = [0]          # stand-jump blocked → no-op
        elif blocked == 1:
            candidates = [0, 2]       # walk-left blocked → try no-op, walk-right
        elif blocked == 2:
            candidates = [0, 1]       # walk-right blocked → try no-op, walk-left
        else:
            candidates = [0]
        for candidate in candidates:
            lethal = self._lethal_cells_for_action(candidate, fixed_lethal, moving_lethal, state.level)
            if self._action_is_safe(state, candidate, lethal):
                return candidate
        return 0

    def _select_safe_action(
        self,
        state: ManicState,
        action: int,
        attr_buffer: Optional[bytes] = None,
    ) -> Tuple[int, bool, str]:
        """Apply attribute-based safety blocking for walk actions only.

        Jump actions (3/4/5) are never blocked — the agent learns from those
        outcomes directly.  Only walk-left (1) and walk-right (2) are blocked
        to prevent trivially-avoidable ground-level nasty collisions.

        Returns (action_used, was_blocked, reason_string).
        """
        if not self.safety_shield:
            self._last_dynamic_lethal_cells_count = 0
            return action, False, ""

        # Jumps are not blocked regardless of safety_shield setting.
        if action not in (1, 2):
            self._last_dynamic_lethal_cells_count = 0
            return action, False, ""

        if attr_buffer is None:
            attr_buffer = self._read_attr_buffer()

        moving_attrs, fixed_attrs = self._configured_lethal_attr_groups_for_level(state.level)
        moving_lethal = self._dynamic_cells_for_attrs(attr_buffer, moving_attrs)
        fixed_lethal_raw = self._dynamic_cells_for_attrs(attr_buffer, fixed_attrs)
        self._last_dynamic_lethal_cells_count = len(fixed_lethal_raw | moving_lethal)

        action_lethal = fixed_lethal_raw | moving_lethal
        if self._action_is_safe(state, action, action_lethal):
            return action, False, ""

        fallback = 0  # blocked walk → no-op; let the agent try a jump next step
        return fallback, True, f"safety_block(action={action},fallback={fallback})"

    # ------------------------------------------------------------------ #
    # Reward                                                               #
    # ------------------------------------------------------------------ #

    def _frontier_bonus(self, new_cells: Set[Tuple[int, int]]) -> float:
        """Reward per unvisited orthogonal neighbour of each newly entered cell.

        Called after visited_cells has been updated, so neighbours still absent
        from visited_cells are genuinely unexplored.
        """
        bonus = 0.0
        for cx, cy in new_cells:
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if 0 <= nx < SCREEN_CELLS_W and 0 <= ny < SCREEN_CELLS_H:
                    if (nx, ny) not in self.visited_cells:
                        bonus += FRONTIER_REWARD_PER_NEIGHBOUR
        return bonus

    def _compute_reward(
        self,
        state: ManicState,
        prev_state: Optional[ManicState],
        action_used: int,
        is_airborne: bool = False,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Compute step reward.

        Returns (total, painting, frontier, key_reward, level_reward, life_penalty, key_approach).
        life_penalty is a positive magnitude; it is *subtracted* from total.

        Painting and frontier are suppressed while Willy is airborne so that
        cells along the jump arc are not counted — only cells visited on the
        ground count.  This removes the large exploration incentive for jumping.

        key_approach is a dense reward for reducing the Manhattan-pixel distance
        to the nearest uncollected key while grounded and no key was collected.
        """
        state.coverage_ratio = self._coverage_ratio()
        if is_airborne:
            painting = 0.0
            frontier = 0.0
            airborne_penalty = AIRBORNE_STEP_PENALTY
        else:
            airborne_penalty = 0.0
            covered = self._covered_cells(state.willy_x_px, state.willy_y_px)
            new_cells = covered - self.visited_cells
            self.visited_cells.update(new_cells)
            state.coverage_ratio = self._coverage_ratio()
            painting = self.pathing_new_cell_reward * float(len(new_cells))
            frontier = self._frontier_bonus(new_cells)

        step_cost = TIME_STEP_COST

        if prev_state is None:
            return painting + frontier - airborne_penalty - step_cost, painting, frontier, 0.0, 0.0, 0.0, 0.0

        keys_collected = max(0, prev_state.keys_remaining - state.keys_remaining)
        key_reward = KEY_COLLECT_REWARD * float(keys_collected)

        level_delta = max(0, state.level - prev_state.level)
        level_reward = LEVEL_COMPLETE_REWARD * float(level_delta)

        lives_delta = min(0, state.lives - prev_state.lives)
        life_penalty = LIFE_LOSS_PENALTY * float(-lives_delta)

        # Dense approach reward: reward each pixel of progress toward the
        # nearest key while grounded.  Only when keys_remaining is unchanged
        # (no collection this step — key_reward already covers that case).
        key_approach = 0.0
        if (not is_airborne
                and keys_collected == 0
                and state.keys_remaining > 0
                and state.keys_remaining == prev_state.keys_remaining):
            prev_dist = abs(prev_state.nearest_key_dx_px) + abs(prev_state.nearest_key_dy_px)
            curr_dist = abs(state.nearest_key_dx_px) + abs(state.nearest_key_dy_px)
            key_approach = KEY_APPROACH_REWARD_PER_PX * max(0.0, float(prev_dist - curr_dist))

        total = painting + frontier + key_reward + level_reward + key_approach - life_penalty - airborne_penalty - step_cost
        return total, painting, frontier, key_reward, level_reward, life_penalty, key_approach
