#!/usr/bin/env python3
"""Stable Manic Miner memory map and data interpretation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Manic Miner memory locations
AIR_HI_ADDR = 0x80BC
LEVEL_ADDR = 0x8407
LIVES_ADDR = 0x8457
SCORE_ADDR = 0x8429
SCORE_LEN = 6
WILLY_Y_ADDR = 0x8068
WILLY_X_FINE_ADDR = 0x8069
WILLY_FLAGS_ADDR = 0x806A
AIR_SUPPLY_ADDR = 0x80BC   # LSB of display address for rightmost air-bar cell
AIR_SUPPLY_MAX  = 0x3F    # full air bar (range 0x24–0x3F; 0x24 = empty → life lost)
WILLY_AIRBORNE_ADDR = 0x806B
WILLY_ATTR_PTR_ADDR = 0x806C
ATTR_BUFFER_ADDR = 0x5C00
ATTR_BUFFER_LEN = 0x0200
ROOM_DATA_BASE_ADDR = 0xB000
ROOM_DATA_SIZE = 0x0400
PORTAL_POS_OFFSET = 0x02B0
ITEMS_OFFSET = 0x0275
ITEMS_COUNT = 5
ITEM_STRIDE = 5
ITEM_INK_CYCLE = (0x03, 0x06, 0x05, 0x04)  # magenta, yellow, cyan, green
NASTY1_OFFSET = 0x024D
NASTY2_OFFSET = 0x0256
NASTY_STRIDE = 9        # bytes per nasty type definition (1 attr + 8 graphic bytes)
NASTY_COUNT = 2         # number of nasty type slots per level
CAVERN_LAYOUT_SIZE = 512  # bytes of attribute layout at start of room data (32×16 cells)
H_GUARD_OFFSET = 0x02BE
V_GUARD_OFFSET = 0x02DD
GUARD_MAX_COUNT = 4
GUARD_STRIDE = 7

# Runtime horizontal guardian table — maintained by the game engine at a fixed
# address regardless of level.  The game copies room guardian data here at level
# start and updates it each frame.
# Offset +04 within each 7-byte entry is the animation frame:
#   0–3 = moving right,  4–7 = moving left.
H_GUARD_RUNTIME_BASE = 0x80BE
H_GUARD_RUNTIME_FRAME_OFFSET = 4   # byte within each entry that encodes direction
H_GUARD_RUNTIME_DIR_LEFT_MIN = 4   # frame >= this value means moving left

# Screen/pathing geometry
CELL_SIZE_PX = 8
WILLY_SIZE_PX = 16
SCREEN_CELLS_W = 32
SCREEN_CELLS_H = 24
PATHING_TOTAL_CELLS = SCREEN_CELLS_W * SCREEN_CELLS_H

# Observation layout:
# [air, level, lives, score, willy_x_px, willy_y_px,
#  keys_remaining, portal_open, willy_flags, coverage_ratio,
#  h_guard_x_cell, nearest_key_dx_px, nearest_key_dy_px]
# h_guard_x_cell: x-cell (0–31) of the first active horizontal guardian.
# nearest_key_dx/dy: signed pixel offset from Willy to nearest uncollected key
#   (positive dx = key is to the right; positive dy = key is below).
#   Set to 0,0 when all keys are collected (portal open).
OBS_RAW_MIN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, -255, -191], dtype=np.float32)
OBS_RAW_MAX = np.array(
    [0x3FFF, 255, 255, 999999, 255, 191, 255, 1, 255, 1, 31,  255,  191],
    dtype=np.float32,
)


@dataclass
class ManicState:
    air: int
    level: int
    lives: int
    score: int
    willy_x_px: int
    willy_y_px: int
    willy_flags: int
    keys_remaining: int
    portal_open: int
    coverage_ratio: float
    h_guard_x_cell: int = 0    # x-cell of first active horizontal guardian, 0 if absent
    nearest_key_dx_px: int = 0  # signed px offset to nearest uncollected key (+ = right)
    nearest_key_dy_px: int = 0  # signed px offset to nearest uncollected key (+ = down)

    def to_observation(self) -> np.ndarray:
        return np.array(
            [
                self.air,
                self.level,
                self.lives,
                self.score,
                self.willy_x_px,
                self.willy_y_px,
                self.keys_remaining,
                self.portal_open,
                self.willy_flags,
                self.coverage_ratio,
                self.h_guard_x_cell,
                self.nearest_key_dx_px,
                self.nearest_key_dy_px,
            ],
            dtype=np.float32,
        )


def parse_score_digits(raw: bytes) -> int:
    """Parse ASCII score bytes to an integer."""
    chars = []
    for index, byte in enumerate(raw):
        if 0x30 <= byte <= 0x39:
            chars.append(chr(byte))
        else:
            logger.error(
                "Non-ASCII score byte at index %s: 0x%02X (raw=%s); returning 0",
                index,
                byte,
                raw.hex(),
            )
            return 0
    return int("".join(chars))


class ManicDataMixin:
    """Methods that decode and interpret raw game memory/state."""

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        normalized = (observation - OBS_RAW_MIN) / (OBS_RAW_MAX - OBS_RAW_MIN)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _decode_attr_addr(self, address: int) -> Optional[Tuple[int, int]]:
        if not (ATTR_BUFFER_ADDR <= address < ATTR_BUFFER_ADDR + ATTR_BUFFER_LEN):
            return None
        offset = address - ATTR_BUFFER_ADDR
        return offset % SCREEN_CELLS_W, offset // SCREEN_CELLS_W

    def _read_willy_xy(self) -> Tuple[int, int]:
        attr_ptr = self._read_u16_le(WILLY_ATTR_PTR_ADDR)
        decoded = self._decode_attr_addr(attr_ptr)
        # 0x8069 stores Willy's X animation frame (0..3), each frame is 2 px.
        fine_x = (self._read_u8(WILLY_X_FINE_ADDR) & 0x03) * 2
        if decoded is None:
            x_px = fine_x
        else:
            # Attr pointer gives Willy's top-left cell; 0x8069 adds 0/2/4/6 px.
            x_px = (decoded[0] * CELL_SIZE_PX) + fine_x
        # 0x8068 holds twice Willy's pixel Y in practice.
        y_px = self._read_u8(WILLY_Y_ADDR) // 2
        y_px = max(0, min(191, y_px))
        x_px = max(0, min(255, x_px))
        return x_px, y_px

    def _is_airborne_status_active(self, status: int) -> bool:
        # 0x00 = grounded. 0x01..0xFE = jumping/falling/in-transition.
        return 0x01 <= status <= 0xFE

    def _covered_cells(self, x_px: int, y_px: int) -> Set[Tuple[int, int]]:
        x0 = max(0, min(SCREEN_CELLS_W - 1, x_px // CELL_SIZE_PX))
        y0 = max(0, min(SCREEN_CELLS_H - 1, y_px // CELL_SIZE_PX))
        x1 = max(0, min(SCREEN_CELLS_W - 1, (x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        y1 = max(0, min(SCREEN_CELLS_H - 1, (y_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}

    def _coverage_ratio(self) -> float:
        return float(len(self.visited_cells)) / float(PATHING_TOTAL_CELLS)

    def _read_portal_xy_for_level(self, level: int) -> Optional[Tuple[int, int]]:
        if level < 0:
            return None
        room_base = ROOM_DATA_BASE_ADDR + (ROOM_DATA_SIZE * level)
        if room_base + PORTAL_POS_OFFSET + 1 > 0xFFFF:
            return None
        packed_pos = self._read_u16_le(room_base + PORTAL_POS_OFFSET)
        x_px = (packed_pos & 0x1F) * CELL_SIZE_PX
        y_px = ((packed_pos >> 5) & 0x0F) * CELL_SIZE_PX
        return x_px, y_px

    def _iter_item_slots_for_level(self, level: int) -> list[Tuple[int, Tuple[int, int]]]:
        if level < 0:
            return []
        room = self._read_room_bytes(level)
        if not room:
            return []

        slots: list[Tuple[int, Tuple[int, int]]] = []
        for slot in range(ITEMS_COUNT):
            base = ITEMS_OFFSET + (slot * ITEM_STRIDE)
            if base + ITEM_STRIDE > len(room):
                break
            item_attr = int(room[base])
            if item_attr == 0xFF:
                break
            if item_attr == 0x00:
                continue
            decoded = self._decode_item_cell(int(room[base + 1]), int(room[base + 2]), int(room[base + 3]))
            if decoded is None:
                continue
            slots.append((item_attr, decoded))
        return slots

    def _count_configured_items_for_level(self, level: int) -> int:
        return len(self._iter_item_slots_for_level(level))

    def _item_attr_candidates(self, item_attr: int) -> Set[int]:
        # Keep paper/bright/flash metadata and allow all item ink-cycle phases.
        base = int(item_attr) & 0xF8
        candidates = {(base | ink) & 0xFF for ink in ITEM_INK_CYCLE}
        candidates.add(int(item_attr) & 0xFF)
        return candidates

    def _is_item_visible_at_cell(self, attr_buffer: bytes, item_attr: int, cell: Tuple[int, int]) -> bool:
        x_cell, y_cell = cell
        if x_cell < 0 or x_cell >= SCREEN_CELLS_W:
            return False
        if y_cell < 0 or y_cell >= SCREEN_CELLS_H:
            return False
        idx = (y_cell * SCREEN_CELLS_W) + x_cell
        if idx < 0 or idx >= len(attr_buffer):
            return False
        live_attr = int(attr_buffer[idx]) & 0x7F
        for candidate in self._item_attr_candidates(item_attr):
            if live_attr == (candidate & 0x7F):
                return True
        return False

    def _count_items_remaining_for_level(self, level: int, attr_buffer: Optional[bytes] = None) -> int:
        if level < 0:
            return 0
        if attr_buffer is None:
            attr_buffer = self._read_attr_buffer()
        count = 0
        for item_attr, cell in self._iter_item_slots_for_level(level):
            if self._is_item_visible_at_cell(attr_buffer, item_attr, cell):
                count += 1
        return count

    def _decode_item_cell(self, pos0: int, pos1: int, pos2: int) -> Optional[Tuple[int, int]]:
        # Item layout byte format:
        # pos0 = YYYXXXXX, pos1 bit0 = Y(msb duplicate), pos2 bit4 = Y(msb duplicate)
        x_cell = pos0 & 0x1F
        y_cell = ((pos0 >> 5) & 0x07) | ((pos1 & 0x01) << 3)
        y_cell_dup = ((pos2 >> 4) & 0x01) << 3
        if y_cell_dup and not (y_cell & 0x08):
            y_cell |= 0x08
        if x_cell < 0 or x_cell >= SCREEN_CELLS_W:
            return None
        if y_cell < 0 or y_cell >= SCREEN_CELLS_H:
            return None
        return x_cell, y_cell

    def _active_item_cells_for_level(
        self, level: int, attr_buffer: Optional[bytes] = None
    ) -> Set[Tuple[int, int]]:
        if level < 0:
            return set()
        if attr_buffer is None:
            attr_buffer = self._read_attr_buffer()
        cells: Set[Tuple[int, int]] = set()
        for item_attr, cell in self._iter_item_slots_for_level(level):
            if self._is_item_visible_at_cell(attr_buffer, item_attr, cell):
                cells.add(cell)
        return cells

    def _read_attr_buffer(self) -> bytes:
        return self.client.read_bytes(ATTR_BUFFER_ADDR, ATTR_BUFFER_LEN)

    def _parse_h_guardian_bounds(self, room: bytes) -> list:
        """Parse horizontal guardian patrol bounds from cavern room data.

        Each active guardian entry (7 bytes at H_GUARD_OFFSET) encodes:
          byte 0   : attribute + speed flag (0xFF = end, 0x00 = unused slot)
          bytes 1-2: 16-bit attr-buffer address of initial position (LE)
          bytes 5-6: LSBs of left/right boundary attr-buffer addresses

        Returns a list of (y_cell, x_left_cell, x_right_cell) tuples.
        """
        result = []
        for idx in range(GUARD_MAX_COUNT):
            s = H_GUARD_OFFSET + idx * GUARD_STRIDE
            if s + GUARD_STRIDE > len(room):
                break
            attr = int(room[s])
            if attr == 0xFF:
                break
            if attr == 0x00:
                continue
            pos_lsb = int(room[s + 1])
            pos_msb = int(room[s + 2])
            addr = (pos_msb << 8) | pos_lsb
            if addr < ATTR_BUFFER_ADDR:
                continue
            offset = addr - ATTR_BUFFER_ADDR
            y_cell = offset // SCREEN_CELLS_W
            # Boundary bytes are just the LSB of the attr-buffer address;
            # low 5 bits encode x_cell (columns 0-31).
            x_left = int(room[s + 5]) & 0x1F
            x_right = int(room[s + 6]) & 0x1F
            result.append((y_cell, x_left, x_right))
        return result

    def _h_guardian_bounds_for_level(self, level: int) -> list:
        """Return cached horizontal guardian patrol bounds for *level*.

        Result is a list of (y_cell, x_left_cell, x_right_cell) tuples.
        Cached because guardian definitions live in ROM and never change.
        """
        cached = self._h_guardian_bounds_by_level.get(level)
        if cached is not None:
            return cached
        room = self._read_room_bytes(level)
        bounds = self._parse_h_guardian_bounds(room) if room else []
        self._h_guardian_bounds_by_level[level] = bounds
        return bounds

    def _read_h_guardian_x_cell(self) -> int:
        """Return the x-cell of the first active horizontal guardian from the
        runtime table, or 0 if no guardian is active.

        The runtime table at H_GUARD_RUNTIME_BASE is updated every game frame.
        Byte 1 of each 7-byte entry is the LSB of the current attribute-buffer
        address; its low 5 bits give the guardian's current x-cell column.
        """
        total_bytes = GUARD_MAX_COUNT * GUARD_STRIDE
        try:
            data = self.client.read_bytes(H_GUARD_RUNTIME_BASE, total_bytes)
        except Exception:
            return 0
        for idx in range(GUARD_MAX_COUNT):
            slot = idx * GUARD_STRIDE
            if slot >= len(data):
                break
            attr = int(data[slot])
            if attr == 0xFF:
                break
            if attr == 0x00:
                continue
            pos_lsb = int(data[slot + 1])
            return pos_lsb & 0x1F
        return 0

    def _read_h_guardian_directions(self) -> list:
        """Return a list of direction strings for each active horizontal guardian.

        Reads the runtime guardian table at H_GUARD_RUNTIME_BASE.  Each active
        entry's frame byte (offset +4) encodes direction: 0–3 = right, 4–7 = left.
        Returns a list of 'left' or 'right' strings, one per active guardian.
        Entries terminated by 0xFF or padded with 0x00 are skipped.
        """
        total_bytes = GUARD_MAX_COUNT * GUARD_STRIDE
        try:
            data = self.client.read_bytes(H_GUARD_RUNTIME_BASE, total_bytes)
        except Exception:
            return []
        directions = []
        for idx in range(GUARD_MAX_COUNT):
            slot = idx * GUARD_STRIDE
            if slot >= len(data):
                break
            control = int(data[slot])
            if control == 0xFF:
                break
            if control == 0x00:
                continue
            frame = int(data[slot + H_GUARD_RUNTIME_FRAME_OFFSET])
            directions.append("left" if frame >= H_GUARD_RUNTIME_DIR_LEFT_MIN else "right")
        return directions

    def _read_room_bytes(self, level: int) -> bytes:
        if level < 0:
            return b""
        room_base = ROOM_DATA_BASE_ADDR + (ROOM_DATA_SIZE * level)
        if room_base + ROOM_DATA_SIZE - 1 > 0xFFFF:
            return b""
        return self.client.read_bytes(room_base, ROOM_DATA_SIZE)

    def _parse_guardian_attrs(self, room: bytes, offset: int) -> Set[int]:
        attrs: Set[int] = set()
        for idx in range(GUARD_MAX_COUNT):
            slot = offset + (idx * GUARD_STRIDE)
            if slot >= len(room):
                break
            raw_attr = int(room[slot])
            if raw_attr == 0xFF:
                break
            if raw_attr == 0x00:
                continue
            # Bit 7 is speed/flash metadata; strip it for attr-map matching.
            attrs.add(raw_attr & 0x7F)
        return attrs

    def _parse_nasty_attrs(self, room: bytes) -> Set[int]:
        """Return the set of nasty attribute bytes that are actually placed in the cavern.

        Each nasty type definition is 9 bytes (1 attr + 8 graphic bytes) at NASTY1_OFFSET
        and NASTY2_OFFSET.  A type is only included as lethal if its attribute byte actually
        appears in the cavern layout (first CAVERN_LAYOUT_SIZE bytes of room data) — this
        prevents unused nasty type slots from adding false-positive lethal attributes.
        """
        layout = room[:CAVERN_LAYOUT_SIZE]
        attrs: Set[int] = set()
        for offset in (NASTY1_OFFSET, NASTY2_OFFSET):
            if offset >= len(room):
                continue
            raw_attr = int(room[offset])
            if raw_attr in (0x00, 0xFF):
                continue
            masked = raw_attr & 0x7F
            if masked in layout:
                attrs.add(masked)
        return attrs

    def _configured_lethal_attr_groups_for_level(self, level: int) -> Tuple[Set[int], Set[int]]:
        cached = self.configured_lethal_attr_groups_by_level.get(level)
        if cached is not None:
            return cached
        room = self._read_room_bytes(level)
        if not room:
            empty = (set(), set())
            self.configured_lethal_attr_groups_by_level[level] = empty
            return empty
        moving_attrs = set()
        moving_attrs.update(self._parse_guardian_attrs(room, H_GUARD_OFFSET))
        moving_attrs.update(self._parse_guardian_attrs(room, V_GUARD_OFFSET))
        fixed_attrs = self._parse_nasty_attrs(room)
        self.configured_lethal_attr_groups_by_level[level] = (moving_attrs, fixed_attrs)

        # Diagnostic: log raw bytes at each lethal-attr source so detection
        # gaps can be spotted against what is visually on screen.
        layout = room[:CAVERN_LAYOUT_SIZE]
        nasty_raws = []
        for offset in (NASTY1_OFFSET, NASTY2_OFFSET):
            if offset < len(room):
                raw = room[offset]
                present = (raw & 0x7F) in layout if raw not in (0x00, 0xFF) else False
                nasty_raws.append(f"0x{raw:02X}({'used' if present else 'unused'})")
        h_guard_raws = [
            room[H_GUARD_OFFSET + i * GUARD_STRIDE]
            for i in range(GUARD_MAX_COUNT)
            if H_GUARD_OFFSET + i * GUARD_STRIDE < len(room)
        ]
        v_guard_raws = [
            room[V_GUARD_OFFSET + i * GUARD_STRIDE]
            for i in range(GUARD_MAX_COUNT)
            if V_GUARD_OFFSET + i * GUARD_STRIDE < len(room)
        ]
        logger.debug(
            "Level %d lethal attrs — nasties=%s "
            "h_guards=%s v_guards=%s → fixed=%s moving=%s",
            level,
            nasty_raws,
            [f"0x{b:02X}" for b in h_guard_raws],
            [f"0x{b:02X}" for b in v_guard_raws],
            sorted(f"0x{a:02X}" for a in fixed_attrs),
            sorted(f"0x{a:02X}" for a in moving_attrs),
        )
        return moving_attrs, fixed_attrs

    def _configured_lethal_attrs_for_level(self, level: int) -> Set[int]:
        cached = self.configured_lethal_attrs_by_level.get(level)
        if cached is not None:
            return cached
        moving_attrs, fixed_attrs = self._configured_lethal_attr_groups_for_level(level)
        attrs = set(moving_attrs)
        attrs.update(fixed_attrs)
        self.configured_lethal_attrs_by_level[level] = attrs
        return attrs

    def _dynamic_cells_for_attrs(self, attr_buffer: bytes, attrs: Set[int]) -> Set[Tuple[int, int]]:
        if not attrs:
            return set()
        cells: Set[Tuple[int, int]] = set()
        rows_in_buffer = len(attr_buffer) // SCREEN_CELLS_W
        for y_cell in range(rows_in_buffer):
            row_base = y_cell * SCREEN_CELLS_W
            for x_cell in range(SCREEN_CELLS_W):
                if (int(attr_buffer[row_base + x_cell]) & 0x7F) in attrs:
                    cells.add((x_cell, y_cell))
        return cells

    def _dynamic_lethal_cells_for_level(self, level: int, attr_buffer: bytes) -> Set[Tuple[int, int]]:
        lethal_attrs = self._configured_lethal_attrs_for_level(level)
        return self._dynamic_cells_for_attrs(attr_buffer, lethal_attrs)

    def _front_cells_for_direction(self, state: ManicState, direction: int) -> Set[Tuple[int, int]]:
        x0 = max(0, min(SCREEN_CELLS_W - 1, state.willy_x_px // CELL_SIZE_PX))
        y0 = max(0, min(SCREEN_CELLS_H - 1, state.willy_y_px // CELL_SIZE_PX))
        x1 = max(0, min(SCREEN_CELLS_W - 1, (state.willy_x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        y1 = max(0, min(SCREEN_CELLS_H - 1, (state.willy_y_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        front_x = x1 + 1 if direction > 0 else x0 - 1
        if front_x < 0 or front_x >= SCREEN_CELLS_W:
            return set()
        return {(front_x, y) for y in range(y0, y1 + 1)}

    def _distance_to_exit_px(self, state: ManicState) -> Optional[float]:
        portal_xy = self._read_portal_xy_for_level(state.level)
        if portal_xy is None:
            return None
        willy_center_x = state.willy_x_px + (WILLY_SIZE_PX // 2)
        willy_center_y = state.willy_y_px + (WILLY_SIZE_PX // 2)
        portal_center_x = portal_xy[0] + (WILLY_SIZE_PX // 2)
        portal_center_y = portal_xy[1] + (WILLY_SIZE_PX // 2)
        return float(abs(willy_center_x - portal_center_x) + abs(willy_center_y - portal_center_y))

    def _nearest_key_offset(
        self, level: int, willy_x_px: int, willy_y_px: int, attr_buffer: Optional[bytes] = None
    ) -> Tuple[int, int]:
        """Return (dx_px, dy_px) from Willy to the nearest uncollected key.

        dx positive = key is to the right; dy positive = key is below.
        Returns (0, 0) when no keys remain (portal open).
        """
        if attr_buffer is None:
            attr_buffer = self._read_attr_buffer()
        best_dist = float("inf")
        best_dx = 0
        best_dy = 0
        for _item_attr, (kx_cell, ky_cell) in self._iter_item_slots_for_level(level):
            if not self._is_item_visible_at_cell(attr_buffer, _item_attr, (kx_cell, ky_cell)):
                continue
            kx_px = kx_cell * CELL_SIZE_PX
            ky_px = ky_cell * CELL_SIZE_PX
            dx = kx_px - willy_x_px
            dy = ky_px - willy_y_px
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best_dist = dist
                best_dx = dx
                best_dy = dy
        return best_dx, best_dy

    def _read_state(self) -> ManicState:
        willy_x_px, willy_y_px = self._read_willy_xy()
        level = self._read_u8(LEVEL_ADDR)
        # Use configured item count as a stable baseline; env-level tracking
        # handles collected-key decrementing from score/life transitions.
        keys_remaining = self._count_configured_items_for_level(level)
        attr_buffer = self._read_attr_buffer()
        nearest_key_dx, nearest_key_dy = self._nearest_key_offset(
            level, willy_x_px, willy_y_px, attr_buffer
        )
        return ManicState(
            air=self._read_u16_be(AIR_HI_ADDR),
            level=level,
            lives=self._read_u8(LIVES_ADDR),
            score=parse_score_digits(self.client.read_bytes(SCORE_ADDR, SCORE_LEN)),
            willy_x_px=willy_x_px,
            willy_y_px=willy_y_px,
            willy_flags=self._read_u8(WILLY_FLAGS_ADDR),
            keys_remaining=keys_remaining,
            portal_open=1 if keys_remaining == 0 else 0,
            coverage_ratio=self._coverage_ratio(),
            h_guard_x_cell=self._read_h_guardian_x_cell(),
            nearest_key_dx_px=nearest_key_dx,
            nearest_key_dy_px=nearest_key_dy,
        )
