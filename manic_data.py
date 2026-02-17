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
H_GUARD_OFFSET = 0x02BE
V_GUARD_OFFSET = 0x02DD
GUARD_MAX_COUNT = 4
GUARD_STRIDE = 7

# Screen/pathing geometry
CELL_SIZE_PX = 8
WILLY_SIZE_PX = 16
SCREEN_CELLS_W = 32
SCREEN_CELLS_H = 24
PATHING_TOTAL_CELLS = SCREEN_CELLS_W * SCREEN_CELLS_H

# Observation layout:
# [air, level, lives, score, willy_x_px, willy_y_px,
#  keys_remaining, portal_open, willy_flags, coverage_ratio]
OBS_RAW_MIN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
OBS_RAW_MAX = np.array(
    [0x3FFF, 255, 255, 999999, 255, 191, 255, 1, 255, 1],
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
        attrs: Set[int] = set()
        for offset in (NASTY1_OFFSET, NASTY2_OFFSET):
            if offset >= len(room):
                continue
            raw_attr = int(room[offset])
            if raw_attr in (0x00, 0xFF):
                continue
            attrs.add(raw_attr & 0x7F)
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

    def _cells_above_willy(self, state: ManicState) -> Set[Tuple[int, int]]:
        x0 = max(0, min(SCREEN_CELLS_W - 1, state.willy_x_px // CELL_SIZE_PX))
        y0 = max(0, min(SCREEN_CELLS_H - 1, state.willy_y_px // CELL_SIZE_PX))
        x1 = max(0, min(SCREEN_CELLS_W - 1, (state.willy_x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        above_y = y0 - 1
        if above_y < 0:
            return set()
        return {(x, above_y) for x in range(x0, x1 + 1)}

    def _cells_above_right_willy(self, state: ManicState) -> Set[Tuple[int, int]]:
        x1 = max(0, min(SCREEN_CELLS_W - 1, (state.willy_x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        y0 = max(0, min(SCREEN_CELLS_H - 1, state.willy_y_px // CELL_SIZE_PX))
        target_x = x1 + 1
        target_y = y0 - 1
        if target_x < 0 or target_x >= SCREEN_CELLS_W:
            return set()
        if target_y < 0 or target_y >= SCREEN_CELLS_H:
            return set()
        return {(target_x, target_y)}

    def _cells_above_left_willy(self, state: ManicState) -> Set[Tuple[int, int]]:
        x0 = max(0, min(SCREEN_CELLS_W - 1, state.willy_x_px // CELL_SIZE_PX))
        y0 = max(0, min(SCREEN_CELLS_H - 1, state.willy_y_px // CELL_SIZE_PX))
        target_x = x0 - 1
        target_y = y0 - 1
        if target_x < 0 or target_x >= SCREEN_CELLS_W:
            return set()
        if target_y < 0 or target_y >= SCREEN_CELLS_H:
            return set()
        return {(target_x, target_y)}

    def _cells_three_right_willy(self, state: ManicState) -> Set[Tuple[int, int]]:
        x1 = max(0, min(SCREEN_CELLS_W - 1, (state.willy_x_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        y0 = max(0, min(SCREEN_CELLS_H - 1, state.willy_y_px // CELL_SIZE_PX))
        y1 = max(0, min(SCREEN_CELLS_H - 1, (state.willy_y_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        target_x = x1 + 3
        if target_x < 0 or target_x >= SCREEN_CELLS_W:
            return set()
        return {(target_x, y) for y in range(y0, y1 + 1)}

    def _cells_three_left_willy(self, state: ManicState) -> Set[Tuple[int, int]]:
        x0 = max(0, min(SCREEN_CELLS_W - 1, state.willy_x_px // CELL_SIZE_PX))
        y0 = max(0, min(SCREEN_CELLS_H - 1, state.willy_y_px // CELL_SIZE_PX))
        y1 = max(0, min(SCREEN_CELLS_H - 1, (state.willy_y_px + WILLY_SIZE_PX - 1) // CELL_SIZE_PX))
        target_x = x0 - 3
        if target_x < 0 or target_x >= SCREEN_CELLS_W:
            return set()
        return {(target_x, y) for y in range(y0, y1 + 1)}

    def _distance_to_exit_px(self, state: ManicState) -> Optional[float]:
        portal_xy = self._read_portal_xy_for_level(state.level)
        if portal_xy is None:
            return None
        willy_center_x = state.willy_x_px + (WILLY_SIZE_PX // 2)
        willy_center_y = state.willy_y_px + (WILLY_SIZE_PX // 2)
        portal_center_x = portal_xy[0] + (WILLY_SIZE_PX // 2)
        portal_center_y = portal_xy[1] + (WILLY_SIZE_PX // 2)
        return float(abs(willy_center_x - portal_center_x) + abs(willy_center_y - portal_center_y))

    def _read_state(self) -> ManicState:
        willy_x_px, willy_y_px = self._read_willy_xy()
        level = self._read_u8(LEVEL_ADDR)
        # Use configured item count as a stable baseline; env-level tracking
        # handles collected-key decrementing from score/life transitions.
        keys_remaining = self._count_configured_items_for_level(level)
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
        )
