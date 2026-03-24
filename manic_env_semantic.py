#!/usr/bin/env python3
"""Semantic grid Manic Miner environment.

Observation: (16, 32, 5) uint8 binary channels, channels-last.
  Ch 0 SOLID:    solid/walkable tiles (floor, wall, crumbling, conveyor) — static
  Ch 1 NASTY:    static lethal tiles (nasty1, nasty2) — static
  Ch 2 WILLY:    Willy's current cell (value 255, all others 0)
  Ch 3 GUARDIAN: guardian's current cell (value 255, all others 0)
  Ch 4 KEY:      uncollected key cells (value 255)

Static channels are built once at reset from ROM tile definitions and the
live game attribute buffer. Willy and guardian positions are read from
live memory each step. Keys start from ROM and are removed as they are
collected (inferred from score delta >= 100 per key).

Actions: 6 discrete (no-op, left, right, jump, jump-left, jump-right).

Reward: score_delta / score_scale per step, −death_penalty on life loss,
        +pathing_reward per newly visited cell.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ml_client import FuseMLClient
from manic_data import (
    AIR_SUPPLY_ADDR,
    AIR_SUPPLY_MAX,
    GAME_ATTR_BUFFER_ADDR,
    ITEMS_COUNT,
    ITEMS_OFFSET,
    ITEM_STRIDE,
    LIVES_ADDR,
    ROOM_DATA_BASE_ADDR,
    ROOM_DATA_SIZE,
    SCORE_ADDR,
    SCORE_LEN,
    SCREEN_CELLS_W,
    WILLY_ATTR_PTR_ADDR,
)

ACTIONS = [
    "0",        # 0: no-op
    "q",        # 1: left
    "w",        # 2: right
    "space",    # 3: jump
    "q+space",  # 4: jump-left
    "w+space",  # 5: jump-right
]

# Observation geometry (game play area only — status bar excluded)
GAME_CELLS_H = 16
GAME_CELLS_W = SCREEN_CELLS_W  # 32
N_CHANNELS = 5

# Channel indices
CH_SOLID    = 0
CH_NASTY    = 1
CH_WILLY    = 2
CH_GUARDIAN = 3
CH_KEY      = 4

# ROM tile definitions: 8 tiles × 9 bytes, at offset 0x0220 within room data.
# Byte 0 of each record is the attribute byte for that tile type.
# Order: background(0), floor(1), crumbling(2), wall(3), conveyor(4),
#        nasty1(5), nasty2(6), extra(7)
TILE_DEFS_OFFSET = 0x0220
TILE_COUNT       = 8
TILE_SIZE        = 9          # bytes per tile record
TILE_SOLID_IDX   = (1, 2, 3, 4)  # floor, crumbling, wall, conveyor
TILE_NASTY_IDX   = (5, 6)        # nasty1, nasty2

# Guardian live runtime address: 2-byte LE pointer into game attr buffer (0x5C00).
# Updated every frame by the guardian movement routine.
GUARDIAN_LIVE_ADDR = 0x80BF


def _decode_score(data: bytes) -> int:
    """Decode a 6-byte Manic Miner BCD score (one decimal digit per byte)."""
    total = 0
    for b in data:
        total = total * 10 + (b & 0x0F)
    return total


class ManicMinerSemanticEnv(gym.Env):
    """Manic Miner environment with a semantic object-type grid observation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        socket_path: str = "/tmp/fuse-ml.sock",
        socket_timeout_s: float = 30.0,
        frames_per_action: int = 3,
        max_steps: int = 4000,
        headless: bool = True,
        visual_pace_ms: int = 0,
        death_penalty: float = 1.0,
        score_scale: float = 100.0,
        warmup_steps: int = 10,
        infinite_air: bool = True,
        pathing_reward: float = 0.01,
    ):
        super().__init__()
        self.socket_path = socket_path
        self.socket_timeout_s = socket_timeout_s
        self.frames_per_action = frames_per_action
        self.max_steps = max_steps
        self.headless = headless
        self.visual_pace_ms = visual_pace_ms
        self.death_penalty = death_penalty
        self.score_scale = score_scale
        self.warmup_steps = warmup_steps
        self.infinite_air = infinite_air
        self.pathing_reward = pathing_reward

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(GAME_CELLS_H, GAME_CELLS_W, N_CHANNELS),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._client: FuseMLClient | None = None
        self._score: int = 0
        self._lives: int = 0
        self._step_count: int = 0
        self._visited_cells: set = set()

        # Static grids built at reset from ROM + live game attr buffer
        self._solid_grid = np.zeros((GAME_CELLS_H, GAME_CELLS_W), dtype=np.uint8)
        self._nasty_grid = np.zeros((GAME_CELLS_H, GAME_CELLS_W), dtype=np.uint8)
        self._key_grid   = np.zeros((GAME_CELLS_H, GAME_CELLS_W), dtype=np.uint8)

        # Active (uncollected) key positions — updated as keys are collected
        self._active_keys: list[tuple[int, int]] = []

    # ── connection ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self._client is None:
            self._client = FuseMLClient(self.socket_path, self.socket_timeout_s)
            if self.headless:
                self._client.mode_headless()
            elif self.visual_pace_ms > 0:
                self._client.mode_visual(self.visual_pace_ms)

    # ── memory readers ─────────────────────────────────────────────────────────

    def _read_state(self) -> tuple[int, int]:
        score_data = self._client.read_bytes(SCORE_ADDR, SCORE_LEN)
        score = _decode_score(score_data)
        lives = self._client.read_bytes(LIVES_ADDR, 1)[0]
        return score, lives

    def _read_willy_cell(self) -> tuple[int, int]:
        """Return Willy's current (x, y) cell within the 32×16 game area."""
        data = self._client.read_bytes(WILLY_ATTR_PTR_ADDR, 2)
        ptr = data[0] | (data[1] << 8)
        offset = ptr - GAME_ATTR_BUFFER_ADDR
        if offset < 0 or offset >= GAME_CELLS_H * GAME_CELLS_W:
            return (0, 0)
        return offset % GAME_CELLS_W, offset // GAME_CELLS_W

    def _read_guardian_cell(self) -> tuple[int, int] | None:
        """Return guardian 1's current (x, y) cell, or None if off-screen."""
        data = self._client.read_bytes(GUARDIAN_LIVE_ADDR, 2)
        attr_addr = data[0] | (data[1] << 8)
        offset = attr_addr - GAME_ATTR_BUFFER_ADDR
        if offset < 0 or offset >= GAME_CELLS_H * GAME_CELLS_W:
            return None
        return offset % GAME_CELLS_W, offset // GAME_CELLS_W

    def _poke_air(self) -> None:
        if self.infinite_air:
            self._client.write_bytes(AIR_SUPPLY_ADDR, bytes([AIR_SUPPLY_MAX]))

    # ── static grid construction ───────────────────────────────────────────────

    def _build_static_grids(self, room_data: bytes) -> None:
        """Build solid and nasty binary grids from ROM tile defs and the live
        game attribute buffer.  Called once per episode after warmup."""

        # Read tile attribute bytes from the ROM block (byte 0 of each 9-byte record)
        tile_attr_bytes = [
            room_data[TILE_DEFS_OFFSET + i * TILE_SIZE]
            for i in range(TILE_COUNT)
        ]
        solid_attrs = np.array([tile_attr_bytes[i] for i in TILE_SOLID_IDX], dtype=np.uint8)
        nasty_attrs = np.array([tile_attr_bytes[i] for i in TILE_NASTY_IDX], dtype=np.uint8)

        # Read the live game attribute buffer (16×32) which reflects the rendered
        # static level layout after the snapshot has finished loading.
        cell_raw = self._client.read_bytes(GAME_ATTR_BUFFER_ADDR, GAME_CELLS_H * GAME_CELLS_W)
        cells = np.frombuffer(cell_raw, dtype=np.uint8).reshape(GAME_CELLS_H, GAME_CELLS_W)

        self._solid_grid = np.isin(cells, solid_attrs).astype(np.uint8) * 255
        self._nasty_grid = np.isin(cells, nasty_attrs).astype(np.uint8) * 255

    def _parse_key_cells(self, room_data: bytes) -> list[tuple[int, int]]:
        """Decode the 5 key positions for level 0 from ROM room data."""
        cells = []
        for slot in range(ITEMS_COUNT):
            off = ITEMS_OFFSET + slot * ITEM_STRIDE
            if off + ITEM_STRIDE > len(room_data):
                break
            attr = room_data[off]
            if attr == 0xFF or attr == 0x00:
                continue
            pos0, pos1, pos2 = room_data[off + 1], room_data[off + 2], room_data[off + 3]
            x = pos0 & 0x1F
            y = ((pos0 >> 5) & 0x07) | ((pos1 & 0x01) << 3)
            if pos2 & 0x10:
                y |= 0x08
            if 0 <= x < GAME_CELLS_W and 0 <= y < GAME_CELLS_H:
                cells.append((x, y))
        return cells

    def _rebuild_key_grid(self) -> None:
        self._key_grid = np.zeros((GAME_CELLS_H, GAME_CELLS_W), dtype=np.uint8)
        for kx, ky in self._active_keys:
            self._key_grid[ky, kx] = 255

    # ── observation assembly ───────────────────────────────────────────────────

    def _build_obs(
        self,
        willy_cell: tuple[int, int],
        guardian_cell: tuple[int, int] | None,
    ) -> np.ndarray:
        obs = np.zeros((GAME_CELLS_H, GAME_CELLS_W, N_CHANNELS), dtype=np.uint8)
        obs[:, :, CH_SOLID]  = self._solid_grid
        obs[:, :, CH_NASTY]  = self._nasty_grid
        obs[:, :, CH_KEY]    = self._key_grid

        wx, wy = willy_cell
        for dy in range(2):  # Willy's sprite is 2 cells tall
            row = wy + dy
            if 0 <= row < GAME_CELLS_H and 0 <= wx < GAME_CELLS_W:
                obs[row, wx, CH_WILLY] = 255

        if guardian_cell is not None:
            gx, gy = guardian_cell
            for dy in range(2):  # guardian sprite is 2 cells tall
                row = gy + dy
                if 0 <= row < GAME_CELLS_H and 0 <= gx < GAME_CELLS_W:
                    obs[row, gx, CH_GUARDIAN] = 255

        return obs

    # ── gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self._client.reset()

        # Warmup: advance the emulator so the snapshot finishes loading
        for _ in range(self.warmup_steps):
            self._client.step_attrs("0", self.frames_per_action)
            self._poke_air()

        # Read the full room data block once (used for tile defs and key positions)
        room_data = self._client.read_bytes(ROOM_DATA_BASE_ADDR, ROOM_DATA_SIZE)

        # Build static grids from tile definitions + live game attr buffer
        self._build_static_grids(room_data)

        # Initialise key tracking
        self._active_keys = self._parse_key_cells(room_data)
        self._rebuild_key_grid()

        self._score, self._lives = self._read_state()
        self._step_count = 0
        self._visited_cells = set()

        willy_cell = self._read_willy_cell()
        guardian_cell = self._read_guardian_cell()
        if self.pathing_reward > 0.0:
            self._visited_cells.add(willy_cell)

        return self._build_obs(willy_cell, guardian_cell), {}

    def step(self, action: int):
        assert self._client is not None, "reset() must be called before step()"

        key_chord = ACTIONS[int(action)]
        self._client.step_attrs(key_chord, self.frames_per_action)
        self._poke_air()
        self._step_count += 1

        new_score, new_lives = self._read_state()
        score_delta = max(0, new_score - self._score)
        reward = score_delta / self.score_scale

        willy_cell = self._read_willy_cell()
        guardian_cell = self._read_guardian_cell()

        if self.pathing_reward > 0.0 and willy_cell not in self._visited_cells:
            self._visited_cells.add(willy_cell)
            reward += self.pathing_reward

        # Remove collected keys (100 points each); use nearest-to-Willy heuristic
        keys_collected = score_delta // 100
        if keys_collected > 0 and self._active_keys:
            wx, wy = willy_cell
            for _ in range(min(keys_collected, len(self._active_keys))):
                nearest = min(
                    self._active_keys,
                    key=lambda k: abs(k[0] - wx) + abs(k[1] - wy),
                )
                self._active_keys.remove(nearest)
            self._rebuild_key_grid()

        life_lost = new_lives < self._lives
        if life_lost:
            reward -= self.death_penalty

        self._score = new_score
        self._lives = new_lives

        terminated = life_lost
        truncated = (not terminated) and (self._step_count >= self.max_steps)

        info = {
            "score": new_score,
            "lives": new_lives,
            "life_lost": life_lost,
            "step": self._step_count,
        }
        return self._build_obs(willy_cell, guardian_cell), reward, terminated, truncated, info

    # ── lifecycle helpers ──────────────────────────────────────────────────────

    def set_runtime_mode(self, headless: bool) -> None:
        if self._client is not None:
            if headless:
                self._client.mode_headless()
            else:
                self._client.mode_visual(self.visual_pace_ms)

    def quit_emulator(self) -> None:
        if self._client is not None:
            try:
                self._client.quit()
            except Exception:
                pass
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def close(self) -> None:
        self.quit_emulator()
