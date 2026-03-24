#!/usr/bin/env python3
"""Visual Manic Miner gymnasium environment using the ZX Spectrum attribute grid."""

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
    SCREEN_CELLS_H,
    SCREEN_CELLS_W,
    WILLY_ATTR_PTR_ADDR,
)

# Discrete actions → Fuse key chord strings.
# Manic Miner controls: Q=left, W=right, SPACE=jump.
ACTIONS = [
    "0",         # 0: no-op
    "q",         # 1: move left
    "w",         # 2: move right
    "space",     # 3: jump (in place)
    "q+space",   # 4: jump left
    "w+space",   # 5: jump right
]


def _decode_score(data: bytes) -> int:
    """Decode a 6-byte Manic Miner score.

    Each byte holds one decimal digit (0-9, one digit per byte, most-significant first).
    """
    total = 0
    for b in data:
        total = total * 10 + (b & 0x0F)
    return total


class ManicMinerVisualEnv(gym.Env):
    """Gymnasium environment using the 32×24 ZX Spectrum attribute grid as observation.

    Single-frame observation shape: (24, 32, 1) uint8, channels-last.
    Wrap with VecFrameStack(n_stack=4) in training to get (24, 32, 4).

    Actions: 6 discrete (no-op, left, right, jump, jump-left, jump-right).

    Reward:
      +score_delta / score_scale  per step (keys, portal give positive score)
      −death_penalty              when a life is lost

    Episode ends when a life is lost (terminated) or max_steps is reached (truncated).
    """

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
        pathing_reward: float = 0.0,
        key_proximity_reward: float = 0.0,
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
        self.key_proximity_reward = key_proximity_reward

        # Key proximity reward — targets the first (green) key only.
        # Positive reward proportional to getting closer each step.
        # Disabled once that key is collected (score increases).
        self._target_key_cell: tuple[int, int] | None = None
        self._target_key_active: bool = False
        self._prev_key_dist: float = 0.0

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(SCREEN_CELLS_H, SCREEN_CELLS_W, 1),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._client: FuseMLClient | None = None
        self._score: int = 0
        self._lives: int = 0
        self._step_count: int = 0
        self._visited_cells: set = set()

    # ── connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self._client is None:
            self._client = FuseMLClient(self.socket_path, self.socket_timeout_s)
            if self.headless:
                self._client.mode_headless()
            elif self.visual_pace_ms > 0:
                self._client.mode_visual(self.visual_pace_ms)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _read_state(self) -> tuple[int, int]:
        """Return (score, lives) read directly from emulator memory."""
        score_data = self._client.read_bytes(SCORE_ADDR, SCORE_LEN)
        score = _decode_score(score_data)
        lives = self._client.read_bytes(LIVES_ADDR, 1)[0]
        return score, lives

    def _read_willy_cell(self) -> tuple[int, int]:
        """Return Willy's current (cell_x, cell_y) from the game's attr buffer pointer.

        0x806C holds a 2-byte LE pointer into the 32×16 game attr buffer at 0x5C00.
        offset = ptr - 0x5C00;  cell_x = offset % 32,  cell_y = offset // 32.
        """
        data = self._client.read_bytes(WILLY_ATTR_PTR_ADDR, 2)
        ptr = data[0] | (data[1] << 8)
        offset = ptr - GAME_ATTR_BUFFER_ADDR
        if offset < 0 or offset >= 512:
            return (0, 0)
        return offset % 32, offset // 32

    def _read_key_cells_level0(self) -> list[tuple[int, int]]:
        """Read the 5 key cell positions for level 0 from ROM data."""
        base = ROOM_DATA_BASE_ADDR  # level 0
        room = self._client.read_bytes(base, ROOM_DATA_SIZE)
        cells = []
        for slot in range(ITEMS_COUNT):
            off = ITEMS_OFFSET + slot * ITEM_STRIDE
            if off + ITEM_STRIDE > len(room):
                break
            attr = room[off]
            if attr == 0xFF or attr == 0x00:
                continue
            # Decode cell from bytes 1-3 of the item slot.
            pos0, pos1, pos2 = room[off + 1], room[off + 2], room[off + 3]
            x = pos0 & 0x1F
            y = ((pos0 >> 5) & 0x07) | ((pos1 & 0x01) << 3)
            if pos2 & 0x10:
                y |= 0x08
            if 0 <= x < SCREEN_CELLS_W and 0 <= y < SCREEN_CELLS_H:
                cells.append((x, y))
        return cells

    @staticmethod
    def _pick_target_key(cells: list[tuple[int, int]]) -> tuple[int, int] | None:
        """Return the first (green) key: rightmost key in the middle vertical band (y 6–12)."""
        candidates = [(x, y) for x, y in cells if x >= 27 and 6 <= y <= 12]
        if candidates:
            return max(candidates, key=lambda c: c[0])
        # Fallback: rightmost key anywhere
        return max(cells, key=lambda c: c[0]) if cells else None

    @staticmethod
    def _key_dist(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _poke_air(self) -> None:
        if self.infinite_air:
            self._client.write_bytes(AIR_SUPPLY_ADDR, bytes([AIR_SUPPLY_MAX]))

    def _to_obs(self, attr_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(attr_bytes, dtype=np.uint8).reshape(SCREEN_CELLS_H, SCREEN_CELLS_W)
        return arr[:, :, np.newaxis]  # (24, 32, 1) channels-last

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self._client.reset()

        # Warmup: step with no-op so the snapshot finishes loading.
        attr_bytes = b"\x00" * (SCREEN_CELLS_H * SCREEN_CELLS_W)
        for _ in range(self.warmup_steps):
            _, attr_bytes = self._client.step_attrs("0", self.frames_per_action)
            self._poke_air()

        self._score, self._lives = self._read_state()
        self._step_count = 0
        self._visited_cells = set()
        willy_cell = self._read_willy_cell()
        if self.pathing_reward > 0.0:
            self._visited_cells.add(willy_cell)

        # Set up key proximity target for this episode.
        try:
            key_cells = self._read_key_cells_level0()
            self._target_key_cell = self._pick_target_key(key_cells)
        except Exception:
            self._target_key_cell = None
        if self._target_key_cell is not None:
            self._target_key_active = True
            self._prev_key_dist = self._key_dist(willy_cell, self._target_key_cell)
        else:
            self._target_key_active = False

        return self._to_obs(attr_bytes), {}

    def step(self, action: int):
        assert self._client is not None, "reset() must be called before step()"

        key_chord = ACTIONS[int(action)]
        _, attr_bytes = self._client.step_attrs(key_chord, self.frames_per_action)
        self._poke_air()
        self._step_count += 1

        new_score, new_lives = self._read_state()

        score_delta = max(0, new_score - self._score)
        reward = score_delta / self.score_scale

        cell = self._read_willy_cell()

        if self.pathing_reward > 0.0:
            if cell not in self._visited_cells:
                self._visited_cells.add(cell)
                reward += self.pathing_reward

        if self.key_proximity_reward > 0.0 and self._target_key_active and self._target_key_cell is not None:
            new_dist = self._key_dist(cell, self._target_key_cell)
            reward += self.key_proximity_reward * max(0.0, self._prev_key_dist - new_dist)
            self._prev_key_dist = new_dist
            if score_delta >= 100:  # key collected — disable for rest of episode
                self._target_key_active = False

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
        return self._to_obs(attr_bytes), reward, terminated, truncated, info

    # ── lifecycle helpers (called via env_method) ─────────────────────────────

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
