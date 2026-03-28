#!/usr/bin/env python3
"""BFS-based waypoint pathfinder for Manic Miner's semantic grid.

Movement model (approximates Manic Miner physics):
  Walk:  (x±1, y)           if target cell is free
  Fall:  (x,   y+1)         if target cell is free (no floor beneath current cell)
  Jump:  (x+dx, y-dy)       for dy in (1,2,3,4), dx in (-3,-2,-1,0,1,2,3)
                             only from a ground cell (solid directly below)
                             target cell must be free

  Conveyor surface cells (solid directly below is a conveyor tile):
    Walk left only — no rightward walk.
    Jump left only (dx ≤ 0) — belt forces leftward momentum on any jump arc.

'Free' means the cell is not solid and not nasty.

Usage:
    graph = build_movement_graph(solid_grid, nasty_grid, conveyor_grid)
    waypoint = find_waypoint(graph, willy_cell, active_keys)
    # waypoint is a (x, y) tuple or None if unreachable
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
Graph = Dict[Cell, List[Cell]]

GAME_H = 16
GAME_W = 32

# Jump offsets: (dx, dy) where dy > 0 means upward.
# dy=1,2,3 with dx up to ±3 covers standard platform jumps.
# dy=4 is required to reach the y=0 ceiling keys from the y=4 walking
# surface above the y=5 all-solid platform floor in Central Cavern.
_JUMP_OFFSETS = [
    (dx, dy)
    for dy in (1, 2, 3, 4)
    for dx in range(-3, 4)
]


def _is_free(solid: np.ndarray, nasty: np.ndarray, x: int, y: int) -> bool:
    """Return True if (x, y) is within bounds, not solid, and not nasty."""
    if x < 0 or x >= GAME_W or y < 0 or y >= GAME_H:
        return False
    return solid[y, x] == 0 and nasty[y, x] == 0


def _has_floor(solid: np.ndarray, x: int, y: int) -> bool:
    """Return True if there is a solid tile directly below (x, y)."""
    below = y + 1
    if below >= GAME_H:
        return True  # bottom of screen acts as floor
    return solid[below, x] != 0


def _on_conveyor(conveyor: Optional[np.ndarray], x: int, y: int) -> bool:
    """Return True if the cell directly below (x, y) is a conveyor tile."""
    if conveyor is None:
        return False
    below = y + 1
    if below >= GAME_H:
        return False
    return conveyor[below, x] != 0


def build_movement_graph(
    solid_grid: np.ndarray,
    nasty_grid: np.ndarray,
    conveyor_grid: Optional[np.ndarray] = None,
) -> Graph:
    """Build a directed adjacency list of reachable moves on the grid.

    Parameters
    ----------
    solid_grid    : (16, 32) uint8 array, 255 = solid tile
    nasty_grid    : (16, 32) uint8 array, 255 = nasty tile
    conveyor_grid : (16, 32) uint8 array, 255 = conveyor tile (optional).
                    Cells whose floor tile is a conveyor are restricted to
                    leftward walk only — no rightward walk and no jumps.

    Returns
    -------
    graph : dict mapping each free cell to a list of reachable neighbour cells
    """
    graph: Graph = {}

    for y in range(GAME_H):
        for x in range(GAME_W):
            if not _is_free(solid_grid, nasty_grid, x, y):
                continue

            neighbours: List[Cell] = []
            on_ground = _has_floor(solid_grid, x, y)
            on_conv   = _on_conveyor(conveyor_grid, x, y)

            # Walk left / right.
            # On a conveyor surface the belt forces Willy left, so rightward
            # walk is omitted — the agent must ride the belt off to the left.
            for nx in (x - 1, x + 1):
                if on_conv and nx > x:   # skip rightward walk on conveyor
                    continue
                if _is_free(solid_grid, nasty_grid, nx, y):
                    neighbours.append((nx, y))

            # Fall (if not on ground)
            if not on_ground:
                ny = y + 1
                if _is_free(solid_grid, nasty_grid, x, ny):
                    neighbours.append((x, ny))

            # Jump (only from ground).
            # On a conveyor surface the belt forces Willy left, so only
            # leftward jump arcs (dx ≤ 0) are reachable.
            if on_ground:
                for dx, dy in _JUMP_OFFSETS:
                    if on_conv and dx > 0:   # belt prevents rightward jump arc
                        continue
                    nx, ny = x + dx, y - dy  # dy > 0 → upward
                    if _is_free(solid_grid, nasty_grid, nx, ny):
                        neighbours.append((nx, ny))

            graph[(x, y)] = neighbours

    return graph


def find_bfs_distance(
    graph: Graph,
    start: Cell,
    targets: List[Cell],
) -> float:
    """Return the BFS distance from *start* to the nearest reachable target.

    Returns 0 if start is already a target, float('inf') if no target is
    reachable.
    """
    if not targets or start not in graph:
        return float("inf")

    target_set: Set[Cell] = set(targets)

    if start in target_set:
        return 0

    visited: Set[Cell] = {start}
    queue: deque[Tuple[Cell, int]] = deque()

    for neighbour in graph.get(start, []):
        if neighbour not in visited:
            visited.add(neighbour)
            queue.append((neighbour, 1))

    while queue:
        cell, dist = queue.popleft()
        if cell in target_set:
            return dist
        for neighbour in graph.get(cell, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, dist + 1))

    return float("inf")


def find_waypoint(
    graph: Graph,
    start: Cell,
    targets: List[Cell],
) -> Optional[Cell]:
    """BFS from *start* toward the nearest reachable target cell.

    Returns the first step along the shortest path, i.e. the cell Willy
    should move into next.  Returns None if no target is reachable or
    the start cell is already a target.

    Parameters
    ----------
    graph   : movement graph from build_movement_graph
    start   : Willy's current (x, y) cell
    targets : list of goal cells (uncollected key positions)
    """
    if not targets or start not in graph:
        return None

    target_set: Set[Cell] = set(targets)

    if start in target_set:
        return None  # already at a target

    # BFS: track (current_cell, first_step_from_start)
    visited: Set[Cell] = {start}
    queue: deque[Tuple[Cell, Cell]] = deque()

    for neighbour in graph.get(start, []):
        if neighbour not in visited:
            visited.add(neighbour)
            queue.append((neighbour, neighbour))

    while queue:
        cell, first_step = queue.popleft()
        if cell in target_set:
            return first_step
        for neighbour in graph.get(cell, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, first_step))

    return None  # no target reachable
