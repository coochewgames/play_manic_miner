#!/usr/bin/env python3
"""Manic Miner memory map constants."""

# Memory locations
AIR_SUPPLY_ADDR = 0x80BC
AIR_SUPPLY_MAX  = 0x3F    # full air bar (0x24 = empty → life lost)
LEVEL_ADDR      = 0x8407
LIVES_ADDR      = 0x8457
SCORE_ADDR      = 0x8429
SCORE_LEN       = 6

# Willy position (used for cell-visit exploration tracking)
# 0x806C holds a 2-byte little-endian pointer into the game's internal
# 32×16 attribute buffer at 0x5C00.  offset = ptr - 0x5C00;
# cell_x = offset % 32,  cell_y = offset // 32.
WILLY_ATTR_PTR_ADDR = 0x806C
GAME_ATTR_BUFFER_ADDR = 0x5C00

# Room / item data (for reading key positions from ROM)
ROOM_DATA_BASE_ADDR = 0xB000
ROOM_DATA_SIZE      = 0x0400
ITEMS_OFFSET        = 0x0275   # offset into room data for item (key) list
ITEMS_COUNT         = 5
ITEM_STRIDE         = 5        # bytes per item slot

# Screen geometry
SCREEN_CELLS_W  = 32
SCREEN_CELLS_H  = 24
ATTR_BASE       = 0x5800
ATTR_COUNT      = SCREEN_CELLS_W * SCREEN_CELLS_H  # 768
