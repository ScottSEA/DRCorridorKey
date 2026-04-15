"""Checkerboard pattern generation for composite previews.

Creates a grayscale checkerboard background that makes alpha
transparency visible when compositing foreground over it.
Used by the CorridorKey post-processing pipeline for the
"comp" preview output.
"""

from __future__ import annotations

import numpy as np


def create_checkerboard(
    width: int,
    height: int,
    checker_size: int = 64,
    color1: float = 0.2,
    color2: float = 0.4,
) -> np.ndarray:
    """Create a grayscale checkerboard pattern.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        checker_size: Size of each checker tile in pixels.
        color1: Brightness of "even" tiles (0–1).
        color2: Brightness of "odd" tiles (0–1).

    Returns:
        Float32 array [H, W, 3] — grayscale RGB checkerboard.
    """
    # Determine which tile each pixel belongs to
    x_tiles = np.arange(width) // checker_size
    y_tiles = np.arange(height) // checker_size

    # 2D grid of tile indices
    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)

    # XOR pattern: (x_tile + y_tile) % 2
    checker = (x_grid + y_grid) % 2

    # Map 0 → color1, 1 → color2
    gray = np.where(checker == 0, color1, color2).astype(np.float32)

    # Expand to 3-channel RGB (all channels identical for grayscale)
    return np.stack([gray, gray, gray], axis=-1)
