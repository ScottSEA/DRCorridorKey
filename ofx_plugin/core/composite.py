"""Alpha compositing operations.

Implements the four fundamental compositing functions used by the
CorridorKey post-processing pipeline.  The C++ port must produce
identical results.

All functions operate on [H, W, C] float32 numpy arrays.
"""

from __future__ import annotations

import numpy as np


def premultiply(fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Premultiply foreground by alpha.

    Args:
        fg: Float32 [H, W, 3] foreground colour.
        alpha: Float32 [H, W, 1] alpha channel.

    Returns:
        Float32 [H, W, 3] premultiplied foreground.
    """
    return (fg * alpha).astype(np.float32)


def unpremultiply(
    fg: np.ndarray,
    alpha: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Reverse premultiplication (divide by alpha).

    Uses an epsilon to prevent division by zero in transparent regions.

    Args:
        fg: Float32 [H, W, 3] premultiplied foreground.
        alpha: Float32 [H, W, 1] alpha channel.
        eps: Small value added to alpha to avoid division by zero.

    Returns:
        Float32 [H, W, 3] straight (unpremultiplied) foreground.
    """
    return (fg / (alpha + eps)).astype(np.float32)


def composite_straight(
    fg: np.ndarray,
    bg: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Composite straight-alpha foreground over background.

    Formula: FG × α + BG × (1 − α)

    Args:
        fg: Float32 [H, W, 3] straight foreground.
        bg: Float32 [H, W, 3] background.
        alpha: Float32 [H, W, 1] alpha channel.

    Returns:
        Float32 [H, W, 3] composited image.
    """
    return (fg * alpha + bg * (1.0 - alpha)).astype(np.float32)


def composite_premul(
    fg: np.ndarray,
    bg: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Composite premultiplied foreground over background.

    Formula: FG + BG × (1 − α)

    Args:
        fg: Float32 [H, W, 3] premultiplied foreground.
        bg: Float32 [H, W, 3] background.
        alpha: Float32 [H, W, 1] alpha channel.

    Returns:
        Float32 [H, W, 3] composited image.
    """
    return (fg + bg * (1.0 - alpha)).astype(np.float32)
