"""Green spill removal (despill).

Removes green contamination from the edges of keyed footage by
capping the green channel at the average of red and blue, then
redistributing the removed green equally to R and B to preserve
overall luminance.

This is the "average" mode despill used by CorridorKey's
``despill_opencv`` function.  The C++ port must match this logic.
"""

from __future__ import annotations

import numpy as np


def despill(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove green spill from an RGB image.

    Args:
        img: Float32 array [H, W, 3] in RGB order, values in [0, 1].
        strength: Despill intensity.  0.0 = no effect, 1.0 = full.

    Returns:
        Float32 array [H, W, 3] with green spill removed.
    """
    if strength <= 0.0:
        return img.copy()

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    # Green limit: average of red and blue
    limit = (r + b) / 2.0

    # Spill is the amount green exceeds the limit (clamped to >= 0)
    spill = np.maximum(g - limit, 0.0)

    # Remove spill from green, add half to R and B (luminance preserving)
    g_new = g - spill
    r_new = r + spill * 0.5
    b_new = b + spill * 0.5

    despilled = np.stack([r_new, g_new, b_new], axis=-1).astype(np.float32)

    # Blend between original and despilled based on strength
    if strength < 1.0:
        return (img * (1.0 - strength) + despilled * strength).astype(np.float32)

    return despilled
