"""sRGB ↔ linear colour space conversion.

Implements the official IEC 61966-2-1 piecewise sRGB transfer function
using pure numpy (no torch dependency).  This is the reference
implementation for the C++ libtorch port.

The sRGB transfer function has two segments:

    linear → sRGB:
        if linear ≤ 0.0031308:  srgb = linear × 12.92
        else:                   srgb = 1.055 × linear^(1/2.4) − 0.055

    sRGB → linear:
        if srgb ≤ 0.04045:     linear = srgb / 12.92
        else:                   linear = ((srgb + 0.055) / 1.055) ^ 2.4

Both functions:
    - Accept any-shape float32 numpy arrays
    - Clamp negative inputs to zero (physically meaningless)
    - Return float32 arrays of the same shape
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert sRGB values to linear light.

    Args:
        x: Float32 array of any shape, values in [0, 1].
           Negative values are clamped to 0.

    Returns:
        Float32 array of the same shape in linear space.
    """
    x = np.clip(x, 0.0, None).astype(np.float32)

    # Piecewise boundary: sRGB 0.04045
    mask = x <= 0.04045

    # Low segment:  linear = sRGB / 12.92
    # High segment: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
    return np.where(
        mask,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert linear light values to sRGB.

    Args:
        x: Float32 array of any shape, values in [0, 1].
           Negative values are clamped to 0.

    Returns:
        Float32 array of the same shape in sRGB space.
    """
    x = np.clip(x, 0.0, None).astype(np.float32)

    # Piecewise boundary: linear 0.0031308
    mask = x <= 0.0031308

    # Low segment:  sRGB = linear × 12.92
    # High segment: sRGB = 1.055 × linear^(1/2.4) − 0.055
    return np.where(
        mask,
        x * 12.92,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)
