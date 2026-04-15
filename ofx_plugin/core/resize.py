"""Image and mask resizing utilities.

Wraps OpenCV resize with the interpolation methods used by the
CorridorKey pipeline:
    - Images: bilinear (cv2.INTER_LINEAR) for upscale/downscale
    - Masks:  bilinear for smooth edges

The C++ port should use the equivalent OpenCV or libtorch
interpolation functions.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_image(
    img: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize an RGB image using bilinear interpolation.

    Args:
        img: Float32 array [H, W, 3].
        target_width: Desired output width.
        target_height: Desired output height.

    Returns:
        Float32 array [target_height, target_width, 3].
    """
    return cv2.resize(
        img,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def resize_mask(
    mask: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize a single-channel mask using bilinear interpolation.

    Accepts [H, W] or [H, W, 1] input.  Always returns [H, W].

    Args:
        mask: Float32 array [H, W] or [H, W, 1].
        target_width: Desired output width.
        target_height: Desired output height.

    Returns:
        Float32 array [target_height, target_width].
    """
    # Squeeze [H, W, 1] → [H, W]
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    return cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
