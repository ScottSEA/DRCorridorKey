"""Matte cleanup — remove small disconnected alpha islands.

Uses OpenCV connected-components analysis to find and remove
regions smaller than a pixel-area threshold.  Optionally dilates
and blurs the surviving mask to restore soft edges.

This is the numpy/OpenCV reference implementation matching
``CorridorKeyModule.core.color_utils.clean_matte_opencv``.
The C++ port should use the equivalent ``cv::connectedComponentsWithStats``.
"""

from __future__ import annotations

import cv2
import numpy as np


def clean_matte(
    alpha: np.ndarray,
    area_threshold: int = 300,
    dilation: int = 15,
    blur_size: int = 5,
) -> np.ndarray:
    """Remove small disconnected components from an alpha matte.

    Args:
        alpha: Float32 array [H, W] or [H, W, 1] in [0, 1].
        area_threshold: Minimum pixel area to keep a component.
            Components smaller than this are removed.  0 keeps everything.
        dilation: Radius in pixels to dilate the surviving mask.
            Restores edges that were trimmed by the binary threshold.
        blur_size: Kernel half-size for Gaussian blur of the mask.
            Softens the hard edges from thresholding + dilation.

    Returns:
        Float32 array of the same shape as input, in [0, 1].
    """
    # Handle [H, W, 1] input
    is_3d = alpha.ndim == 3
    if is_3d:
        alpha_2d = alpha[:, :, 0]
    else:
        alpha_2d = alpha

    # Threshold to binary for connected-components analysis
    mask_8u = (alpha_2d > 0.5).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_8u,
        connectivity=8,
    )

    # Build a mask keeping only components above the area threshold.
    # Label 0 is always background — skip it.
    cleaned = np.zeros_like(mask_8u)
    for i in range(1, num_labels):
        if area_threshold == 0 or stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            cleaned[labels == i] = 255

    # Dilate to restore edges of large regions
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        cleaned = cv2.dilate(cleaned, kernel)

    # Blur for soft edges
    if blur_size > 0:
        k = int(blur_size * 2 + 1)
        cleaned = cv2.GaussianBlur(cleaned, (k, k), 0)

    # Convert back to [0, 1] float and multiply with original alpha
    safe_zone = cleaned.astype(np.float32) / 255.0
    result = alpha_2d * safe_zone

    if is_3d:
        result = result[:, :, np.newaxis]

    return result.astype(np.float32)
