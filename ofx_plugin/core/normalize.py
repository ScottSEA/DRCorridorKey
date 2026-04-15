"""ImageNet normalisation and denormalisation.

The CorridorKey model expects inputs normalised with ImageNet
statistics.  These constants and functions must be replicated
exactly in the C++ libtorch port.

    normalised = (pixel - mean) / std
    denormalised = pixel * std + mean

Applied per-channel on [H, W, 3] float32 arrays in RGB order.
"""

from __future__ import annotations

import numpy as np

# ImageNet channel means and standard deviations (RGB order).
# These are the canonical values used by torchvision.transforms.Normalize.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imagenet_normalize(img: np.ndarray) -> np.ndarray:
    """Normalize an image with ImageNet mean and std.

    Args:
        img: Float32 array [H, W, 3] in RGB order, values in [0, 1].

    Returns:
        Float32 array [H, W, 3] with zero-mean, unit-variance per channel.
    """
    return ((img - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def imagenet_denormalize(img: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation.

    Args:
        img: Float32 array [H, W, 3] that was previously normalised.

    Returns:
        Float32 array [H, W, 3] in original [0, 1] scale.
    """
    return (img * IMAGENET_STD + IMAGENET_MEAN).astype(np.float32)
