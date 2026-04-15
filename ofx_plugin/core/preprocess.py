"""Preprocessing pipeline — prepare raw inputs for the model.

Combines resize, color conversion, and normalisation into the exact
sequence the CorridorKey GreenFormer model expects:

    1. Resize image and mask to ``model_size × model_size``
    2. If the input is linear, convert to sRGB (the model was trained on sRGB)
    3. Normalize the image with ImageNet mean/std
    4. Concatenate image [H,W,3] + mask [H,W,1] → [H,W,4]

The C++ port must replicate this pipeline exactly — any deviation
in step ordering or constants will produce wrong predictions.
"""

from __future__ import annotations

import numpy as np

from .color import linear_to_srgb
from .normalize import imagenet_normalize
from .resize import resize_image, resize_mask


def preprocess(
    image: np.ndarray,
    mask: np.ndarray,
    model_size: int = 2048,
    input_is_linear: bool = False,
) -> np.ndarray:
    """Prepare a frame + mask for model inference.

    Args:
        image: Float32 [H, W, 3] RGB input image in [0, 1].
        mask: Float32 [H, W] or [H, W, 1] alpha hint in [0, 1].
        model_size: Target resolution (square) for the model.
        input_is_linear: If True, convert image from linear → sRGB
            before normalisation.

    Returns:
        Float32 [model_size, model_size, 4] — normalised RGB (ch 0-2)
        + raw mask (ch 3).
    """
    # 1. Resize both to model input dimensions
    img_resized = resize_image(image, model_size, model_size)
    mask_resized = resize_mask(mask, model_size, model_size)

    # 2. Color space conversion (linear → sRGB if needed)
    if input_is_linear:
        img_resized = linear_to_srgb(img_resized)

    # 3. ImageNet normalisation (the model expects this)
    img_normed = imagenet_normalize(img_resized)

    # 4. Concatenate image + mask into 4 channels
    # Mask is NOT normalised — it's a raw [0, 1] hint
    mask_channel = mask_resized[:, :, np.newaxis]  # [H, W] → [H, W, 1]
    return np.concatenate([img_normed, mask_channel], axis=-1).astype(np.float32)
