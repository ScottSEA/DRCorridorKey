"""Postprocessing pipeline — transform raw model output to final images.

After the model produces raw alpha [H,W] and fg [H,W,3] at model
resolution, this pipeline transforms them back to usable outputs:

    1. Resize alpha + fg back to original resolution (bilinear)
    2. Clean matte — remove small disconnected islands (despeckle)
    3. Despill — remove green contamination from edges
    4. Convert fg from sRGB → linear (model outputs sRGB)
    5. Premultiply fg by alpha
    6. Pack into RGBA [H,W,4]
    7. Optionally generate a checker-composite preview

The C++ port must replicate this pipeline exactly.
"""

from __future__ import annotations

import numpy as np

from .checkerboard import create_checkerboard
from .color import linear_to_srgb, srgb_to_linear
from .composite import composite_straight, premultiply
from .despill import despill as despill_fn
from .matte import clean_matte
from .resize import resize_image, resize_mask


def postprocess(
    alpha: np.ndarray,
    fg: np.ndarray,
    original_height: int,
    original_width: int,
    *,
    despill_strength: float = 1.0,
    auto_despeckle: bool = True,
    despeckle_size: int = 400,
    generate_comp: bool = False,
) -> dict[str, np.ndarray]:
    """Transform raw model output into final compositing-ready images.

    Args:
        alpha: Float32 [H, W] or [H, W, 1] raw model alpha in [0, 1].
        fg: Float32 [H, W, 3] raw model foreground in sRGB [0, 1].
        original_height: Height to resize outputs to.
        original_width: Width to resize outputs to.
        despill_strength: Green spill removal intensity (0 = off).
        auto_despeckle: Whether to run matte cleanup.
        despeckle_size: Minimum island pixel count for cleanup.
        generate_comp: Whether to produce a checker-composite preview.

    Returns:
        Dict with keys:
            "alpha": [H, W] float32 alpha matte
            "fg": [H, W, 3] float32 foreground (sRGB, despilled)
            "processed": [H, W, 4] float32 RGBA (linear, premultiplied)
            "comp": [H, W, 3] float32 checker composite (optional)
    """
    # Squeeze [H, W, 1] → [H, W] if needed
    if alpha.ndim == 3 and alpha.shape[2] == 1:
        alpha = alpha[:, :, 0]

    # 1. Resize back to original resolution
    alpha_full = resize_mask(alpha, original_width, original_height)
    fg_full = resize_image(fg, original_width, original_height)

    # 2. Matte cleanup (despeckle)
    if auto_despeckle and despeckle_size > 0:
        alpha_full = clean_matte(alpha_full, area_threshold=despeckle_size)

    # 3. Despill green from foreground
    if despill_strength > 0.0:
        fg_full = despill_fn(fg_full, strength=despill_strength)

    # 4. Convert fg from sRGB → linear for compositing
    fg_linear = srgb_to_linear(fg_full)

    # 5. Premultiply
    alpha_3ch = alpha_full[:, :, np.newaxis]  # [H, W, 1] for broadcasting
    fg_premul = premultiply(fg_linear, alpha_3ch)

    # 6. Pack RGBA
    processed = np.concatenate(
        [fg_premul, alpha_full[:, :, np.newaxis]], axis=-1,
    ).astype(np.float32)

    result: dict[str, np.ndarray] = {
        "alpha": alpha_full,
        "fg": fg_full,
        "processed": processed,
    }

    # 7. Optional checker composite (sRGB for display)
    if generate_comp:
        checker = create_checkerboard(original_width, original_height)
        # Composite straight fg over checker
        comp = composite_straight(fg_full, checker, alpha_3ch)
        result["comp"] = comp.astype(np.float32)

    return result
