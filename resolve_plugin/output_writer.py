"""Output writer — writes inference results to disk as EXR/PNG files.

Separated from the engine manager (which produces numpy arrays) and
from route handlers (which deal with HTTP concerns).  This module
owns the disk-writing responsibility exclusively (SRP).

All writes are atomic: data is written to a ``.tmp`` file first, then
renamed into place.  This prevents the Fuse from reading a partially-
written EXR if it polls for the output file.

Colour-space note:
    The inference engine returns foreground in sRGB colour space and
    alpha in linear.  This module writes them as-is — the Fuse is
    responsible for any colour-space conversion needed by Resolve.
"""

from __future__ import annotations

import logging
import os
import uuid

import cv2
import numpy as np

from backend.frame_io import EXR_WRITE_FLAGS

from .config import ServiceSettings
from .engine_manager import SingleFrameResult

logger = logging.getLogger(__name__)


def _atomic_write_image(
    img: np.ndarray,
    dest_path: str,
    write_flags: list[int] | None = None,
) -> None:
    """Write an image atomically via tmp-file + rename.

    Args:
        img: Image array in BGR/BGRA channel order (OpenCV convention).
        dest_path: Final output path.
        write_flags: Optional cv2.imwrite flags (e.g. EXR compression).

    Raises:
        IOError: If the write or rename fails.
    """
    tmp_path = dest_path + f".{uuid.uuid4().hex[:8]}.tmp"
    try:
        if write_flags:
            ok = cv2.imwrite(tmp_path, img, write_flags)
        else:
            ok = cv2.imwrite(tmp_path, img)

        if not ok:
            raise IOError(f"cv2.imwrite failed for: {dest_path}")

        # Atomic replace — works on both POSIX and Windows
        os.replace(tmp_path, dest_path)
    except Exception:
        # Clean up the temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_inference_outputs(
    result: SingleFrameResult,
    output_dir: str,
    settings: ServiceSettings,
) -> dict[str, str]:
    """Write inference results to disk and return the output paths.

    Creates three files in ``output_dir``:
        - ``fg.exr``    — foreground RGB (linear float, EXR half-float)
        - ``alpha.exr`` — alpha matte (single-channel, EXR half-float)
        - ``comp.png``  — checker-composite preview (8-bit PNG, optional)

    Args:
        result: Inference output arrays from ``EngineManager.infer_single_frame``.
        output_dir: Validated, normalised directory path.
        settings: Service settings (currently unused, reserved for future
                  format options).

    Returns:
        Dict mapping output names to their absolute file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # ── Foreground EXR ───────────────────────────────────────────────────
    fg_bgr = cv2.cvtColor(result.fg, cv2.COLOR_RGB2BGR)
    # Ensure float32 for EXR
    if fg_bgr.dtype != np.float32:
        fg_bgr = fg_bgr.astype(np.float32)
    fg_path = os.path.join(output_dir, "fg.exr")
    _atomic_write_image(fg_bgr, fg_path, EXR_WRITE_FLAGS)
    paths["fg_path"] = fg_path

    # ── Alpha EXR ────────────────────────────────────────────────────────
    alpha = result.alpha
    # Squeeze to 2D if [H, W, 1]
    if alpha.ndim == 3 and alpha.shape[2] == 1:
        alpha = alpha[:, :, 0]
    if alpha.dtype != np.float32:
        alpha = alpha.astype(np.float32)
    alpha_path = os.path.join(output_dir, "alpha.exr")
    _atomic_write_image(alpha, alpha_path, EXR_WRITE_FLAGS)
    paths["alpha_path"] = alpha_path

    # ── Composite PNG (optional preview) ─────────────────────────────────
    if result.comp is not None:
        comp_bgr = cv2.cvtColor(
            (np.clip(result.comp, 0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        comp_path = os.path.join(output_dir, "comp.png")
        _atomic_write_image(comp_bgr, comp_path)
        paths["comp_path"] = comp_path

    logger.info("Wrote outputs to %s: %s", output_dir, list(paths.keys()))
    return paths
