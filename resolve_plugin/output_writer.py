"""Output writer — writes inference results to disk as EXR/PNG/PPM files.

Separated from the engine manager (which produces numpy arrays) and
from route handlers (which deal with HTTP concerns).  This module
owns the disk-writing responsibility exclusively (SRP).

All writes are atomic: data is written to a ``.tmp`` file first, then
renamed into place.  This prevents the Fuse from reading a partially-
written file if it polls for the output.

PPM output:
    Alongside the full-precision EXR files, this module writes 8-bit
    PPM/PGM copies.  PPM (Portable PixMap, format P6) and PGM (Portable
    GrayMap, format P5) are trivially parseable in Lua 5.1 — the
    language used by Fusion Fuses.  This avoids the need for an EXR
    parser in the Fuse.  The 8-bit precision is acceptable for Fusion's
    viewer/comp preview; the user has the full EXR for final renders.

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


def _atomic_write_bytes(data: bytes, dest_path: str) -> None:
    """Write raw bytes to a file atomically via tmp-file + rename.

    Used for PPM/PGM files which are written as raw bytes rather than
    through OpenCV.

    Args:
        data: Raw bytes to write.
        dest_path: Final output path.

    Raises:
        IOError: If the write or rename fails.
    """
    tmp_path = dest_path + f".{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _array_to_ppm_rgb(img_rgb: np.ndarray) -> bytes:
    """Convert a float32 RGB [H,W,3] array to PPM P6 bytes.

    PPM P6 format:
        Header:  ``P6\\nWIDTH HEIGHT\\n255\\n``
        Body:    width * height * 3 bytes (row-major, top-to-bottom, RGB)

    Args:
        img_rgb: Float32 array [H, W, 3] in [0, 1] range, RGB order.

    Returns:
        Complete PPM file as bytes.
    """
    h, w = img_rgb.shape[:2]
    # Clamp to [0, 1] and convert to uint8
    pixels = (np.clip(img_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return header + pixels.tobytes()


def _array_to_pgm_gray(img_gray: np.ndarray) -> bytes:
    """Convert a float32 grayscale [H,W] array to PGM P5 bytes.

    PGM P5 format:
        Header:  ``P5\\nWIDTH HEIGHT\\n255\\n``
        Body:    width * height bytes (row-major, top-to-bottom)

    Args:
        img_gray: Float32 array [H, W] in [0, 1] range.

    Returns:
        Complete PGM file as bytes.
    """
    h, w = img_gray.shape[:2]
    pixels = (np.clip(img_gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    return header + pixels.tobytes()


def write_inference_outputs(
    result: SingleFrameResult,
    output_dir: str,
    settings: ServiceSettings,
) -> dict[str, str]:
    """Write inference results to disk and return the output paths.

    Creates files in ``output_dir``:

    Full-precision (for final compositing):
        - ``fg.exr``    — foreground RGB (float, EXR half-float PXR24)
        - ``alpha.exr`` — alpha matte (single-channel, EXR half-float)

    Fuse-friendly (8-bit, trivially parseable in Lua 5.1):
        - ``fg.ppm``    — foreground RGB (PPM P6, 8-bit)
        - ``alpha.pgm`` — alpha matte (PGM P5, 8-bit)

    Optional preview:
        - ``comp.png``  — checker-composite preview (8-bit PNG)

    Args:
        result: Inference output arrays from ``EngineManager.infer_single_frame``.
        output_dir: Validated, normalised directory path.
        settings: Service settings (reserved for future format options).

    Returns:
        Dict mapping output names to their absolute file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # ── Foreground EXR (full precision) ──────────────────────────────────
    fg_bgr = cv2.cvtColor(result.fg, cv2.COLOR_RGB2BGR)
    if fg_bgr.dtype != np.float32:
        fg_bgr = fg_bgr.astype(np.float32)
    fg_exr_path = os.path.join(output_dir, "fg.exr")
    _atomic_write_image(fg_bgr, fg_exr_path, EXR_WRITE_FLAGS)
    paths["fg_path"] = fg_exr_path

    # ── Alpha EXR (full precision) ───────────────────────────────────────
    alpha = result.alpha
    if alpha.ndim == 3 and alpha.shape[2] == 1:
        alpha = alpha[:, :, 0]
    if alpha.dtype != np.float32:
        alpha = alpha.astype(np.float32)
    alpha_exr_path = os.path.join(output_dir, "alpha.exr")
    _atomic_write_image(alpha, alpha_exr_path, EXR_WRITE_FLAGS)
    paths["alpha_path"] = alpha_exr_path

    # ── Foreground PPM (8-bit, for the Fusion Fuse) ──────────────────────
    fg_ppm_path = os.path.join(output_dir, "fg.ppm")
    _atomic_write_bytes(_array_to_ppm_rgb(result.fg), fg_ppm_path)
    paths["fg_ppm_path"] = fg_ppm_path

    # ── Alpha PGM (8-bit, for the Fusion Fuse) ──────────────────────────
    alpha_pgm_path = os.path.join(output_dir, "alpha.pgm")
    _atomic_write_bytes(_array_to_pgm_gray(alpha), alpha_pgm_path)
    paths["alpha_pgm_path"] = alpha_pgm_path

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
