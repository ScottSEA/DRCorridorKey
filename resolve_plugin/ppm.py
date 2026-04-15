"""PPM/PGM encoding and atomic file writing.

Pure-numpy utilities with no torch/cv2/backend dependencies.  This
module is intentionally free of heavy imports so that:

    1. The Fuse interchange tests can import it directly
    2. The output_writer can reuse it without duplication

PPM (Portable PixMap, P6) and PGM (Portable GrayMap, P5) are dead-
simple binary formats: a short ASCII header followed by raw pixel
bytes.  They're used as the interchange format between the Python
service and the Lua Fuse because Lua 5.1 can parse them trivially.
"""

from __future__ import annotations

import os
import uuid

import numpy as np


def array_to_ppm_rgb(img_rgb: np.ndarray) -> bytes:
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
    pixels = (np.clip(img_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return header + pixels.tobytes()


def array_to_pgm_gray(img_gray: np.ndarray) -> bytes:
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


def atomic_write_bytes(data: bytes, dest_path: str) -> None:
    """Write raw bytes to a file atomically via tmp-file + rename.

    Data is written to a temporary file first, then renamed into
    place.  This prevents readers (e.g. the Fuse) from seeing a
    partially-written file.

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
