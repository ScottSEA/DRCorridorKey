"""Path validation and security utilities.

The Resolve service accepts arbitrary file paths from HTTP requests.
Since it binds to localhost only, the risk surface is limited — but
defence-in-depth is still warranted.  This module centralises all
path-safety checks so route handlers stay clean.

Security measures:
    1. Paths are normalised (resolve symlinks, collapse ``..``).
    2. Optional allowlist restricts paths to configured directory roots.
    3. Existence checks with clear error messages.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import ServiceSettings


def _normalise(raw_path: str) -> str:
    """Resolve a raw path string to a canonical, absolute form.

    Collapses ``..``, ``~``, and symlinks so that allowlist comparisons
    are reliable and cannot be bypassed with path tricks.
    """
    return str(Path(os.path.expanduser(raw_path)).resolve())


def validate_input_path(raw_path: str, settings: ServiceSettings) -> str:
    """Validate and normalise an *input* file path.

    Args:
        raw_path: The path string received in the HTTP request.
        settings: Current service settings (for allowed_roots).

    Returns:
        The normalised, validated absolute path.

    Raises:
        ValueError: If the path fails any security or existence check.
    """
    norm = _normalise(raw_path)

    # Allowlist check — only when the operator configured roots
    if settings.allowed_roots:
        normalised_roots = [_normalise(r) for r in settings.allowed_roots]
        if not any(norm.startswith(root) for root in normalised_roots):
            raise ValueError(
                f"Path '{norm}' is outside the allowed roots: "
                f"{settings.allowed_roots}"
            )

    if not os.path.isfile(norm):
        raise ValueError(f"File does not exist: {norm}")

    return norm


def validate_output_dir(raw_path: str, settings: ServiceSettings) -> str:
    """Validate and normalise an *output* directory path.

    Creates the directory if it does not exist (``mkdir -p`` semantics).

    Args:
        raw_path: The directory path string from the request or config.
        settings: Current service settings (for allowed_roots).

    Returns:
        The normalised, validated absolute directory path.

    Raises:
        ValueError: If the path fails any security check.
    """
    norm = _normalise(raw_path)

    # Allowlist check (same logic as input paths)
    if settings.allowed_roots:
        normalised_roots = [_normalise(r) for r in settings.allowed_roots]
        if not any(norm.startswith(root) for root in normalised_roots):
            raise ValueError(
                f"Output directory '{norm}' is outside the allowed roots: "
                f"{settings.allowed_roots}"
            )

    os.makedirs(norm, exist_ok=True)
    return norm
