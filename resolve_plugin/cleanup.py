"""Temp directory garbage collection.

The Resolve service writes output files (EXR, PPM, PGM, PNG) into
per-request subdirectories under ``settings.temp_dir``.  Over time
these accumulate.  This module provides age-based cleanup so old
request directories are removed automatically.

Cleanup runs:
    - At service startup (``purge_old_outputs``)
    - On demand via ``POST /cleanup`` endpoint
    - Never deletes files younger than ``max_age_hours``

Design notes:
    - Only removes directories that look like request output dirs
      (12-char hex names created by the /infer route).
    - Never touches files outside the configured temp_dir.
    - Errors during deletion are logged but don't propagate — cleanup
      is best-effort and must never crash the service.
"""

from __future__ import annotations

import logging
import os
import shutil
import time

logger = logging.getLogger(__name__)

# Default: delete request directories older than 24 hours
DEFAULT_MAX_AGE_HOURS = 24

# Request output directories are named with 12-char hex strings
# (e.g. "a1b2c3d4e5f6").  This length check prevents accidental
# deletion of unrelated directories.
_REQUEST_DIR_NAME_LEN = 12


def _is_request_dir(name: str) -> bool:
    """Check if a directory name looks like a request output directory.

    Request directories are created by the /infer route as 12-char
    hex strings (from uuid4().hex[:12]).

    Args:
        name: Directory basename to check.

    Returns:
        True if the name matches the expected pattern.
    """
    if len(name) != _REQUEST_DIR_NAME_LEN:
        return False
    try:
        int(name, 16)
        return True
    except ValueError:
        return False


def purge_old_outputs(
    temp_dir: str,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> int:
    """Remove request output directories older than ``max_age_hours``.

    Scans ``temp_dir`` for subdirectories that match the request
    directory naming pattern (12-char hex) and deletes any whose
    modification time is older than the threshold.

    Args:
        temp_dir: Root temp directory (``settings.temp_dir``).
        max_age_hours: Maximum age in hours before a directory is
            eligible for deletion.

    Returns:
        Number of directories removed.
    """
    if not os.path.isdir(temp_dir):
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    try:
        entries = os.listdir(temp_dir)
    except OSError as exc:
        logger.warning("Cannot list temp dir %s: %s", temp_dir, exc)
        return 0

    for name in entries:
        full_path = os.path.join(temp_dir, name)

        # Only consider directories that match our naming pattern
        if not os.path.isdir(full_path):
            continue
        if not _is_request_dir(name):
            continue

        # Check modification time
        try:
            mtime = os.path.getmtime(full_path)
        except OSError:
            continue

        if mtime < cutoff:
            try:
                shutil.rmtree(full_path)
                removed += 1
                logger.debug("Cleaned up old request dir: %s", name)
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", full_path, exc)

    if removed > 0:
        logger.info(
            "Cleaned up %d old request dir(s) from %s (older than %.0fh)",
            removed,
            temp_dir,
            max_age_hours,
        )

    return removed
