"""Conftest for resolve_plugin tests — no torch/torchvision dependencies."""

import os
import sys

# Ensure project root is on sys.path so 'resolve_plugin' is importable.
# This must happen before pytest tries to collect test modules.
_project_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
