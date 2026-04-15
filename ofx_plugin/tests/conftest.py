"""Test configuration for ofx_plugin tests."""

import os
import sys

_project_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
