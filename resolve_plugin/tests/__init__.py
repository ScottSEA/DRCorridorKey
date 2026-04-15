"""Conftest for resolve_plugin tests.

This conftest exists in a subdirectory to isolate resolve_plugin tests
from the root conftest.py, which has autouse fixtures that import
clip_manager → CorridorKeyModule → torch/torchvision.  Those heavy
dependencies are not available on every test machine.

The resolve_plugin tests only need pydantic, numpy, and pytest.
"""
