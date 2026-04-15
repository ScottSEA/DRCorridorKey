"""Cross-validation tests — compare OFX reference vs upstream CorridorKey.

These tests import both our pure-numpy implementations and the
upstream CorridorKeyModule.core.color_utils torch implementations,
feed them identical inputs, and verify the outputs match within
floating-point tolerance.

This guarantees our reference implementations are faithful to the
original — if these tests pass, the C++ port can target our numpy
code with confidence.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_project_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Our reference implementations (pure numpy)
from ofx_plugin.core.color import (
    linear_to_srgb as ref_linear_to_srgb,
    srgb_to_linear as ref_srgb_to_linear,
)
from ofx_plugin.core.composite import (
    composite_straight as ref_composite_straight,
    premultiply as ref_premultiply,
)
from ofx_plugin.core.despill import despill as ref_despill

# Upstream implementations (torch + numpy)
from CorridorKeyModule.core.color_utils import (
    composite_straight as upstream_composite_straight,
    despill_opencv as upstream_despill,
    linear_to_srgb as upstream_linear_to_srgb,
    premultiply as upstream_premultiply,
    srgb_to_linear as upstream_srgb_to_linear,
)


class TestColorCrossValidation:
    """Verify our color functions match upstream exactly."""

    @pytest.fixture
    def ramp(self):
        """Linear ramp from 0 to 1 as float32."""
        return np.linspace(0.0, 1.0, 256, dtype=np.float32)

    @pytest.fixture
    def random_image(self):
        """Random 64×64 RGB image."""
        rng = np.random.default_rng(42)
        return rng.random((64, 64, 3), dtype=np.float32)

    def test_srgb_to_linear_matches(self, ramp):
        """Our srgb_to_linear must match upstream on a full ramp."""
        ours = ref_srgb_to_linear(ramp)
        theirs = upstream_srgb_to_linear(ramp)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_linear_to_srgb_matches(self, ramp):
        """Our linear_to_srgb must match upstream on a full ramp."""
        ours = ref_linear_to_srgb(ramp)
        theirs = upstream_linear_to_srgb(ramp)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_srgb_to_linear_image(self, random_image):
        """Full image srgb_to_linear must match upstream."""
        ours = ref_srgb_to_linear(random_image)
        theirs = upstream_srgb_to_linear(random_image)
        np.testing.assert_allclose(ours, theirs, atol=1e-5)

    def test_linear_to_srgb_image(self, random_image):
        """Full image linear_to_srgb must match upstream."""
        ours = ref_linear_to_srgb(random_image)
        theirs = upstream_linear_to_srgb(random_image)
        np.testing.assert_allclose(ours, theirs, atol=1e-5)

    def test_srgb_to_linear_torch_tensor(self, ramp):
        """Our numpy output must match upstream's torch tensor output."""
        ours = ref_srgb_to_linear(ramp)
        theirs_tensor = upstream_srgb_to_linear(torch.from_numpy(ramp))
        theirs = theirs_tensor.numpy()
        np.testing.assert_allclose(ours, theirs, atol=1e-5)


class TestCompositeCrossValidation:
    """Verify our compositing functions match upstream."""

    def test_premultiply_matches(self):
        """Our premultiply must match upstream."""
        fg = np.random.rand(32, 32, 3).astype(np.float32)
        alpha = np.random.rand(32, 32, 1).astype(np.float32)
        ours = ref_premultiply(fg, alpha)
        theirs = upstream_premultiply(fg, alpha)
        np.testing.assert_allclose(ours, theirs, atol=1e-7)

    def test_composite_straight_matches(self):
        """Our composite_straight must match upstream."""
        fg = np.random.rand(32, 32, 3).astype(np.float32)
        bg = np.random.rand(32, 32, 3).astype(np.float32)
        alpha = np.random.rand(32, 32, 1).astype(np.float32)
        ours = ref_composite_straight(fg, bg, alpha)
        theirs = upstream_composite_straight(fg, bg, alpha)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)


class TestDespillCrossValidation:
    """Verify our despill matches upstream despill_opencv."""

    def test_full_strength_matches(self):
        """Full-strength despill must match upstream."""
        rng = np.random.default_rng(42)
        img = rng.random((64, 64, 3), dtype=np.float32)
        ours = ref_despill(img, strength=1.0)
        theirs = upstream_despill(img, strength=1.0)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_partial_strength_matches(self):
        """Half-strength despill must match upstream."""
        rng = np.random.default_rng(99)
        img = rng.random((32, 32, 3), dtype=np.float32)
        ours = ref_despill(img, strength=0.5)
        theirs = upstream_despill(img, strength=0.5)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_zero_strength_matches(self):
        """Zero-strength despill must match upstream (identity)."""
        img = np.random.rand(16, 16, 3).astype(np.float32)
        ours = ref_despill(img, strength=0.0)
        theirs = upstream_despill(img, strength=0.0)
        np.testing.assert_allclose(ours, theirs, atol=1e-7)

    def test_green_screen_plate(self):
        """Heavily green image should match upstream."""
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[:, :, 1] = 0.9  # strong green
        img[:, :, 0] = 0.1
        img[:, :, 2] = 0.1
        ours = ref_despill(img, strength=1.0)
        theirs = upstream_despill(img, strength=1.0)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)
