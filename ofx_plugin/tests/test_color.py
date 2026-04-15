"""TDD tests for sRGB ↔ linear color space conversion.

These tests define the exact behaviour the C++ port must match.
The reference is the official IEC 61966-2-1 piecewise sRGB transfer
function — the same one used by CorridorKeyModule.core.color_utils.

Test strategy:
    - Known analytical values at the piecewise boundary
    - Round-trip identity (linear → sRGB → linear ≈ original)
    - Edge cases (zero, one, negative, >1.0)
    - Array shape preservation
    - Dtype preservation (float32 in, float32 out)
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.color import linear_to_srgb, srgb_to_linear

# ═══════════════════════════════════════════════════════════════════════
# srgb_to_linear
# ═══════════════════════════════════════════════════════════════════════


class TestSrgbToLinear:
    """Test the sRGB → linear transfer function."""

    def test_zero_stays_zero(self):
        """sRGB 0.0 → linear 0.0 (exact)."""
        result = srgb_to_linear(np.array([0.0], dtype=np.float32))
        assert result[0] == pytest.approx(0.0, abs=1e-7)

    def test_one_stays_one(self):
        """sRGB 1.0 → linear 1.0 (exact)."""
        result = srgb_to_linear(np.array([1.0], dtype=np.float32))
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_piecewise_boundary(self):
        """The transition point at sRGB 0.04045.

        Below this value: linear = sRGB / 12.92
        Above this value: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
        Both branches must agree at the boundary.
        """
        boundary = np.array([0.04045], dtype=np.float32)
        result = srgb_to_linear(boundary)
        # Linear side:  0.04045 / 12.92 = 0.003130804953...
        expected_linear = 0.04045 / 12.92
        assert result[0] == pytest.approx(expected_linear, rel=1e-4)

    def test_mid_gray(self):
        """sRGB 0.5 → linear ~0.2140 (known value)."""
        result = srgb_to_linear(np.array([0.5], dtype=np.float32))
        # ((0.5 + 0.055) / 1.055) ^ 2.4 ≈ 0.21404
        expected = ((0.5 + 0.055) / 1.055) ** 2.4
        assert result[0] == pytest.approx(expected, rel=1e-4)

    def test_negative_clamped_to_zero(self):
        """Negative inputs should be clamped to 0.0."""
        result = srgb_to_linear(np.array([-0.5, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_preserves_shape(self):
        """Output shape must match input shape."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = srgb_to_linear(img)
        assert result.shape == img.shape

    def test_preserves_dtype(self):
        """Output must be float32."""
        img = np.array([0.5], dtype=np.float32)
        result = srgb_to_linear(img)
        assert result.dtype == np.float32

    def test_monotonic(self):
        """srgb_to_linear must be monotonically increasing."""
        x = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        y = srgb_to_linear(x)
        assert np.all(np.diff(y) >= 0)

    def test_above_one_handled(self):
        """Values > 1.0 (HDR / super-white) should not crash.

        The function should apply the high-segment formula.
        Result must be > 1.0 (brighter than reference white).
        """
        result = srgb_to_linear(np.array([1.5], dtype=np.float32))
        assert result[0] > 1.0
        assert np.isfinite(result[0])


# ═══════════════════════════════════════════════════════════════════════
# linear_to_srgb
# ═══════════════════════════════════════════════════════════════════════


class TestLinearToSrgb:
    """Test the linear → sRGB transfer function."""

    def test_zero_stays_zero(self):
        """Linear 0.0 → sRGB 0.0 (exact)."""
        result = linear_to_srgb(np.array([0.0], dtype=np.float32))
        assert result[0] == pytest.approx(0.0, abs=1e-7)

    def test_one_stays_one(self):
        """Linear 1.0 → sRGB 1.0 (exact)."""
        result = linear_to_srgb(np.array([1.0], dtype=np.float32))
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_piecewise_boundary(self):
        """The transition point at linear 0.0031308.

        Below: sRGB = linear * 12.92
        Above: sRGB = 1.055 * linear^(1/2.4) - 0.055
        Both branches must agree at the boundary.
        """
        boundary = np.array([0.0031308], dtype=np.float32)
        result = linear_to_srgb(boundary)
        expected = 0.0031308 * 12.92  # ≈ 0.04045
        assert result[0] == pytest.approx(expected, rel=1e-3)

    def test_mid_gray(self):
        """Linear 0.2140 → sRGB ~0.5 (inverse of the srgb_to_linear test)."""
        linear_val = ((0.5 + 0.055) / 1.055) ** 2.4  # ≈ 0.21404
        result = linear_to_srgb(np.array([linear_val], dtype=np.float32))
        assert result[0] == pytest.approx(0.5, rel=1e-3)

    def test_negative_clamped_to_zero(self):
        """Negative inputs should be clamped to 0.0."""
        result = linear_to_srgb(np.array([-0.5], dtype=np.float32))
        assert result[0] == pytest.approx(0.0, abs=1e-7)

    def test_preserves_shape(self):
        """Output shape must match input shape."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = linear_to_srgb(img)
        assert result.shape == img.shape

    def test_preserves_dtype(self):
        """Output must be float32."""
        result = linear_to_srgb(np.array([0.5], dtype=np.float32))
        assert result.dtype == np.float32

    def test_monotonic(self):
        """linear_to_srgb must be monotonically increasing."""
        x = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        y = linear_to_srgb(x)
        assert np.all(np.diff(y) >= 0)


# ═══════════════════════════════════════════════════════════════════════
# Round-trip identity
# ═══════════════════════════════════════════════════════════════════════


class TestRoundTrip:
    """Verify that sRGB → linear → sRGB ≈ identity."""

    def test_round_trip_srgb(self):
        """sRGB → linear → sRGB should recover the original values."""
        original = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        recovered = linear_to_srgb(srgb_to_linear(original))
        np.testing.assert_allclose(recovered, original, atol=1e-5)

    def test_round_trip_linear(self):
        """linear → sRGB → linear should recover the original values."""
        original = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        recovered = srgb_to_linear(linear_to_srgb(original))
        np.testing.assert_allclose(recovered, original, atol=1e-5)

    def test_round_trip_image(self):
        """Round-trip on a full [H,W,3] image."""
        rng = np.random.default_rng(42)
        img = rng.random((64, 64, 3), dtype=np.float32)
        recovered = linear_to_srgb(srgb_to_linear(img))
        np.testing.assert_allclose(recovered, img, atol=1e-5)
