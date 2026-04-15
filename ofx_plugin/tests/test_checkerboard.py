"""TDD tests for checkerboard pattern generation.

The checkerboard is used as the background for composite previews
so the user can see alpha transparency.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.checkerboard import create_checkerboard


class TestCreateCheckerboard:
    """Test checkerboard pattern generation."""

    def test_output_shape(self):
        """Output must be [H, W, 3]."""
        result = create_checkerboard(100, 50)
        assert result.shape == (50, 100, 3)

    def test_output_dtype(self):
        """Output must be float32."""
        result = create_checkerboard(64, 64)
        assert result.dtype == np.float32

    def test_two_distinct_colors(self):
        """Checkerboard should contain exactly two distinct values."""
        result = create_checkerboard(128, 128, checker_size=64)
        unique = np.unique(result[:, :, 0])  # single channel
        assert len(unique) == 2

    def test_custom_colors(self):
        """Custom color1 and color2 should appear in the output."""
        result = create_checkerboard(
            128,
            128,
            checker_size=64,
            color1=0.1,
            color2=0.9,
        )
        unique = sorted(np.unique(result[:, :, 0]))
        assert unique[0] == pytest.approx(0.1, abs=1e-6)
        assert unique[1] == pytest.approx(0.9, abs=1e-6)

    def test_pattern_alternates(self):
        """Adjacent tiles should have different colors."""
        result = create_checkerboard(128, 128, checker_size=64)
        # Top-left tile (0,0) and tile (0,1) should differ
        val_00 = result[0, 0, 0]
        val_01 = result[0, 64, 0]
        assert val_00 != val_01

    def test_three_channels_equal(self):
        """All three channels should be identical (grayscale)."""
        result = create_checkerboard(64, 64)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_values_in_range(self):
        """All values must be in [0, 1]."""
        result = create_checkerboard(256, 256)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
