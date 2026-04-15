"""TDD tests for matte cleanup (despeckle).

The despeckle algorithm removes small disconnected alpha islands
(like tracking markers) while preserving the main subject.  It
uses connected components analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.matte import clean_matte


class TestCleanMatte:
    """Test connected-components-based matte cleanup."""

    def test_large_region_preserved(self):
        """A large connected region should survive cleanup."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[10:90, 10:90] = 1.0  # 80×80 = 6400 pixels
        result = clean_matte(mask, area_threshold=300)
        # Center of the large region should still be ~1.0
        assert result[50, 50] > 0.5

    def test_small_region_removed(self):
        """A small isolated region should be removed."""
        mask = np.zeros((100, 100), dtype=np.float32)
        # Large region
        mask[10:90, 10:90] = 1.0
        # Tiny island — 4 pixels (well below threshold of 300)
        mask[2:4, 2:4] = 1.0
        result = clean_matte(mask, area_threshold=300)
        # Small island should be gone
        assert result[3, 3] < 0.1
        # Large region should survive
        assert result[50, 50] > 0.5

    def test_empty_mask_stays_empty(self):
        """An all-zero mask should remain all-zero."""
        mask = np.zeros((64, 64), dtype=np.float32)
        result = clean_matte(mask, area_threshold=300)
        np.testing.assert_array_equal(result, 0.0)

    def test_full_mask_stays_full(self):
        """An all-one mask (one big component) should survive."""
        mask = np.ones((64, 64), dtype=np.float32)
        result = clean_matte(mask, area_threshold=300)
        # Should be mostly 1.0 (dilation/blur may affect edges slightly)
        assert result[32, 32] > 0.5

    def test_output_shape_matches(self):
        """Output shape must match input."""
        mask = np.random.rand(64, 128).astype(np.float32)
        result = clean_matte(mask, area_threshold=100)
        assert result.shape == (64, 128)

    def test_output_dtype_float32(self):
        """Output must be float32."""
        mask = np.zeros((32, 32), dtype=np.float32)
        result = clean_matte(mask, area_threshold=100)
        assert result.dtype == np.float32

    def test_values_in_range(self):
        """Output values must be in [0, 1]."""
        mask = np.random.rand(64, 64).astype(np.float32)
        result = clean_matte(mask, area_threshold=100)
        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_handles_3d_input(self):
        """[H, W, 1] input should be handled and returned as [H, W, 1]."""
        mask = np.zeros((64, 64, 1), dtype=np.float32)
        mask[10:50, 10:50, 0] = 1.0
        result = clean_matte(mask, area_threshold=100)
        assert result.shape == (64, 64, 1)
        assert result[30, 30, 0] > 0.5

    def test_threshold_zero_keeps_everything(self):
        """area_threshold=0 should keep all components."""
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[0:2, 0:2] = 1.0  # 4-pixel island
        mask[30:60, 30:60] = 1.0  # large region
        result = clean_matte(mask, area_threshold=0)
        # Both regions should survive (or at least not be zeroed)
        assert result[1, 1] > 0.0
        assert result[45, 45] > 0.5
