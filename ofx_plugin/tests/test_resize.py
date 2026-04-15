"""TDD tests for image resizing utilities.

The CorridorKey pipeline resizes inputs to 2048×2048 before inference,
then resizes outputs back to the original resolution.  The C++ port
must use identical interpolation methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.resize import resize_image, resize_mask


class TestResizeImage:
    """Test RGB image resizing."""

    def test_output_shape(self):
        """Output must match the requested dimensions."""
        img = np.random.rand(100, 200, 3).astype(np.float32)
        result = resize_image(img, target_width=64, target_height=32)
        assert result.shape == (32, 64, 3)

    def test_identity_resize(self):
        """Resizing to the same size should preserve values (approximately)."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = resize_image(img, target_width=64, target_height=64)
        np.testing.assert_allclose(result, img, atol=1e-5)

    def test_preserves_dtype(self):
        """Output must be float32."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = resize_image(img, target_width=64, target_height=64)
        assert result.dtype == np.float32

    def test_preserves_value_range(self):
        """Values in [0, 1] should stay in [0, 1] after resize."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = resize_image(img, target_width=128, target_height=128)
        assert result.min() >= -0.01  # small tolerance for interpolation
        assert result.max() <= 1.01

    def test_downscale(self):
        """Downscaling should produce a smaller image."""
        img = np.random.rand(256, 256, 3).astype(np.float32)
        result = resize_image(img, target_width=64, target_height=64)
        assert result.shape == (64, 64, 3)

    def test_non_square(self):
        """Non-square resize should work correctly."""
        img = np.random.rand(100, 200, 3).astype(np.float32)
        result = resize_image(img, target_width=50, target_height=25)
        assert result.shape == (25, 50, 3)


class TestResizeMask:
    """Test single-channel mask resizing."""

    def test_output_shape(self):
        """Output must match requested dimensions as [H, W]."""
        mask = np.random.rand(100, 200).astype(np.float32)
        result = resize_mask(mask, target_width=64, target_height=32)
        assert result.shape == (32, 64)

    def test_preserves_dtype(self):
        """Output must be float32."""
        mask = np.random.rand(32, 32).astype(np.float32)
        result = resize_mask(mask, target_width=64, target_height=64)
        assert result.dtype == np.float32

    def test_binary_mask_stays_reasonable(self):
        """A binary mask should remain mostly binary after resize."""
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = resize_mask(mask, target_width=128, target_height=128)
        # Interior should be ~1.0, exterior should be ~0.0
        assert result[64, 64] > 0.9  # center of the white region
        assert result[0, 0] < 0.1    # corner (black region)

    def test_handles_3d_input(self):
        """[H, W, 1] masks should be accepted and returned as [H, W]."""
        mask = np.random.rand(32, 32, 1).astype(np.float32)
        result = resize_mask(mask, target_width=16, target_height=16)
        assert result.shape == (16, 16)
