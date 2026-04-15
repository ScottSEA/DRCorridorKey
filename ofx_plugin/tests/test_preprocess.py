"""TDD tests for the full preprocessing pipeline.

The preprocessing pipeline transforms a raw green-screen frame and
alpha hint into the 4-channel tensor the model expects:
    1. Resize image + mask to model input size
    2. If input is linear, convert to sRGB
    3. Normalize with ImageNet mean/std
    4. Concatenate image (3ch) + mask (1ch) → 4ch

The C++ port must produce identical output.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.preprocess import preprocess


class TestPreprocess:
    """Test the complete preprocessing pipeline."""

    def test_output_shape(self):
        """Output must be [model_size, model_size, 4]."""
        img = np.random.rand(100, 200, 3).astype(np.float32)
        mask = np.random.rand(100, 200).astype(np.float32)
        result = preprocess(img, mask, model_size=64)
        assert result.shape == (64, 64, 4)

    def test_output_dtype(self):
        """Output must be float32."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.random.rand(64, 64).astype(np.float32)
        result = preprocess(img, mask, model_size=32)
        assert result.dtype == np.float32

    def test_first_three_channels_normalized(self):
        """Channels 0-2 should be normalised (not in [0, 1] anymore)."""
        # All-zero image normalised should have negative values (due to mean subtraction)
        img = np.zeros((32, 32, 3), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)
        result = preprocess(img, mask, model_size=32, input_is_linear=False)
        # After normalisation of zeros: (0 - mean) / std → negative values
        assert result[:, :, 0].mean() < 0  # R channel
        assert result[:, :, 1].mean() < 0  # G channel
        assert result[:, :, 2].mean() < 0  # B channel

    def test_fourth_channel_is_mask(self):
        """Channel 3 should contain the resized mask (not normalised)."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.ones((64, 64), dtype=np.float32)  # all-white mask
        result = preprocess(img, mask, model_size=32)
        # Mask channel should be ~1.0 everywhere
        np.testing.assert_allclose(result[:, :, 3], 1.0, atol=0.05)

    def test_linear_input_gets_converted(self):
        """When input_is_linear=True, the image should be converted to sRGB before normalizing."""
        # Linear mid-gray ≈ 0.2140 → sRGB ≈ 0.5
        linear_val = ((0.5 + 0.055) / 1.055) ** 2.4
        img = np.full((32, 32, 3), linear_val, dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)

        result_linear = preprocess(img, mask, model_size=32, input_is_linear=True)
        result_srgb = preprocess(
            np.full((32, 32, 3), 0.5, dtype=np.float32),
            mask,
            model_size=32,
            input_is_linear=False,
        )
        # Both should produce similar normalised values
        np.testing.assert_allclose(
            result_linear[:, :, :3],
            result_srgb[:, :, :3],
            atol=0.05,
        )

    def test_srgb_input_not_converted(self):
        """When input_is_linear=False, no color space conversion should happen."""
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)

        result = preprocess(img, mask, model_size=32, input_is_linear=False)
        # After normalising 0.5: (0.5 - mean) / std
        # R: (0.5 - 0.485) / 0.229 ≈ 0.065
        expected_r = (0.5 - 0.485) / 0.229
        assert result[16, 16, 0] == pytest.approx(expected_r, abs=0.05)

    def test_mask_resize_matches_image(self):
        """Mask should be resized to the same dimensions as the image."""
        img = np.random.rand(100, 200, 3).astype(np.float32)
        mask = np.random.rand(50, 100).astype(np.float32)  # different size
        result = preprocess(img, mask, model_size=64)
        assert result.shape == (64, 64, 4)
