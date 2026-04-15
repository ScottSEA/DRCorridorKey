"""TDD tests for ImageNet normalisation / denormalisation.

The CorridorKey model expects inputs normalised with ImageNet
mean and std.  The C++ port must apply identical transforms.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.normalize import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    imagenet_denormalize,
    imagenet_normalize,
)


class TestImagenetNormalize:
    """Test ImageNet normalisation."""

    def test_mean_maps_to_zero(self):
        """An image whose pixels equal the ImageNet mean should normalise to ~0."""
        img = np.stack([
            np.full((4, 4), IMAGENET_MEAN[0]),
            np.full((4, 4), IMAGENET_MEAN[1]),
            np.full((4, 4), IMAGENET_MEAN[2]),
        ], axis=-1).astype(np.float32)
        result = imagenet_normalize(img)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_output_shape_matches_input(self):
        """Shape must be preserved."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        assert imagenet_normalize(img).shape == (64, 64, 3)

    def test_output_dtype_float32(self):
        """Output must be float32."""
        img = np.random.rand(8, 8, 3).astype(np.float32)
        assert imagenet_normalize(img).dtype == np.float32

    def test_per_channel(self):
        """Each channel should be normalised with its own mean/std."""
        # Create an image where each channel has a single known value
        img = np.zeros((1, 1, 3), dtype=np.float32)
        img[0, 0, 0] = 0.485  # R = ImageNet mean R
        img[0, 0, 1] = 0.0    # G = 0
        img[0, 0, 2] = 1.0    # B = 1
        result = imagenet_normalize(img)
        # R channel: (0.485 - 0.485) / 0.229 = 0.0
        assert result[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
        # G channel: (0.0 - 0.456) / 0.224 ≈ -2.036
        assert result[0, 0, 1] == pytest.approx(-0.456 / 0.224, rel=1e-4)
        # B channel: (1.0 - 0.406) / 0.225 ≈ 2.640
        assert result[0, 0, 2] == pytest.approx(0.594 / 0.225, rel=1e-4)


class TestImagenetDenormalize:
    """Test ImageNet denormalisation (inverse of normalize)."""

    def test_round_trip(self):
        """normalize → denormalize should recover the original."""
        rng = np.random.default_rng(42)
        img = rng.random((32, 32, 3), dtype=np.float32)
        recovered = imagenet_denormalize(imagenet_normalize(img))
        np.testing.assert_allclose(recovered, img, atol=1e-5)

    def test_zero_denormalizes_to_mean(self):
        """All-zero normalised input should denormalise to the ImageNet mean."""
        normed = np.zeros((1, 1, 3), dtype=np.float32)
        result = imagenet_denormalize(normed)
        np.testing.assert_allclose(result[0, 0], IMAGENET_MEAN, atol=1e-6)
