"""TDD tests for green spill removal (despill).

The despill algorithm removes green contamination from edges of
keyed footage.  It caps the green channel at the average of R and B,
then redistributes the removed green equally to R and B to
preserve luminance.
"""

from __future__ import annotations

import numpy as np

from ofx_plugin.core.despill import despill


class TestDespill:
    """Test the green spill removal function."""

    def test_no_spill_unchanged(self):
        """Pixels with G ≤ avg(R,B) should not be modified."""
        # R=0.5, G=0.3, B=0.5 → avg(R,B)=0.5, G < limit → no change
        img = np.array([[[0.5, 0.3, 0.5]]], dtype=np.float32)
        result = despill(img, strength=1.0)
        np.testing.assert_allclose(result, img, atol=1e-7)

    def test_pure_green_fully_despilled(self):
        """Pure green pixel [0, 1, 0]: limit = 0, spill = 1.0.

        After despill: R = 0 + 0.5 = 0.5, G = 0, B = 0 + 0.5 = 0.5
        """
        img = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        result = despill(img, strength=1.0)
        expected = np.array([[[0.5, 0.0, 0.5]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_partial_spill(self):
        """R=0.3, G=0.8, B=0.3 → limit=0.3, spill=0.5.

        G_new = 0.3, R_new = 0.3 + 0.25 = 0.55, B_new = 0.3 + 0.25 = 0.55
        """
        img = np.array([[[0.3, 0.8, 0.3]]], dtype=np.float32)
        result = despill(img, strength=1.0)
        expected = np.array([[[0.55, 0.3, 0.55]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_strength_zero_no_change(self):
        """strength=0.0 should return the original image unchanged."""
        img = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        result = despill(img, strength=0.0)
        np.testing.assert_allclose(result, img, atol=1e-7)

    def test_strength_half_blends(self):
        """strength=0.5 should blend 50% original + 50% despilled."""
        img = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        full_despill = np.array([[[0.5, 0.0, 0.5]]], dtype=np.float32)
        expected = img * 0.5 + full_despill * 0.5
        result = despill(img, strength=0.5)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_preserves_shape(self):
        """Output shape must match input."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        assert despill(img, strength=1.0).shape == (64, 64, 3)

    def test_preserves_dtype(self):
        """Output must be float32."""
        img = np.random.rand(4, 4, 3).astype(np.float32)
        assert despill(img, strength=1.0).dtype == np.float32

    def test_green_channel_never_exceeds_limit(self):
        """After full despill, G ≤ avg(R, B) everywhere."""
        rng = np.random.default_rng(42)
        img = rng.random((64, 64, 3), dtype=np.float32)
        result = despill(img, strength=1.0)
        limit = (result[..., 0] + result[..., 2]) / 2.0
        assert np.all(result[..., 1] <= limit + 1e-6)
