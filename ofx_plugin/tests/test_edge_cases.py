"""Edge-case tests for production VFX inputs.

Covers scenarios that real-world EXR/VFX pipelines produce but
that clean [0,1] test data misses: HDR values, negative pixels,
NaN/Inf, alpha overshoot, soft mattes, and singleton-channel masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.color import linear_to_srgb, srgb_to_linear
from ofx_plugin.core.composite import composite_straight, premultiply
from ofx_plugin.core.despill import despill
from ofx_plugin.core.matte import clean_matte
from ofx_plugin.core.postprocess import postprocess
from ofx_plugin.core.preprocess import preprocess


# ═══════════════════════════════════════════════════════════════════════
# HDR / out-of-range values
# ═══════════════════════════════════════════════════════════════════════


class TestHDRInputs:
    """Test behaviour with values outside [0, 1]."""

    def test_srgb_to_linear_hdr(self):
        """sRGB values > 1.0 (super-whites) must not produce NaN/Inf."""
        x = np.array([1.5, 2.0, 5.0], dtype=np.float32)
        result = srgb_to_linear(x)
        assert np.all(np.isfinite(result))
        assert np.all(result > 1.0)

    def test_linear_to_srgb_hdr(self):
        """Linear values > 1.0 must not produce NaN/Inf."""
        x = np.array([1.5, 2.0, 5.0], dtype=np.float32)
        result = linear_to_srgb(x)
        assert np.all(np.isfinite(result))
        assert np.all(result > 1.0)

    def test_negative_rgb_through_despill(self):
        """Negative pixel values (valid in EXR) should not crash despill."""
        img = np.array([[[-0.1, 0.5, -0.2]]], dtype=np.float32)
        result = despill(img, strength=1.0)
        assert np.all(np.isfinite(result))

    def test_negative_rgb_through_preprocess(self):
        """Negative pixel values should survive preprocessing."""
        img = np.full((16, 16, 3), -0.1, dtype=np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        result = preprocess(img, mask, model_size=8)
        assert np.all(np.isfinite(result))

    def test_alpha_above_one(self):
        """Alpha > 1.0 (model overshoot) should not crash postprocess."""
        alpha = np.full((16, 16), 1.05, dtype=np.float32)
        fg = np.full((16, 16, 3), 0.5, dtype=np.float32)
        result = postprocess(alpha, fg, 16, 16, auto_despeckle=False)
        assert np.all(np.isfinite(result["processed"]))

    def test_alpha_slightly_negative(self):
        """Alpha slightly below 0 (model undershoot) should not crash."""
        alpha = np.full((16, 16), -0.05, dtype=np.float32)
        fg = np.full((16, 16, 3), 0.5, dtype=np.float32)
        result = postprocess(alpha, fg, 16, 16, auto_despeckle=False)
        assert np.all(np.isfinite(result["processed"]))

    def test_premultiply_hdr(self):
        """HDR foreground * alpha > 1 should not clamp or crash."""
        fg = np.array([[[2.0, 3.0, 1.5]]], dtype=np.float32)
        alpha = np.array([[[0.8]]], dtype=np.float32)
        result = premultiply(fg, alpha)
        assert np.all(np.isfinite(result))
        assert result[0, 0, 0] == pytest.approx(1.6, abs=1e-6)

    def test_composite_hdr_fg(self):
        """Compositing with HDR foreground should not clamp."""
        fg = np.array([[[2.0, 0.0, 0.0]]], dtype=np.float32)
        bg = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        alpha = np.array([[[1.0]]], dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        assert result[0, 0, 0] == pytest.approx(2.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# NaN / Inf propagation
# ═══════════════════════════════════════════════════════════════════════


class TestNaNInf:
    """Test that NaN/Inf inputs don't silently propagate into outputs."""

    def test_nan_in_color_conversion(self):
        """NaN input to color conversion should produce NaN (not crash)."""
        x = np.array([float("nan")], dtype=np.float32)
        result = srgb_to_linear(x)
        # NaN propagation is acceptable — crashing is not
        assert not np.any(np.isinf(result))

    def test_inf_in_color_conversion(self):
        """Inf input should not crash color conversion."""
        x = np.array([float("inf")], dtype=np.float32)
        result = linear_to_srgb(x)
        assert result.dtype == np.float32

    def test_nan_in_despill(self):
        """NaN in image should not crash despill."""
        img = np.array([[[float("nan"), 0.5, 0.5]]], dtype=np.float32)
        result = despill(img, strength=1.0)
        assert result.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════
# Soft mattes and matte cleanup edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestSoftMattes:
    """Test matte cleanup with semi-transparent values."""

    def test_all_below_threshold(self):
        """A matte where all values are below 0.5 should be zeroed out.

        clean_matte thresholds at 0.5 for connected components — values
        below that become background.  This is a known limitation.
        """
        mask = np.full((64, 64), 0.49, dtype=np.float32)
        result = clean_matte(mask, area_threshold=100)
        assert result.max() < 0.01

    def test_boundary_value_0_5(self):
        """Pixels at exactly 0.5 are below the > 0.5 threshold."""
        mask = np.full((64, 64), 0.5, dtype=np.float32)
        result = clean_matte(mask, area_threshold=100)
        # 0.5 is NOT > 0.5, so it's treated as background
        assert result.max() < 0.01

    def test_just_above_threshold(self):
        """Pixels at 0.51 should survive as foreground."""
        mask = np.full((64, 64), 0.51, dtype=np.float32)
        result = clean_matte(mask, area_threshold=100)
        assert result[32, 32] > 0.3


# ═══════════════════════════════════════════════════════════════════════
# Singleton-channel [H,W,1] inputs through pipelines
# ═══════════════════════════════════════════════════════════════════════


class TestSingletonChannelIntegration:
    """Test [H,W,1] mask inputs through the full pipelines."""

    def test_preprocess_with_3d_mask(self):
        """preprocess should handle [H,W,1] masks."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.rand(32, 32, 1).astype(np.float32)
        result = preprocess(img, mask, model_size=16)
        assert result.shape == (16, 16, 4)

    def test_postprocess_with_3d_alpha(self):
        """postprocess should handle [H,W,1] alpha."""
        alpha = np.random.rand(16, 16, 1).astype(np.float32)
        fg = np.random.rand(16, 16, 3).astype(np.float32)
        result = postprocess(alpha, fg, 32, 32, auto_despeckle=False)
        assert result["alpha"].shape == (32, 32)
        assert result["processed"].shape == (32, 32, 4)

    def test_full_pipeline_with_3d_mask(self):
        """End-to-end pipeline with [H,W,1] mask should not crash."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.random.rand(64, 64, 1).astype(np.float32)
        preprocessed = preprocess(img, mask, model_size=16)
        # Mock model output
        alpha_raw = preprocessed[:, :, 3]
        fg_raw = np.full((16, 16, 3), 0.5, dtype=np.float32)
        result = postprocess(alpha_raw, fg_raw, 64, 64, auto_despeckle=False)
        assert np.all(np.isfinite(result["processed"]))
