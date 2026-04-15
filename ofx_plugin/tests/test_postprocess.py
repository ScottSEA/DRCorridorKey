"""TDD tests for the full postprocessing pipeline.

After model inference produces raw alpha [H,W] and fg [H,W,3] at
model resolution, postprocessing transforms them back to the
original resolution with the correct color space and compositing:

    1. Resize alpha + fg back to original dimensions
    2. Clean matte (despeckle small islands)
    3. Despill green from foreground
    4. Convert fg from sRGB to linear
    5. Premultiply fg by alpha
    6. Pack RGBA
    7. Optionally generate a checker composite preview
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.postprocess import postprocess


class TestPostprocess:
    """Test the complete postprocessing pipeline."""

    def test_output_keys(self):
        """Result dict must contain 'alpha', 'fg', and 'processed'."""
        alpha = np.random.rand(32, 32).astype(np.float32)
        fg = np.random.rand(32, 32, 3).astype(np.float32)
        result = postprocess(alpha, fg, original_height=64, original_width=128)
        assert "alpha" in result
        assert "fg" in result
        assert "processed" in result

    def test_output_shapes(self):
        """Outputs must be resized to original dimensions."""
        alpha = np.random.rand(32, 32).astype(np.float32)
        fg = np.random.rand(32, 32, 3).astype(np.float32)
        result = postprocess(alpha, fg, original_height=100, original_width=200)
        assert result["alpha"].shape == (100, 200)
        assert result["fg"].shape == (100, 200, 3)
        assert result["processed"].shape == (100, 200, 4)  # RGBA

    def test_output_dtypes(self):
        """All outputs must be float32."""
        alpha = np.random.rand(16, 16).astype(np.float32)
        fg = np.random.rand(16, 16, 3).astype(np.float32)
        result = postprocess(alpha, fg, original_height=32, original_width=32)
        assert result["alpha"].dtype == np.float32
        assert result["fg"].dtype == np.float32
        assert result["processed"].dtype == np.float32

    def test_processed_has_four_channels(self):
        """The 'processed' output must be RGBA (4 channels)."""
        alpha = np.ones((16, 16), dtype=np.float32)
        fg = np.ones((16, 16, 3), dtype=np.float32) * 0.5
        result = postprocess(alpha, fg, original_height=32, original_width=32)
        assert result["processed"].shape[2] == 4

    def test_processed_alpha_channel(self):
        """The alpha channel of 'processed' should match the alpha output."""
        alpha = np.ones((16, 16), dtype=np.float32) * 0.7
        fg = np.ones((16, 16, 3), dtype=np.float32) * 0.5
        result = postprocess(
            alpha, fg, original_height=16, original_width=16,
            auto_despeckle=False,
        )
        # Alpha channel (ch 3) of processed should ≈ the alpha output
        np.testing.assert_allclose(
            result["processed"][:, :, 3], result["alpha"], atol=0.05,
        )

    def test_comp_generated_when_requested(self):
        """When generate_comp=True, 'comp' should be in the result."""
        alpha = np.random.rand(16, 16).astype(np.float32)
        fg = np.random.rand(16, 16, 3).astype(np.float32)
        result = postprocess(
            alpha, fg, original_height=32, original_width=32,
            generate_comp=True,
        )
        assert "comp" in result
        assert result["comp"].shape == (32, 32, 3)

    def test_comp_not_generated_when_disabled(self):
        """When generate_comp=False, 'comp' should not be in the result."""
        alpha = np.random.rand(16, 16).astype(np.float32)
        fg = np.random.rand(16, 16, 3).astype(np.float32)
        result = postprocess(
            alpha, fg, original_height=32, original_width=32,
            generate_comp=False,
        )
        assert "comp" not in result

    def test_despill_strength_zero(self):
        """despill_strength=0 should skip despill entirely."""
        alpha = np.ones((16, 16), dtype=np.float32)
        # Pure green pixel — no despill should leave green intact
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        fg[:, :, 1] = 1.0  # all green
        result = postprocess(
            alpha, fg, original_height=16, original_width=16,
            despill_strength=0.0, auto_despeckle=False,
        )
        # The green channel of fg output should still be dominant
        assert result["fg"][:, :, 1].mean() > result["fg"][:, :, 0].mean()

    def test_transparent_alpha_zeroes_processed(self):
        """Alpha = 0 everywhere → processed RGB should be near-zero (premul)."""
        alpha = np.zeros((16, 16), dtype=np.float32)
        fg = np.ones((16, 16, 3), dtype=np.float32)
        result = postprocess(
            alpha, fg, original_height=16, original_width=16,
            auto_despeckle=False,
        )
        # Premultiplied: fg * alpha = 0
        np.testing.assert_allclose(
            result["processed"][:, :, :3], 0.0, atol=0.01,
        )
