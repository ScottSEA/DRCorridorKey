"""TDD tests for the end-to-end pipeline (pre → model → post).

Uses a mock model (identity-like) to verify that the full pipeline
produces correctly shaped, typed, and valued outputs.  This test
doesn't validate model accuracy — it validates the plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofx_plugin.core.preprocess import preprocess
from ofx_plugin.core.postprocess import postprocess


def _mock_model(preprocessed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fake model that returns plausible alpha and fg from preprocessed input.

    Takes [H, W, 4] preprocessed input and returns:
        alpha: [H, W] — sigmoid of the mask channel (ch 3)
        fg: [H, W, 3] — clamped version of the first 3 channels

    This doesn't do real inference — it just exercises the pipeline
    with deterministic, shape-correct outputs.
    """
    h, w = preprocessed.shape[:2]
    # Use the mask channel as alpha (it's already in [0, 1])
    alpha = preprocessed[:, :, 3]
    # Use a constant mid-gray as fg (in sRGB)
    fg = np.full((h, w, 3), 0.5, dtype=np.float32)
    return alpha, fg


class TestEndToEnd:
    """Test the full pre → mock model → post pipeline."""

    def test_pipeline_produces_valid_output(self):
        """The full pipeline should produce correctly shaped results."""
        # Simulate a 200×300 green screen frame
        rng = np.random.default_rng(42)
        image = rng.random((200, 300, 3), dtype=np.float32)
        mask = rng.random((200, 300), dtype=np.float32)

        # Preprocess
        model_input = preprocess(image, mask, model_size=64)
        assert model_input.shape == (64, 64, 4)

        # Mock inference
        alpha_raw, fg_raw = _mock_model(model_input)
        assert alpha_raw.shape == (64, 64)
        assert fg_raw.shape == (64, 64, 3)

        # Postprocess
        result = postprocess(
            alpha_raw, fg_raw,
            original_height=200, original_width=300,
            generate_comp=True,
        )

        # Validate output shapes match the original frame
        assert result["alpha"].shape == (200, 300)
        assert result["fg"].shape == (200, 300, 3)
        assert result["processed"].shape == (200, 300, 4)
        assert result["comp"].shape == (200, 300, 3)

    def test_pipeline_dtypes(self):
        """All pipeline outputs must be float32."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.random.rand(64, 64).astype(np.float32)

        model_input = preprocess(image, mask, model_size=32)
        alpha_raw, fg_raw = _mock_model(model_input)
        result = postprocess(
            alpha_raw, fg_raw,
            original_height=64, original_width=64,
        )

        assert result["alpha"].dtype == np.float32
        assert result["fg"].dtype == np.float32
        assert result["processed"].dtype == np.float32

    def test_pipeline_values_finite(self):
        """No NaN or Inf values should appear in any output."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.random.rand(64, 64).astype(np.float32)

        model_input = preprocess(image, mask, model_size=32)
        alpha_raw, fg_raw = _mock_model(model_input)
        result = postprocess(
            alpha_raw, fg_raw,
            original_height=64, original_width=64,
        )

        assert np.all(np.isfinite(result["alpha"]))
        assert np.all(np.isfinite(result["fg"]))
        assert np.all(np.isfinite(result["processed"]))

    def test_different_input_output_sizes(self):
        """Pipeline should handle any input size and resize correctly."""
        # Odd-sized input
        image = np.random.rand(137, 251, 3).astype(np.float32)
        mask = np.random.rand(137, 251).astype(np.float32)

        model_input = preprocess(image, mask, model_size=64)
        alpha_raw, fg_raw = _mock_model(model_input)
        result = postprocess(
            alpha_raw, fg_raw,
            original_height=137, original_width=251,
            auto_despeckle=False,
        )

        assert result["alpha"].shape == (137, 251)
        assert result["fg"].shape == (137, 251, 3)

    def test_linear_input_roundtrip(self):
        """Linear input should be handled without errors."""
        image = np.random.rand(64, 64, 3).astype(np.float32) * 0.5
        mask = np.ones((64, 64), dtype=np.float32)

        model_input = preprocess(
            image, mask, model_size=32, input_is_linear=True,
        )
        alpha_raw, fg_raw = _mock_model(model_input)
        result = postprocess(
            alpha_raw, fg_raw,
            original_height=64, original_width=64,
            auto_despeckle=False,
        )

        assert np.all(np.isfinite(result["processed"]))
