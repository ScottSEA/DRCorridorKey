"""TDD tests for alpha compositing operations.

Tests premultiply, unpremultiply, composite_straight, and
composite_premul — the fundamental compositing math that the
C++ port must match exactly.
"""

from __future__ import annotations

import numpy as np

from ofx_plugin.core.composite import (
    composite_premul,
    composite_straight,
    premultiply,
    unpremultiply,
)


class TestPremultiply:
    """Test foreground × alpha premultiplication."""

    def test_opaque(self):
        """Alpha = 1.0 → foreground unchanged."""
        fg = np.array([[[0.5, 0.3, 0.8]]], dtype=np.float32)
        alpha = np.array([[[1.0]]], dtype=np.float32)
        result = premultiply(fg, alpha)
        np.testing.assert_allclose(result, fg, atol=1e-7)

    def test_transparent(self):
        """Alpha = 0.0 → result is black."""
        fg = np.array([[[0.5, 0.3, 0.8]]], dtype=np.float32)
        alpha = np.array([[[0.0]]], dtype=np.float32)
        result = premultiply(fg, alpha)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_half_alpha(self):
        """Alpha = 0.5 → result is half the foreground."""
        fg = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        alpha = np.array([[[0.5]]], dtype=np.float32)
        result = premultiply(fg, alpha)
        np.testing.assert_allclose(result, 0.5, atol=1e-7)


class TestUnpremultiply:
    """Test foreground / alpha unpremultiplication."""

    def test_round_trip(self):
        """premultiply → unpremultiply should recover the original."""
        fg = np.array([[[0.5, 0.3, 0.8]]], dtype=np.float32)
        alpha = np.array([[[0.7]]], dtype=np.float32)
        premul = premultiply(fg, alpha)
        recovered = unpremultiply(premul, alpha)
        np.testing.assert_allclose(recovered, fg, atol=1e-5)

    def test_zero_alpha_safe(self):
        """Alpha = 0 should not produce inf/nan (eps prevents division by zero)."""
        premul = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        alpha = np.array([[[0.0]]], dtype=np.float32)
        result = unpremultiply(premul, alpha)
        assert np.all(np.isfinite(result))


class TestCompositeStraight:
    """Test straight-alpha (over) compositing: FG × α + BG × (1 − α)."""

    def test_opaque_fg_covers_bg(self):
        """Alpha = 1 → only foreground visible."""
        fg = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        bg = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        alpha = np.array([[[1.0]]], dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        np.testing.assert_allclose(result, fg, atol=1e-7)

    def test_transparent_fg_shows_bg(self):
        """Alpha = 0 → only background visible."""
        fg = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        bg = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        alpha = np.array([[[0.0]]], dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        np.testing.assert_allclose(result, bg, atol=1e-7)

    def test_half_alpha_blends(self):
        """Alpha = 0.5 → equal mix of FG and BG."""
        fg = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        bg = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        alpha = np.array([[[0.5]]], dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        np.testing.assert_allclose(result, 0.5, atol=1e-7)


class TestCompositePremul:
    """Test premultiplied-alpha compositing: FG + BG × (1 − α)."""

    def test_opaque_fg_covers_bg(self):
        """Alpha = 1 → only premul foreground visible."""
        fg_premul = np.array([[[0.5, 0.3, 0.8]]], dtype=np.float32)
        bg = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        alpha = np.array([[[1.0]]], dtype=np.float32)
        result = composite_premul(fg_premul, bg, alpha)
        np.testing.assert_allclose(result, fg_premul, atol=1e-7)

    def test_transparent_fg_shows_bg(self):
        """Alpha = 0, premul FG = black → only background visible."""
        fg_premul = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        bg = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        alpha = np.array([[[0.0]]], dtype=np.float32)
        result = composite_premul(fg_premul, bg, alpha)
        np.testing.assert_allclose(result, bg, atol=1e-7)
