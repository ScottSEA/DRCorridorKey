"""Tests for resolve_plugin.models — Pydantic request/response schemas."""

from __future__ import annotations

import os
import sys

import pytest
from pydantic import ValidationError

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from resolve_plugin.models import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    InferResponse,
    ModelState,
)


class TestInferRequest:
    """Test InferRequest validation."""

    def test_required_fields(self):
        """image_path and alpha_hint_path are required."""
        req = InferRequest(
            image_path="/tmp/input.exr",
            alpha_hint_path="/tmp/hint.exr",
        )
        assert req.image_path == "/tmp/input.exr"
        assert req.alpha_hint_path == "/tmp/hint.exr"

    def test_optional_defaults_to_none(self):
        """Optional inference params should default to None (use service defaults)."""
        req = InferRequest(
            image_path="/tmp/input.exr",
            alpha_hint_path="/tmp/hint.exr",
        )
        assert req.despill_strength is None
        assert req.auto_despeckle is None
        assert req.despeckle_size is None
        assert req.refiner_scale is None
        assert req.input_is_linear is None
        assert req.output_dir is None

    def test_despill_range_validation(self):
        """despill_strength must be in [0, 1]."""
        InferRequest(image_path="/a", alpha_hint_path="/b", despill_strength=0.5)
        with pytest.raises(ValidationError):
            InferRequest(image_path="/a", alpha_hint_path="/b", despill_strength=1.5)
        with pytest.raises(ValidationError):
            InferRequest(image_path="/a", alpha_hint_path="/b", despill_strength=-0.1)

    def test_despeckle_size_non_negative(self):
        """despeckle_size must be >= 0."""
        InferRequest(image_path="/a", alpha_hint_path="/b", despeckle_size=0)
        with pytest.raises(ValidationError):
            InferRequest(image_path="/a", alpha_hint_path="/b", despeckle_size=-1)


class TestInferResponse:
    """Test InferResponse serialisation."""

    def test_all_fields(self):
        """Response should include all path fields."""
        resp = InferResponse(
            fg_path="/out/fg.exr",
            alpha_path="/out/alpha.exr",
            fg_ppm_path="/out/fg.ppm",
            alpha_pgm_path="/out/alpha.pgm",
            comp_path="/out/comp.png",
        )
        assert resp.fg_path == "/out/fg.exr"
        assert resp.comp_path == "/out/comp.png"

    def test_comp_path_optional(self):
        """comp_path should be optional (None by default)."""
        resp = InferResponse(
            fg_path="/out/fg.exr",
            alpha_path="/out/alpha.exr",
            fg_ppm_path="/out/fg.ppm",
            alpha_pgm_path="/out/alpha.pgm",
        )
        assert resp.comp_path is None


class TestModelState:
    """Test ModelState enum values."""

    def test_all_states_exist(self):
        """All expected lifecycle states should be defined."""
        assert ModelState.COLD == "cold"
        assert ModelState.LOADING == "loading"
        assert ModelState.READY == "ready"
        assert ModelState.FAILED == "failed"


class TestHealthResponse:
    """Test HealthResponse construction."""

    def test_basic_construction(self):
        resp = HealthResponse(
            status="ok",
            model_state=ModelState.COLD,
            device="cpu",
            checkpoint_found=False,
        )
        assert resp.model_state == ModelState.COLD
        assert resp.vram is None


class TestErrorResponse:
    """Test ErrorResponse construction."""

    def test_detail_field(self):
        err = ErrorResponse(detail="something went wrong")
        assert err.detail == "something went wrong"
