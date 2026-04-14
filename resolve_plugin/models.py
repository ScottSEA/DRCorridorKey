"""Pydantic models for the Resolve service HTTP API.

Keeps request validation, response serialization, and domain types in one
place — separate from route handlers (SRP) and reusable in tests.

Every field has a docstring-level ``description`` so FastAPI auto-generates
useful OpenAPI docs.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Inference Request ────────────────────────────────────────────────────


class InferRequest(BaseModel):
    """POST /infer — request body.

    The caller provides file paths (both sides are on the same machine)
    rather than uploading bytes, which avoids unnecessary serialisation
    of large EXR frames over the loopback.
    """

    image_path: str = Field(
        ...,
        description="Absolute path to the input green-screen frame (EXR, PNG, etc.).",
    )
    alpha_hint_path: str = Field(
        ...,
        description="Absolute path to the alpha-hint / rough matte image.",
    )

    # ── Optional inference parameters (override service defaults) ────────
    despill_strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Despill multiplier.  0 = no despill, 1 = full.",
    )
    auto_despeckle: Optional[bool] = Field(
        default=None,
        description="Clean up small disconnected alpha islands automatically.",
    )
    despeckle_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum pixel-count to keep an alpha island.",
    )
    refiner_scale: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Refiner network delta multiplier.",
    )
    input_is_linear: Optional[bool] = Field(
        default=None,
        description="True if the input image is in linear colour space.",
    )

    # ── Output control ───────────────────────────────────────────────────
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory to write output files into.  Defaults to the "
            "service's temp directory."
        ),
    )


# ── Inference Response ───────────────────────────────────────────────────


class InferResponse(BaseModel):
    """POST /infer — response body.

    Returns paths to the output files so the Fuse can read them back.
    Full-precision EXR files are always written for final compositing.
    PPM/PGM files (8-bit) are also written for easy parsing in the
    Lua-based Fusion Fuse.
    """

    fg_path: str = Field(
        ...,
        description="Path to the foreground EXR (RGB, straight/linear).",
    )
    alpha_path: str = Field(
        ...,
        description="Path to the alpha matte EXR (single-channel).",
    )
    fg_ppm_path: str = Field(
        ...,
        description="Path to the foreground PPM (8-bit RGB, for the Fuse).",
    )
    alpha_pgm_path: str = Field(
        ...,
        description="Path to the alpha PGM (8-bit grayscale, for the Fuse).",
    )
    comp_path: Optional[str] = Field(
        default=None,
        description="Path to the checker-composite PNG (optional, for preview).",
    )


# ── Health / Status ──────────────────────────────────────────────────────


class ModelState(str, Enum):
    """Lifecycle states for the inference model."""

    COLD = "cold"          # Not loaded, never loaded
    LOADING = "loading"    # Currently loading (checkpoint download / compile)
    READY = "ready"        # Loaded and ready to infer
    FAILED = "failed"      # Load attempted but failed


class HealthResponse(BaseModel):
    """GET /health — service health check."""

    status: str = Field(
        default="ok",
        description="Overall service status.",
    )
    model_state: ModelState = Field(
        ...,
        description="Current lifecycle state of the inference model.",
    )
    device: str = Field(
        ...,
        description="Active compute device (cuda, mps, cpu).",
    )
    vram: Optional[dict] = Field(
        default=None,
        description="GPU VRAM info in GB (total, reserved, allocated, free, name).  Null on CPU/MPS.",
    )
    checkpoint_found: bool = Field(
        ...,
        description="Whether a model checkpoint file was found on disk.",
    )


# ── Error ────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error envelope returned for 4xx / 5xx responses."""

    detail: str = Field(
        ...,
        description="Human-readable error description.",
    )
