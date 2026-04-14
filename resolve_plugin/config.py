"""Service configuration — single source of truth for all tunables.

Uses Pydantic BaseSettings so every value can be overridden via environment
variable (prefixed ``CK_``) or a ``.env`` file.  Reasonable defaults are
chosen so the service works out of the box with zero configuration.

Example:
    CK_PORT=8080 CK_DEVICE=cpu  uv run python -m resolve_plugin.server
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ServiceSettings(BaseSettings):
    """All configurable knobs for the CorridorKey Resolve service."""

    # ── Network ──────────────────────────────────────────────────────────
    # Bind to localhost only — the service handles arbitrary file paths,
    # so it MUST NOT be reachable from the network.  See PluginPlan.md
    # "Path security" note.
    host: str = Field(
        default="127.0.0.1",
        description="IP address to bind to.  Keep 127.0.0.1 for security.",
    )
    port: int = Field(
        default=5309,
        ge=1024,
        le=65535,
        description="TCP port for the HTTP service.",
    )

    # ── Compute ──────────────────────────────────────────────────────────
    device: str = Field(
        default="auto",
        description=(
            "Compute device: 'auto' (detect best), 'cuda', 'mps', or 'cpu'."
        ),
    )
    preload_model: bool = Field(
        default=False,
        description=(
            "If True, load the inference model at startup instead of on the "
            "first /infer request.  Avoids a slow first frame but adds "
            "startup latency."
        ),
    )

    # ── Paths ────────────────────────────────────────────────────────────
    temp_dir: str = Field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "corridorkey_resolve"),
        description="Root directory for temporary frame I/O between the Fuse and the service.",
    )
    allowed_roots: list[str] = Field(
        default_factory=list,
        description=(
            "Optional allowlist of directory roots.  When non-empty, the "
            "service rejects any file path that does not start with one of "
            "these roots (after normalization).  An empty list disables the "
            "check — acceptable when the service is localhost-only."
        ),
    )

    # ── Inference defaults ───────────────────────────────────────────────
    # These are used when the /infer request omits a parameter.
    default_despill_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    default_auto_despeckle: bool = True
    default_despeckle_size: int = Field(default=400, ge=0)
    default_refiner_scale: float = Field(default=1.0, ge=0.0)
    default_input_is_linear: bool = False

    # ── Pydantic configuration ───────────────────────────────────────────
    model_config = {
        "env_prefix": "CK_",          # e.g. CK_PORT=8080
        "env_file": ".env",           # optional .env in cwd
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "mps", "cpu"}
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v_lower


@lru_cache(maxsize=1)
def get_settings() -> ServiceSettings:
    """Return the cached singleton settings instance.

    Using ``lru_cache`` ensures settings are parsed once and shared across
    the application.  Override in tests by clearing the cache::

        get_settings.cache_clear()
    """
    return ServiceSettings()
