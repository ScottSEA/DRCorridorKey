"""Tests for resolve_plugin.config — ServiceSettings and get_settings."""

from __future__ import annotations

import os

# Ensure project root is on sys.path so resolve_plugin can be imported
import sys

import pytest
from pydantic import ValidationError

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from resolve_plugin.config import ServiceSettings, get_settings


class TestServiceSettings:
    """Test default values and validation of ServiceSettings."""

    def test_defaults(self):
        """Settings should have sensible defaults without any env vars."""
        s = ServiceSettings()
        assert s.host == "127.0.0.1"
        assert s.port == 5309
        assert s.device == "auto"
        assert s.preload_model is False
        assert s.default_input_is_linear is True
        assert s.default_despill_strength == 1.0
        assert s.default_auto_despeckle is True
        assert s.default_despeckle_size == 400
        assert s.default_refiner_scale == 1.0
        assert s.cleanup_max_age_hours == 24.0

    def test_device_validation_accepts_valid(self):
        """All valid device strings should be accepted (case-insensitive)."""
        for dev in ("auto", "cuda", "mps", "cpu", "AUTO", "Cuda", "CPU"):
            s = ServiceSettings(device=dev)
            assert s.device == dev.lower().strip()

    def test_device_validation_rejects_invalid(self):
        """Invalid device strings should raise a validation error."""
        with pytest.raises(ValidationError):  # Pydantic ValidationError
            ServiceSettings(device="vulkan")

    def test_port_range_validation(self):
        """Port must be in [1024, 65535]."""
        ServiceSettings(port=1024)  # min
        ServiceSettings(port=65535)  # max
        with pytest.raises(ValidationError):
            ServiceSettings(port=80)  # below min
        with pytest.raises(ValidationError):
            ServiceSettings(port=99999)  # above max

    def test_env_override(self, monkeypatch):
        """CK_-prefixed env vars should override defaults."""
        monkeypatch.setenv("CK_PORT", "8080")
        monkeypatch.setenv("CK_DEVICE", "cpu")
        monkeypatch.setenv("CK_PRELOAD_MODEL", "true")
        s = ServiceSettings()
        assert s.port == 8080
        assert s.device == "cpu"
        assert s.preload_model is True

    def test_allowed_roots_defaults_to_empty(self):
        """allowed_roots should default to an empty list (no restriction)."""
        s = ServiceSettings()
        assert s.allowed_roots == []

    def test_temp_dir_is_set(self):
        """temp_dir should point to a corridorkey_resolve subdirectory."""
        s = ServiceSettings()
        assert "corridorkey_resolve" in s.temp_dir


class TestGetSettings:
    """Test the cached settings singleton."""

    def test_returns_same_instance(self):
        """get_settings() should return the same object on repeated calls."""
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_resets(self, monkeypatch):
        """Clearing the cache should allow new env vars to take effect."""
        get_settings.cache_clear()
        monkeypatch.setenv("CK_PORT", "9999")
        get_settings.cache_clear()
        s = get_settings()
        assert s.port == 9999
        get_settings.cache_clear()  # clean up
