"""Tests for resolve_plugin.path_security — path validation and safety."""

from __future__ import annotations

import os
import sys

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from resolve_plugin.config import ServiceSettings
from resolve_plugin.path_security import validate_input_path, validate_output_dir


class TestValidateInputPath:
    """Test input path normalisation, allowlist, and existence checks."""

    def test_accepts_existing_file(self, tmp_path):
        """An existing file with no allowlist should pass."""
        f = tmp_path / "test.exr"
        f.write_text("fake")
        settings = ServiceSettings(allowed_roots=[])
        result = validate_input_path(str(f), settings)
        assert os.path.isabs(result)
        assert os.path.isfile(result)

    def test_rejects_nonexistent_file(self, tmp_path):
        """A path to a non-existent file should raise ValueError."""
        settings = ServiceSettings(allowed_roots=[])
        with pytest.raises(ValueError, match="does not exist"):
            validate_input_path(str(tmp_path / "ghost.exr"), settings)

    def test_allowlist_permits_inside(self, tmp_path):
        """Files inside an allowed root should pass."""
        f = tmp_path / "data" / "image.exr"
        f.parent.mkdir()
        f.write_text("fake")
        settings = ServiceSettings(allowed_roots=[str(tmp_path)])
        result = validate_input_path(str(f), settings)
        assert result.endswith("image.exr")

    def test_allowlist_rejects_outside(self, tmp_path):
        """Files outside all allowed roots should be rejected."""
        f = tmp_path / "image.exr"
        f.write_text("fake")
        settings = ServiceSettings(allowed_roots=["C:\\nonexistent_root"])
        with pytest.raises(ValueError, match="outside the allowed roots"):
            validate_input_path(str(f), settings)

    def test_allowlist_rejects_sibling_prefix(self, tmp_path):
        """A sibling directory sharing a prefix must NOT pass the allowlist.

        If allowed root is /data/job1, then /data/job1_backup/frame.exr
        must be rejected — it's a different directory, not a child.
        """
        allowed = tmp_path / "job1"
        allowed.mkdir()
        sibling = tmp_path / "job1_backup"
        sibling.mkdir()
        f = sibling / "frame.exr"
        f.write_text("fake")
        settings = ServiceSettings(allowed_roots=[str(allowed)])
        with pytest.raises(ValueError, match="outside the allowed roots"):
            validate_input_path(str(f), settings)

    def test_normalises_path(self, tmp_path):
        """Paths with .. and redundant separators should be normalised."""
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        f = subdir / "test.exr"
        f.write_text("fake")
        # Feed a path with .. in it
        messy = str(tmp_path / "a" / "b" / ".." / "b" / "test.exr")
        settings = ServiceSettings(allowed_roots=[])
        result = validate_input_path(messy, settings)
        assert ".." not in result


class TestValidateOutputDir:
    """Test output directory validation and creation."""

    def test_creates_directory(self, tmp_path):
        """validate_output_dir should create the directory if missing."""
        new_dir = str(tmp_path / "output" / "subdir")
        settings = ServiceSettings(allowed_roots=[])
        result = validate_output_dir(new_dir, settings)
        assert os.path.isdir(result)

    def test_existing_directory_ok(self, tmp_path):
        """An existing directory should be accepted without error."""
        settings = ServiceSettings(allowed_roots=[])
        result = validate_output_dir(str(tmp_path), settings)
        assert os.path.isdir(result)

    def test_allowlist_rejects_outside(self, tmp_path):
        """Output directories outside allowed roots should be rejected."""
        settings = ServiceSettings(allowed_roots=["C:\\nonexistent_root"])
        with pytest.raises(ValueError, match="outside the allowed roots"):
            validate_output_dir(str(tmp_path / "out"), settings)
