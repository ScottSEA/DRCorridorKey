"""Tests for resolve_plugin.cleanup — temp directory garbage collection."""

from __future__ import annotations

import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from resolve_plugin.cleanup import (
    _is_request_dir,
    purge_old_outputs,
)


class TestIsRequestDir:
    """Test the request directory name pattern matcher."""

    def test_valid_hex_12_chars(self):
        """12-char hex strings should be recognised as request dirs."""
        assert _is_request_dir("a1b2c3d4e5f6") is True
        assert _is_request_dir("000000000000") is True
        assert _is_request_dir("abcdef123456") is True

    def test_wrong_length(self):
        """Strings not exactly 12 chars should be rejected."""
        assert _is_request_dir("a1b2c3") is False  # too short
        assert _is_request_dir("a1b2c3d4e5f6a7") is False  # too long
        assert _is_request_dir("") is False

    def test_non_hex_chars(self):
        """Strings with non-hex characters should be rejected."""
        assert _is_request_dir("a1b2c3d4e5gz") is False
        assert _is_request_dir("hello_world!") is False

    def test_real_directory_names(self):
        """Common directory names should NOT match."""
        assert _is_request_dir("__pycache__") is False
        assert _is_request_dir("Fuses") is False
        assert _is_request_dir(".git") is False


class TestPurgeOldOutputs:
    """Test age-based directory cleanup."""

    def test_removes_old_dirs(self, tmp_path):
        """Directories older than max_age_hours should be removed."""
        # Create a 12-char hex directory and backdate its mtime
        old_dir = tmp_path / "a1b2c3d4e5f6"
        old_dir.mkdir()
        (old_dir / "fg.exr").write_text("fake")
        # Set mtime to 48 hours ago
        old_time = time.time() - 48 * 3600
        os.utime(str(old_dir), (old_time, old_time))

        removed = purge_old_outputs(str(tmp_path), max_age_hours=24.0)
        assert removed == 1
        assert not old_dir.exists()

    def test_keeps_recent_dirs(self, tmp_path):
        """Directories younger than max_age_hours should be kept."""
        recent_dir = tmp_path / "b2c3d4e5f6a1"
        recent_dir.mkdir()
        (recent_dir / "fg.exr").write_text("fake")

        removed = purge_old_outputs(str(tmp_path), max_age_hours=24.0)
        assert removed == 0
        assert recent_dir.exists()

    def test_ignores_non_request_dirs(self, tmp_path):
        """Directories that don't match the naming pattern should be left alone."""
        other_dir = tmp_path / "my_data"
        other_dir.mkdir()
        old_time = time.time() - 48 * 3600
        os.utime(str(other_dir), (old_time, old_time))

        removed = purge_old_outputs(str(tmp_path), max_age_hours=1.0)
        assert removed == 0
        assert other_dir.exists()

    def test_ignores_files(self, tmp_path):
        """Regular files (not directories) should be left alone."""
        # A file with a 12-char hex name
        f = tmp_path / "a1b2c3d4e5f6"
        f.write_text("I'm a file, not a dir")
        old_time = time.time() - 48 * 3600
        os.utime(str(f), (old_time, old_time))

        removed = purge_old_outputs(str(tmp_path), max_age_hours=1.0)
        assert removed == 0
        assert f.exists()

    def test_nonexistent_dir(self):
        """Purging a non-existent directory should return 0, not crash."""
        removed = purge_old_outputs("/nonexistent/path/12345", max_age_hours=1.0)
        assert removed == 0

    def test_mixed_old_and_new(self, tmp_path):
        """Only old directories should be removed; new ones kept."""
        old_dir = tmp_path / "aaaaaaaaaaaa"
        old_dir.mkdir()
        old_time = time.time() - 48 * 3600
        os.utime(str(old_dir), (old_time, old_time))

        new_dir = tmp_path / "bbbbbbbbbbbb"
        new_dir.mkdir()

        removed = purge_old_outputs(str(tmp_path), max_age_hours=24.0)
        assert removed == 1
        assert not old_dir.exists()
        assert new_dir.exists()
