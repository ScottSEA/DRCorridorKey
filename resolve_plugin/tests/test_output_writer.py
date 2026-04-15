"""Tests for PPM/PGM encoding and atomic file writes.

These tests import the real resolve_plugin.ppm module, which has
no torch/cv2 dependencies.  This ensures we're testing the actual
production code, not a copy.
"""

from __future__ import annotations

import os

import numpy as np

from resolve_plugin.ppm import array_to_pgm_gray, array_to_ppm_rgb, atomic_write_bytes


class TestPPMEncoding:
    """Test PPM P6 (RGB) binary encoding."""

    def test_header_format(self):
        """PPM header should follow the P6 spec exactly."""
        img = np.zeros((2, 3, 3), dtype=np.float32)
        data = array_to_ppm_rgb(img)
        # Header: "P6\n3 2\n255\n" then 2*3*3 = 18 bytes of pixel data
        header = data[: len(data) - 2 * 3 * 3]
        assert header == b"P6\n3 2\n255\n"

    def test_pixel_values_black(self):
        """All-zero image should produce all-zero pixel bytes."""
        img = np.zeros((1, 1, 3), dtype=np.float32)
        data = array_to_ppm_rgb(img)
        pixels = data[data.index(b"\n255\n") + 5 :]
        assert pixels == b"\x00\x00\x00"

    def test_pixel_values_white(self):
        """All-one image should produce all-255 pixel bytes."""
        img = np.ones((1, 1, 3), dtype=np.float32)
        data = array_to_ppm_rgb(img)
        pixels = data[data.index(b"\n255\n") + 5 :]
        assert pixels == b"\xff\xff\xff"

    def test_pixel_values_mid(self):
        """0.5 should round to 128."""
        img = np.full((1, 1, 3), 0.5, dtype=np.float32)
        data = array_to_ppm_rgb(img)
        pixels = data[data.index(b"\n255\n") + 5 :]
        assert pixels == bytes([128, 128, 128])

    def test_clamping_negative(self):
        """Negative values should be clamped to 0."""
        img = np.full((1, 1, 3), -1.0, dtype=np.float32)
        data = array_to_ppm_rgb(img)
        pixels = data[data.index(b"\n255\n") + 5 :]
        assert pixels == b"\x00\x00\x00"

    def test_clamping_above_one(self):
        """Values > 1.0 should be clamped to 255."""
        img = np.full((1, 1, 3), 2.5, dtype=np.float32)
        data = array_to_ppm_rgb(img)
        pixels = data[data.index(b"\n255\n") + 5 :]
        assert pixels == b"\xff\xff\xff"

    def test_dimensions_in_header(self):
        """Header should reflect the actual image dimensions."""
        img = np.zeros((100, 200, 3), dtype=np.float32)
        data = array_to_ppm_rgb(img)
        header_end = data.index(b"\n255\n") + 5
        header = data[:header_end].decode("ascii")
        assert "200 100" in header  # width height

    def test_total_size(self):
        """Total byte count should be header + width * height * 3."""
        h, w = 10, 15
        img = np.zeros((h, w, 3), dtype=np.float32)
        data = array_to_ppm_rgb(img)
        header_end = data.index(b"\n255\n") + 5
        pixel_bytes = len(data) - header_end
        assert pixel_bytes == h * w * 3


class TestPGMEncoding:
    """Test PGM P5 (grayscale) binary encoding."""

    def test_header_format(self):
        """PGM header should follow the P5 spec exactly."""
        img = np.zeros((2, 3), dtype=np.float32)
        data = array_to_pgm_gray(img)
        header = data[: len(data) - 2 * 3]
        assert header == b"P5\n3 2\n255\n"

    def test_pixel_values_white(self):
        """All-one mask should produce all-255 bytes."""
        img = np.ones((1, 1), dtype=np.float32)
        data = array_to_pgm_gray(img)
        pixels = data[data.index(b"\n255\n") + 5 :]
        assert pixels == b"\xff"

    def test_pixel_values_black(self):
        """All-zero mask should produce all-zero bytes."""
        img = np.zeros((1, 1), dtype=np.float32)
        data = array_to_pgm_gray(img)
        pixels = data[data.index(b"\n255\n") + 5 :]
        assert pixels == b"\x00"

    def test_total_size(self):
        """Total byte count should be header + width * height."""
        h, w = 8, 12
        img = np.zeros((h, w), dtype=np.float32)
        data = array_to_pgm_gray(img)
        header_end = data.index(b"\n255\n") + 5
        pixel_bytes = len(data) - header_end
        assert pixel_bytes == h * w


class TestAtomicWriteBytes:
    """Test atomic file writing (tmp + rename)."""

    def test_writes_file(self, tmp_path):
        """File should exist with correct content after write."""
        dest = str(tmp_path / "test.ppm")
        atomic_write_bytes(b"hello world", dest)
        assert os.path.isfile(dest)
        with open(dest, "rb") as f:
            assert f.read() == b"hello world"

    def test_no_tmp_files_left(self, tmp_path):
        """No .tmp files should remain after a successful write."""
        dest = str(tmp_path / "test.ppm")
        atomic_write_bytes(b"data", dest)
        remaining = [f for f in os.listdir(tmp_path) if ".tmp" in f]
        assert remaining == []

    def test_overwrites_existing(self, tmp_path):
        """Writing to an existing path should replace the content."""
        dest = str(tmp_path / "test.ppm")
        atomic_write_bytes(b"first", dest)
        atomic_write_bytes(b"second", dest)
        with open(dest, "rb") as f:
            assert f.read() == b"second"
