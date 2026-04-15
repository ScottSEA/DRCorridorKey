"""TDD tests for TorchScript model export.

The OFX plugin needs the CorridorKey model exported as a TorchScript
file (.pt) that can be loaded by libtorch in C++.  These tests verify
the export process and validate that the exported model produces
identical outputs to the original.

NOTE: These tests require the model checkpoint file.  If no checkpoint
is found, they are skipped automatically.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from ofx_plugin.core.export import can_export, export_torchscript

# Skip all tests in this module if no checkpoint is available
pytestmark = pytest.mark.skipif(
    not can_export(),
    reason="Model checkpoint not found — export tests require the .pth file",
)


class TestTorchScriptExport:
    """Test exporting the GreenFormer model to TorchScript."""

    def test_export_produces_file(self):
        """Export should create a .pt file on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "corridorkey.pt")
            export_torchscript(output_path, img_size=256)
            assert os.path.isfile(output_path)
            assert os.path.getsize(output_path) > 1_000_000  # > 1MB

    def test_exported_model_loads(self):
        """The exported .pt file should load back with torch.jit.load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "corridorkey.pt")
            export_torchscript(output_path, img_size=256)
            model = torch.jit.load(output_path, map_location="cpu")
            assert model is not None

    def test_exported_model_forward(self):
        """The exported model should accept a 4-channel input and return alpha + fg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "corridorkey.pt")
            export_torchscript(output_path, img_size=256)
            model = torch.jit.load(output_path, map_location="cpu")

            # Create a dummy 4-channel input [B, C, H, W]
            dummy_input = torch.randn(1, 4, 256, 256)
            with torch.no_grad():
                output = model(dummy_input)

            # Output should be a dict with 'alpha' and 'fg' keys
            assert "alpha" in output
            assert "fg" in output
            assert output["alpha"].shape == (1, 1, 256, 256)
            assert output["fg"].shape == (1, 3, 256, 256)

    def test_exported_matches_original(self):
        """Exported model output must match the original within tolerance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "corridorkey.pt")
            original_model = export_torchscript(
                output_path,
                img_size=256,
                return_original=True,
            )
            exported_model = torch.jit.load(output_path, map_location="cpu")

            dummy_input = torch.randn(1, 4, 256, 256)

            with torch.no_grad():
                orig_out = original_model(dummy_input)
                export_out = exported_model(dummy_input)

            torch.testing.assert_close(
                orig_out["alpha"],
                export_out["alpha"],
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                orig_out["fg"],
                export_out["fg"],
                atol=1e-4,
                rtol=1e-4,
            )
