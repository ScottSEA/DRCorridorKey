"""TorchScript model export for the OFX plugin.

Exports the CorridorKey GreenFormer model to a TorchScript file
that can be loaded by libtorch in C++.  The exported model accepts
a 4-channel input [B, 4, H, W] and returns a dict with 'alpha'
[B, 1, H, W] and 'fg' [B, 3, H, W].

Usage:
    python -m ofx_plugin.core.export --output corridorkey.pt

The exported .pt file should be bundled with the OFX plugin binary.
"""

from __future__ import annotations

import glob
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

# Ensure project root is on path for imports
_project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _find_checkpoint() -> str | None:
    """Find the CorridorKey .pth checkpoint file.

    Returns:
        Path to the checkpoint, or None if not found or if the
        CorridorKeyModule is not importable (missing timm, etc.).
    """
    try:
        from CorridorKeyModule.backend import CHECKPOINT_DIR, TORCH_EXT
    except ImportError:
        return None

    matches = glob.glob(os.path.join(str(CHECKPOINT_DIR), f"*{TORCH_EXT}"))
    if len(matches) == 1:
        return matches[0]
    return None


def can_export() -> bool:
    """Check whether model export is possible (checkpoint exists).

    Returns False gracefully if dependencies (timm, etc.) are missing,
    so that test collection doesn't crash.

    Returns:
        True if a checkpoint file is found on disk.
    """
    return _find_checkpoint() is not None


def export_torchscript(
    output_path: str,
    img_size: int = 2048,
    return_original: bool = False,
) -> torch.nn.Module | None:
    """Export the GreenFormer model to TorchScript.

    Loads the model from the checkpoint, traces it with a dummy input,
    and saves the traced module to ``output_path``.

    Args:
        output_path: Where to write the .pt TorchScript file.
        img_size: Model input resolution (must match training config).
        return_original: If True, also return the original (non-traced)
            model for comparison testing.

    Returns:
        The original model if ``return_original`` is True, else None.

    Raises:
        FileNotFoundError: If no checkpoint is found.
        RuntimeError: If tracing fails.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(
            "No CorridorKey checkpoint found.  Run the installer first or download from HuggingFace."
        )

    logger.info("Loading model from %s", ckpt_path)

    # Import the model class and checkpoint loader
    from CorridorKeyModule.core.model_transformer import GreenFormer

    # Build the model
    model = GreenFormer(img_size=img_size)

    # Load checkpoint weights
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Handle both raw state_dict and wrapped {"state_dict": ...} formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip torch.compile prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "")
        cleaned[clean_key] = v

    # Handle positional embedding size mismatch
    _resize_pos_embed_if_needed(model, cleaned, img_size)

    model.load_state_dict(cleaned, strict=False)
    model.eval()

    logger.info("Tracing model with input size [1, 4, %d, %d]", img_size, img_size)

    # Create a dummy input for tracing
    dummy_input = torch.randn(1, 4, img_size, img_size)

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    # Save the traced model
    traced.save(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Exported TorchScript model to %s (%.1f MB)", output_path, file_size_mb)

    if return_original:
        return model
    return None


def _resize_pos_embed_if_needed(
    model: torch.nn.Module,
    state_dict: dict,
    target_size: int,
) -> None:
    """Resize positional embeddings in the state dict if they don't match.

    The Hiera backbone has positional embeddings sized for the training
    resolution.  If ``target_size`` differs, we need to interpolate
    them to the new size.

    This modifies ``state_dict`` in place.
    """
    for key in list(state_dict.keys()):
        if "pos_embed" not in key:
            continue

        embed = state_dict[key]
        # Get the model's expected shape for this parameter
        try:
            model_param = dict(model.named_parameters())[key]
        except KeyError:
            continue

        if embed.shape != model_param.shape:
            logger.info(
                "Resizing pos_embed %s: %s → %s",
                key,
                embed.shape,
                model_param.shape,
            )
            # Reshape for interpolation: [1, N, D] → [1, D, H, W] → interpolate → reshape back
            # This is a simplified version — the full logic is in the upstream backend.py
            state_dict[key] = (
                torch.nn.functional.interpolate(
                    embed.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1),
                    size=(model_param.shape[1], 1),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(-1)
                .permute(0, 2, 1)
                .squeeze(0)
            )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export CorridorKey to TorchScript")
    parser.add_argument("--output", default="corridorkey.pt", help="Output .pt file path")
    parser.add_argument("--img-size", type=int, default=2048, help="Model input resolution")
    args = parser.parse_args()

    export_torchscript(args.output, img_size=args.img_size)
