"""Engine manager — thin wrapper around the existing CorridorKeyService.

This module owns the lifecycle of the inference engine: cold → loading →
ready → (optional) unloaded.  It deliberately does NOT duplicate any
inference or I/O logic from the upstream ``backend.service`` module;
instead it reuses ``CorridorKeyService`` for device detection, model
loading, VRAM management, and frame processing.

Thread safety:
    All public methods are safe to call from any thread.  The underlying
    ``CorridorKeyService`` already uses a ``threading.Lock`` around GPU
    operations.  This module adds a second lock around model-lifecycle
    transitions (load/unload) to prevent concurrent first-request races.

Design rationale (SRP):
    Route handlers deal with HTTP concerns (validation, serialisation).
    This module deals with model concerns (lifecycle, device, inference).
    ``path_security`` deals with filesystem concerns.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ServiceSettings
from .models import ModelState

logger = logging.getLogger(__name__)


@dataclass
class SingleFrameResult:
    """Inference output for a single frame, as numpy arrays.

    Attributes:
        fg: Foreground RGB [H, W, 3] float32 in sRGB, straight alpha.
        alpha: Alpha matte [H, W] or [H, W, 1] float32 in [0, 1].
        comp: Optional checker-composite [H, W, 3] float32 sRGB.
    """

    fg: np.ndarray
    alpha: np.ndarray
    comp: np.ndarray | None = None


class EngineManager:
    """Manages the CorridorKey inference engine lifecycle.

    Wraps ``backend.service.CorridorKeyService`` — the upstream module
    already handles lazy model loading, device detection, VRAM cleanup,
    and GPU locking.  This class adds:

    - Explicit lifecycle states (``ModelState``) for the ``/health`` endpoint.
    - A lifecycle lock to prevent concurrent first-request load races.
    - A simplified single-frame inference method for the Resolve service.
    - Checkpoint discovery status for health reporting.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._state = ModelState.COLD
        self._lifecycle_lock = threading.Lock()
        self._last_error: str | None = None

        # Lazy import — we set env vars (ROCm, EXR) before importing
        # anything that touches torch or cv2.
        self._backend_service = None  # type: ignore[assignment]

    # ── Lifecycle ────────────────────────────────────────────────────────

    @property
    def state(self) -> ModelState:
        """Current lifecycle state of the inference model."""
        return self._state

    @property
    def last_error(self) -> str | None:
        """Human-readable description of the last load failure, if any."""
        return self._last_error

    def _get_backend(self):
        """Lazy-initialise the upstream CorridorKeyService singleton.

        Separated from ``_ensure_ready`` so we can call ``detect_device``
        and ``get_vram_info`` without triggering a full model load.
        """
        if self._backend_service is not None:
            return self._backend_service

        # Import here to respect ROCm / EXR env-var ordering
        from backend.service import CorridorKeyService

        svc = CorridorKeyService()

        # Resolve the compute device (auto / cuda / mps / cpu)
        if self._settings.device == "auto":
            svc.detect_device()
        else:
            svc._device = self._settings.device

        self._backend_service = svc
        return svc

    def _ensure_ready(self) -> None:
        """Ensure the model is loaded and ready for inference.

        This is idempotent — if the model is already ``READY``, it
        returns immediately.  Uses a lifecycle lock so that concurrent
        first requests don't trigger duplicate loads.
        """
        if self._state == ModelState.READY:
            return

        with self._lifecycle_lock:
            # Double-check after acquiring the lock
            if self._state == ModelState.READY:
                return

            self._state = ModelState.LOADING
            logger.info("Loading CorridorKey inference engine...")
            t0 = time.monotonic()

            try:
                svc = self._get_backend()
                # Trigger the lazy engine load via the existing backend
                svc._get_engine()
                self._state = ModelState.READY
                elapsed = time.monotonic() - t0
                logger.info("Engine ready in %.1fs", elapsed)
            except Exception as exc:
                self._state = ModelState.FAILED
                self._last_error = str(exc)
                logger.error("Engine load failed: %s", exc, exc_info=True)
                raise

    def warmup(self) -> None:
        """Explicitly pre-load the model.  Called at startup when
        ``preload_model`` is True, or via the ``POST /warmup`` endpoint.
        """
        self._ensure_ready()

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._backend_service is not None:
            self._backend_service.unload_engines()
        self._state = ModelState.COLD
        logger.info("Engine unloaded")

    # ── Health info ──────────────────────────────────────────────────────

    def get_device(self) -> str:
        """Return the resolved compute device string."""
        svc = self._get_backend()
        return svc._device

    def get_vram_info(self) -> dict | None:
        """Return GPU VRAM stats, or None if not applicable."""
        svc = self._get_backend()
        info = svc.get_vram_info()
        return info if info else None

    def is_checkpoint_found(self) -> bool:
        """Check whether a model checkpoint file exists on disk."""
        try:
            from CorridorKeyModule.backend import CHECKPOINT_DIR, TORCH_EXT

            import glob as _glob

            matches = _glob.glob(os.path.join(str(CHECKPOINT_DIR), f"*{TORCH_EXT}"))
            return len(matches) > 0
        except Exception:
            return False

    # ── Inference ────────────────────────────────────────────────────────

    def infer_single_frame(
        self,
        image_path: str,
        alpha_hint_path: str,
        *,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
    ) -> SingleFrameResult:
        """Run inference on a single frame.

        Reads the input image and alpha hint from disk, runs the model,
        and returns numpy arrays.  Writing outputs to disk is the
        caller's responsibility (separation of concerns).

        Args:
            image_path: Absolute path to the green-screen frame.
            alpha_hint_path: Absolute path to the alpha hint / rough matte.
            despill_strength: Despill multiplier (0–1).
            auto_despeckle: Auto-remove small alpha islands.
            despeckle_size: Minimum island pixel count.
            refiner_scale: Refiner delta multiplier.
            input_is_linear: Whether the input is linear (vs sRGB).

        Returns:
            SingleFrameResult with fg, alpha, and optional comp arrays.

        Raises:
            RuntimeError: If the model fails to load.
            ValueError: If a frame cannot be read.
        """
        # Ensure the model is loaded (idempotent)
        self._ensure_ready()

        svc = self._get_backend()

        # --- Read input frame ---
        from backend.frame_io import read_image_frame, read_mask_frame

        img = read_image_frame(image_path)
        if img is None:
            raise ValueError(f"Could not read input image: {image_path}")

        mask = read_mask_frame(alpha_hint_path)
        if mask is None:
            raise ValueError(f"Could not read alpha hint: {alpha_hint_path}")

        # Resize mask to match input dimensions if needed
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(
                mask, (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # --- Run inference (GPU-locked by the upstream service) ---
        t0 = time.monotonic()
        with svc._gpu_lock:
            engine = svc._get_engine()
            result = engine.process_frame(
                img,
                mask,
                input_is_linear=input_is_linear,
                fg_is_straight=True,
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size,
                refiner_scale=refiner_scale,
                generate_comp=True,
            )
        elapsed = time.monotonic() - t0
        logger.info("Inference completed in %.3fs", elapsed)

        return SingleFrameResult(
            fg=result["fg"],
            alpha=result["alpha"],
            comp=result.get("comp"),
        )
