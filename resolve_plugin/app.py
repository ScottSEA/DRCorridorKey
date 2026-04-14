"""FastAPI application — HTTP routes for the CorridorKey Resolve service.

This is the top-level FastAPI app.  It is deliberately thin: routes
validate input, delegate to the engine manager, write outputs via the
output writer, and return responses.  No inference or I/O logic lives
here (SRP).

Environment setup:
    ROCm env vars and OpenEXR flags MUST be set before any module
    imports torch or cv2.  This is handled in the module-level init
    block below — keep it at the top of the file.

Threading model:
    Routes are defined as regular ``def`` (sync) functions, not
    ``async def``.  FastAPI runs sync route handlers in a threadpool,
    which is exactly what we want: the PyTorch inference is blocking
    CPU/GPU work that would starve an async event loop.

    The service MUST run with a single worker process (``--workers 1``)
    to prevent multiple processes from loading the model into VRAM
    simultaneously.  The ``__main__`` entry point enforces this.
"""

from __future__ import annotations

# ── Environment setup (MUST come before any torch/cv2 imports) ───────────
# These env vars affect library initialisation at import time, so they
# must be set before anything else in the dependency chain runs.
import os
import sys

# Enable OpenEXR support in OpenCV (must precede cv2 import anywhere)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Set up ROCm env vars if on an AMD system (must precede torch import)
# Add the project root to sys.path so we can import device_utils, backend, etc.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from device_utils import setup_rocm_env

setup_rocm_env()

# ── Standard imports (safe now that env is configured) ───────────────────
import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import ServiceSettings, get_settings
from .engine_manager import EngineManager
from .models import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    InferResponse,
    ModelState,
)
from .cleanup import purge_old_outputs
from .output_writer import write_inference_outputs
from .path_security import validate_input_path, validate_output_dir

logger = logging.getLogger(__name__)

# ── Application factory ─────────────────────────────────────────────────


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Using a factory function (rather than a module-level ``app``) makes
    testing easier — each test can create a fresh app with custom
    settings.

    Args:
        settings: Override settings for testing.  If None, uses the
                  singleton from ``get_settings()``.

    Returns:
        Configured FastAPI instance with all routes registered.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="CorridorKey Resolve Service",
        description=(
            "Local HTTP backend for the CorridorKey DaVinci Resolve "
            "Fusion plugin.  Wraps the CorridorKey inference engine "
            "and exposes single-frame keying via a simple REST API."
        ),
        version="0.1.0",
    )

    # Create the engine manager — lives for the lifetime of the app.
    # Stored on app.state so route handlers (and tests) can access it.
    engine = EngineManager(settings)
    app.state.engine = engine
    app.state.settings = settings

    # ── Startup event ────────────────────────────────────────────────
    @app.on_event("startup")
    def _on_startup() -> None:
        """Run once when the server starts.

        If ``preload_model`` is True, eagerly load the model so the
        first ``/infer`` request doesn't pay the cold-start penalty.
        """
        # Ensure the temp directory exists
        os.makedirs(settings.temp_dir, exist_ok=True)
        logger.info("Temp directory: %s", settings.temp_dir)
        logger.info("Device: %s", settings.device)

        # Clean up stale request directories from previous sessions
        purge_old_outputs(settings.temp_dir)

        if settings.preload_model:
            logger.info("Preloading model (preload_model=True)...")
            try:
                engine.warmup()
            except Exception:
                # Log but don't crash — the /health endpoint will
                # report FAILED state, and /infer will retry.
                logger.error("Preload failed (will retry on first /infer)")

    # ── Routes ───────────────────────────────────────────────────────

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service health check",
        description="Returns the current state of the service, model, and GPU.",
    )
    def health() -> HealthResponse:
        """Report service and model health.

        This endpoint is lightweight — it never triggers a model load.
        The Fuse can poll this to check readiness before sending frames.
        """
        return HealthResponse(
            status="ok",
            model_state=engine.state,
            device=engine.get_device(),
            vram=engine.get_vram_info(),
            checkpoint_found=engine.is_checkpoint_found(),
        )

    @app.post(
        "/warmup",
        summary="Pre-load the inference model",
        description=(
            "Eagerly loads the model into GPU memory.  Call this after "
            "starting the service to avoid a slow first /infer request."
        ),
        responses={500: {"model": ErrorResponse}},
    )
    def warmup() -> dict:
        """Trigger model load without running inference.

        Returns a simple success message.  If the model is already
        loaded, this is a no-op.
        """
        try:
            engine.warmup()
            return {"status": "ready"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post(
        "/infer",
        response_model=InferResponse,
        summary="Run single-frame inference",
        description=(
            "Reads a green-screen frame and alpha hint from disk, runs "
            "the CorridorKey model, and writes foreground + alpha EXR "
            "outputs.  Returns the output file paths."
        ),
        responses={
            400: {"model": ErrorResponse, "description": "Bad input (missing file, bad path)"},
            500: {"model": ErrorResponse, "description": "Inference or I/O failure"},
        },
    )
    def infer(req: InferRequest) -> InferResponse:
        """Process a single frame through the CorridorKey model.

        Steps:
            1. Validate and normalise input paths (security).
            2. Resolve inference parameters (request overrides → config defaults).
            3. Run inference via the engine manager.
            4. Write outputs to disk atomically.
            5. Return output file paths.
        """
        # ── 1. Validate paths ────────────────────────────────────────
        try:
            image_path = validate_input_path(req.image_path, settings)
            alpha_path = validate_input_path(req.alpha_hint_path, settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Determine output directory: request-specified or temp subdir
        if req.output_dir:
            try:
                out_dir = validate_output_dir(req.output_dir, settings)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        else:
            # Create a unique subdirectory per request to avoid collisions
            out_dir = os.path.join(settings.temp_dir, uuid.uuid4().hex[:12])
            os.makedirs(out_dir, exist_ok=True)

        # ── 2. Resolve parameters (request → config defaults) ────────
        despill = (
            req.despill_strength
            if req.despill_strength is not None
            else settings.default_despill_strength
        )
        auto_ds = (
            req.auto_despeckle
            if req.auto_despeckle is not None
            else settings.default_auto_despeckle
        )
        ds_size = (
            req.despeckle_size
            if req.despeckle_size is not None
            else settings.default_despeckle_size
        )
        refiner = (
            req.refiner_scale
            if req.refiner_scale is not None
            else settings.default_refiner_scale
        )
        is_linear = (
            req.input_is_linear
            if req.input_is_linear is not None
            else settings.default_input_is_linear
        )

        # ── 3. Run inference ─────────────────────────────────────────
        try:
            result = engine.infer_single_frame(
                image_path=image_path,
                alpha_hint_path=alpha_path,
                despill_strength=despill,
                auto_despeckle=auto_ds,
                despeckle_size=ds_size,
                refiner_scale=refiner,
                input_is_linear=is_linear,
            )
        except ValueError as exc:
            # Frame read failures → 400
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            # Model load failures → 500
            raise HTTPException(status_code=500, detail=str(exc))

        # ── 4. Write outputs ─────────────────────────────────────────
        try:
            paths = write_inference_outputs(result, out_dir, settings)
        except IOError as exc:
            raise HTTPException(status_code=500, detail=f"Output write failed: {exc}")

        # ── 5. Return paths ──────────────────────────────────────────
        return InferResponse(
            fg_path=paths["fg_path"],
            alpha_path=paths["alpha_path"],
            fg_ppm_path=paths["fg_ppm_path"],
            alpha_pgm_path=paths["alpha_pgm_path"],
            comp_path=paths.get("comp_path"),
        )

    @app.post(
        "/shutdown",
        summary="Gracefully shut down the service",
        description=(
            "Unloads the model, frees GPU memory, and stops the server. "
            "Only accessible from localhost."
        ),
    )
    def shutdown() -> dict:
        """Unload the model and signal the server to stop.

        The actual process exit is handled by raising SystemExit after
        cleanup — uvicorn catches this and shuts down gracefully.
        """
        engine.unload()
        logger.info("Shutdown requested via API")
        # Schedule process exit after the response is sent
        import threading

        threading.Timer(0.5, lambda: os._exit(0)).start()
        return {"status": "shutting_down"}

    @app.post(
        "/cleanup",
        summary="Clean up old temp files",
        description=(
            "Removes request output directories older than the specified "
            "age (default: 24 hours).  Runs automatically at startup; "
            "this endpoint allows manual triggering."
        ),
    )
    def cleanup(max_age_hours: float = 24.0) -> dict:
        """Purge stale request directories from the temp directory."""
        removed = purge_old_outputs(settings.temp_dir, max_age_hours)
        return {"removed": removed, "temp_dir": settings.temp_dir}

    return app


# ── Module-level app instance (used by uvicorn) ─────────────────────────
# When running ``uvicorn resolve_plugin.app:app``, uvicorn imports this
# module and looks for the ``app`` variable.
app = create_app()
