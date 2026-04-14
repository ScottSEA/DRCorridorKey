"""Entry point for running the CorridorKey Resolve service.

Usage:
    uv run python -m resolve_plugin
    uv run python -m resolve_plugin --port 8080 --device cuda
    uv run python -m resolve_plugin --preload

This module intentionally enforces ``workers=1`` — the inference model
must only be loaded once per process to avoid VRAM exhaustion.  Do NOT
override this when deploying with uvicorn directly.
"""

from __future__ import annotations

import argparse
import logging
import sys

# Configure logging early so startup messages are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resolve_plugin")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    CLI args take priority over environment variables.  Environment
    variables (``CK_PORT``, ``CK_DEVICE``, etc.) are handled by the
    Pydantic settings in ``config.py`` and are NOT duplicated here.
    This parser covers the most common flags for convenience.
    """
    parser = argparse.ArgumentParser(
        prog="resolve_plugin",
        description="CorridorKey inference service for DaVinci Resolve",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address (default: 127.0.0.1).  Keep localhost for security.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port (default: 5309).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Compute device (default: auto-detect).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=False,
        help="Pre-load the model at startup instead of on first request.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Start the uvicorn server with enforced single-worker mode."""
    args = parse_args(argv)

    # Apply log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Override settings via environment variables so the Pydantic config
    # picks them up when the app factory reads get_settings().
    import os

    if args.host is not None:
        os.environ["CK_HOST"] = args.host
    if args.port is not None:
        os.environ["CK_PORT"] = str(args.port)
    if args.device is not None:
        os.environ["CK_DEVICE"] = args.device
    if args.preload:
        os.environ["CK_PRELOAD_MODEL"] = "true"

    # Clear the settings cache so new env vars take effect
    from .config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    logger.info("Starting CorridorKey Resolve service")
    logger.info("  Host:    %s", settings.host)
    logger.info("  Port:    %s", settings.port)
    logger.info("  Device:  %s", settings.device)
    logger.info("  Preload: %s", settings.preload_model)

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn is not installed.  Install it with:\n"
            "  uv pip install uvicorn[standard]"
        )
        sys.exit(1)

    # CRITICAL: workers=1 — the model MUST only load once per process.
    # Multiple workers would each load a copy, exhausting VRAM.
    uvicorn.run(
        "resolve_plugin.app:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
