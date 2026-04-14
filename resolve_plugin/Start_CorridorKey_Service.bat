@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  Start the CorridorKey Resolve Service (Windows)
REM
REM  This script launches the local HTTP backend that the Fusion Fuse
REM  connects to.  Keep this window open while using CorridorKey in
REM  DaVinci Resolve.
REM
REM  Options (set as environment variables before running):
REM    CK_PORT=5309          TCP port (default: 5309)
REM    CK_DEVICE=auto        Device: auto, cuda, mps, cpu
REM    CK_PRELOAD_MODEL=true Pre-load the model at startup
REM ═══════════════════════════════════════════════════════════════════

setlocal

echo.
echo  ╔═══════════════════════════════════════════════╗
echo  ║   CorridorKey — DaVinci Resolve Service       ║
echo  ╚═══════════════════════════════════════════════╝
echo.

REM Navigate to the project root (parent of resolve_plugin/)
cd /d "%~dp0.."

REM Check for uv
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    echo         Install it from: https://docs.astral.sh/uv/
    echo         Then re-run this script.
    pause
    exit /b 1
)

REM Install the resolve extras if not already present
echo [INFO] Ensuring dependencies are installed...
uv sync --extra resolve --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

REM Launch the service
echo [INFO] Starting service on http://127.0.0.1:%CK_PORT%...
echo [INFO] Press Ctrl+C to stop.
echo.

uv run python -m resolve_plugin %*

pause
