#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Start the CorridorKey Resolve Service (Linux / macOS)
#
#  This script launches the local HTTP backend that the Fusion Fuse
#  connects to.  Keep this terminal open while using CorridorKey in
#  DaVinci Resolve.
#
#  Options (set as environment variables before running):
#    CK_PORT=5309          TCP port (default: 5309)
#    CK_DEVICE=auto        Device: auto, cuda, mps, cpu
#    CK_PRELOAD_MODEL=true Pre-load the model at startup
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║   CorridorKey — DaVinci Resolve Service       ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Navigate to the project root (parent of resolve_plugin/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv is not installed or not on PATH."
    echo "        Install it from: https://docs.astral.sh/uv/"
    exit 1
fi

# Install the resolve extras if not already present
echo "[INFO] Ensuring dependencies are installed..."
uv sync --extra resolve --quiet

# Launch the service
PORT="${CK_PORT:-5309}"
echo "[INFO] Starting service on http://127.0.0.1:${PORT}..."
echo "[INFO] Press Ctrl+C to stop."
echo ""

uv run python -m resolve_plugin "$@"
