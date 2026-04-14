#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Install the CorridorKey Fuse into DaVinci Resolve (Linux / macOS)
#
#  Copies CorridorKey.fuse to the Fusion Fuses directory so it
#  appears as a node in the Fusion page.  Restart Resolve after
#  running this script.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║   CorridorKey — Fuse Installer                ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUSE_SRC="${SCRIPT_DIR}/Fuses/CorridorKey.fuse"

# Detect OS and set the Fuses directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    FUSE_DIR="$HOME/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses"
else
    FUSE_DIR="$HOME/.fusion/BlackmagicDesign/DaVinci Resolve/Fuses"
fi

# Check source exists
if [[ ! -f "$FUSE_SRC" ]]; then
    echo "[ERROR] Cannot find CorridorKey.fuse at:"
    echo "        $FUSE_SRC"
    exit 1
fi

# Create destination directory if needed
mkdir -p "$FUSE_DIR"

# Copy the Fuse file
echo "[INFO] Installing CorridorKey.fuse to:"
echo "       $FUSE_DIR"
cp -f "$FUSE_SRC" "$FUSE_DIR/CorridorKey.fuse"

echo ""
echo "[OK] CorridorKey Fuse installed successfully."
echo "     Restart DaVinci Resolve to see it in the Fusion page."
echo "     Look for it under: Add Tool > Fuses > Keying > CorridorKey"
echo ""
