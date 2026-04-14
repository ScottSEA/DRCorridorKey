# CorridorKey — DaVinci Resolve Plugin

AI green-screen keyer for DaVinci Resolve's Fusion page.  Separates a green-screen plate into clean foreground + alpha using the CorridorKey neural network, directly inside your Resolve compositing workflow.

## How It Works

The plugin has two parts:

1. **CorridorKey Service** — A local Python HTTP server that runs the neural network.  It keeps the model loaded in GPU memory so each frame processes quickly (1–5 seconds depending on resolution and GPU).

2. **CorridorKey Fuse** — A Fusion node that sends frames to the service and displays the results.  It appears in the Fusion page under **Add Tool → Fuses → Keying → CorridorKey**.

```
  Fusion Page                        Local Service
  ┌──────────────────┐              ┌───────────────────┐
  │ CorridorKey Fuse │──HTTP/JSON──▶│ Python + PyTorch  │
  │  (Lua node)      │◀─file paths─│  (GPU inference)  │
  └──────────────────┘              └───────────────────┘
```

## Quick Start

### 1. Install Dependencies

From the CorridorKey project root:

```bash
# Install the base project + Resolve plugin dependencies
uv sync --extra resolve
```

If you use CUDA:
```bash
uv sync --extra resolve --extra cuda
```

### 2. Install the Fuse

**Windows:**
```
Double-click  resolve_plugin\install_fuse.bat
```

**Linux / macOS:**
```bash
bash resolve_plugin/install_fuse.sh
```

This copies `CorridorKey.fuse` into Resolve's Fuses directory.  Restart Resolve after installing.

### 3. Start the Service

**Windows:**
```
Double-click  resolve_plugin\Start_CorridorKey_Service.bat
```

**Linux / macOS:**
```bash
bash resolve_plugin/start_corridorkey_service.sh
```

The service starts on `http://127.0.0.1:5309` by default.  Keep the terminal open while working in Resolve.

**Options (CLI):**
```bash
uv run python -m resolve_plugin --port 8080 --device cuda --preload
```

**Options (environment variables):**
```bash
CK_PORT=8080 CK_DEVICE=cuda CK_PRELOAD_MODEL=true uv run python -m resolve_plugin
```

### 4. Use in Resolve

1. Open the **Fusion** page
2. Add the **CorridorKey** node: right-click → Add Tool → Fuses → Keying → CorridorKey
3. Connect your **green-screen plate** to the `Image` input
4. Connect a **rough matte** (BiRefNet, roto, Magic Mask, etc.) to the `AlphaHint` input
5. The node will process the frame and output foreground + alpha

## Controls

| Control | Default | Description |
|---------|---------|-------------|
| Service URL | `http://localhost:5309` | Address of the running service |
| Despill Strength | 1.0 | Green spill removal intensity (0 = none, 1 = full) |
| Auto Despeckle | ✓ | Remove small disconnected alpha islands |
| Despeckle Size | 400 | Minimum pixel count to keep an alpha island |
| Refiner Scale | 1.0 | Neural network refiner intensity |
| Input Is Linear | ✓ | Check if your plate is in linear colour space (typical for EXR) |

## Output Files

The service writes full-precision outputs alongside the preview shown in Fusion:

| File | Format | Use |
|------|--------|-----|
| `fg.exr` | 32-bit float EXR | Full-quality foreground for final comp |
| `alpha.exr` | 32-bit float EXR | Full-quality alpha matte |
| `fg.ppm` | 8-bit PPM | Fuse preview (auto-loaded by the node) |
| `alpha.pgm` | 8-bit PGM | Fuse preview (auto-loaded by the node) |
| `comp.png` | 8-bit PNG | Checker-composite preview |

Output files are in the temp directory (`%TEMP%\corridorkey_resolve` on Windows, `/tmp/corridorkey_resolve` on Linux/macOS).

For final renders, use the EXR outputs via Loader nodes for full precision.

## API Reference

The service exposes these HTTP endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, model state, GPU info |
| `/warmup` | POST | Pre-load the model (avoids slow first frame) |
| `/infer` | POST | Process a single frame (JSON body with file paths) |
| `/shutdown` | POST | Gracefully stop the service |
| `/docs` | GET | Interactive OpenAPI documentation (auto-generated) |

## Troubleshooting

**"Is the service running?"**
→ Start the service first.  The Fuse can't process frames without it.

**First frame is very slow**
→ The model loads on first use (~10–30 seconds).  Use `--preload` or call `/warmup` after starting the service.

**"Could not read input image"**
→ Check that the file paths in the error message exist and are readable.

**VRAM errors**
→ Close other GPU-heavy applications.  Try `--device cpu` for CPU-only mode (much slower).

**Fuse not appearing in Resolve**
→ Ensure the `.fuse` file is in the correct directory (run the installer).  Restart Resolve.

## Hardware Requirements

Same as the main CorridorKey project:
- **NVIDIA GPU**: 6+ GB VRAM recommended (CUDA 12.8+)
- **Apple Silicon**: M1+ with unified memory
- **CPU**: Works but significantly slower

## Architecture Notes

- The service binds to **localhost only** (127.0.0.1) — it is not accessible from the network.
- The service runs a **single worker process** to prevent duplicate model loads.
- The Fuse uses **PPM/PGM** as an interchange format — trivially parseable in Lua 5.1 without external libraries.
- Results are **cached per frame + parameters** to avoid redundant re-processing.
- Full-precision **EXR outputs** are always written alongside the 8-bit Fuse previews.
