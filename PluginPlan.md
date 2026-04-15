# CorridorKey → DaVinci Resolve Fusion Plugin Plan

## Problem Statement

CorridorKey is a powerful AI green-screen keyer that produces physically accurate foreground/alpha separation — but it's currently CLI-only. The goal is to integrate it as a **Fusion Fuse node** inside DaVinci Resolve, so artists can use it directly in their compositing workflow without leaving the application.

## Proposed Approach: Hybrid Fuse + Python Backend

We'll build a **Lua Fuse** (Fusion node) that communicates with a **local Python service** running the CorridorKey inference engine. This avoids rewriting the PyTorch model in C++ (OFX) while providing a native-feeling Resolve experience.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  DaVinci Resolve / Fusion Page                  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  CorridorKey Fuse Node (.fuse)            │  │
│  │  - Accepts image input + alpha hint       │  │
│  │  - Exposes controls (despill, despeckle)  │  │
│  │  - Writes input frame to temp dir         │  │
│  │  - Calls local HTTP API                   │  │
│  │  - Reads back result EXR (FG + Alpha)     │  │
│  │  - Outputs FG image + Alpha channel       │  │
│  └──────────────┬────────────────────────────┘  │
│                 │ HTTP (localhost)               │
└─────────────────┼───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  CorridorKey Local Service (Python)             │
│                                                 │
│  - FastAPI / Flask on localhost:PORT             │
│  - Wraps CorridorKeyEngine                      │
│  - Accepts frame + mask via multipart/file path │
│  - Returns processed FG + Alpha EXR             │
│  - Manages GPU, model loading, VRAM             │
│  - Job queue for sequential GPU work            │
└─────────────────────────────────────────────────┘
```

### Why HTTP instead of subprocess-per-frame?

- **Model stays loaded**: The PyTorch model (~300MB) loads once and stays resident in VRAM. Subprocess-per-frame would reload the model every time — unusably slow.
- **Job queue built-in**: The existing `GPUJobQueue` already serializes GPU work.
- **Cross-platform**: HTTP works identically on Windows, Mac, and Linux.
- **Decoupled lifecycle**: The service can start independently or be launched by the Fuse on first use.

## Deliverables

### Phase 1: Python Backend Service

Build a lightweight HTTP API that wraps the existing `CorridorKeyService` / `CorridorKeyEngine`.

- **Endpoint: `POST /infer`**
  - Input: image file path + alpha hint file path (or multipart upload)
  - Parameters: `despill_strength`, `despeckle_size`, `auto_despeckle`, `refiner_scale`, `input_is_linear`, `fg_is_straight`
  - Output: JSON with paths to output FG and Alpha EXR files (in a temp/output dir)
- **Endpoint: `GET /health`**
  - Returns service status, GPU info, model loaded state
- **Endpoint: `POST /shutdown`**
  - Graceful shutdown for cleanup
- **Startup**: Auto-downloads model on first run (existing logic). Configurable port, device selection.
- **File**: `resolve_plugin/service.py`

### Phase 2: Fusion Fuse Node (Lua)

Create a `.fuse` file that appears as a node in the Fusion page.

- **Inputs**:
  - `Image` — the green screen plate (primary input)
  - `AlphaHint` — rough matte / roto (secondary input, optional)
- **Outputs**:
  - `Output` — composited/processed foreground with embedded alpha
- **Controls** (in the Inspector panel):
  - Service URL (default: `http://localhost:5309`)
  - Despill Strength (slider, 0.0–1.0)
  - Auto Despeckle (checkbox)
  - Despeckle Size (integer)
  - Refiner Scale (slider)
  - Input Is Linear (checkbox, default true for EXR workflows)
  - Process button / auto-process toggle
- **Behavior per frame**:
  1. Write input image + alpha hint to temp EXR files
  2. POST to `/infer` with file paths and parameters
  3. Read result EXR files
  4. Load into Fusion image buffer and set as output
- **File**: `resolve_plugin/Fuses/CorridorKey.fuse`

### Phase 3: Launcher & Installer

- **Service launcher script** — starts the Python backend service
  - Windows: `Start_CorridorKey_Service.bat`
  - Linux/Mac: `start_corridorkey_service.sh`
  - Auto-detects GPU, sets up environment, launches on configurable port
- **Fuse installer script** — copies the `.fuse` file to the correct Resolve/Fusion directory
  - Windows: `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Fuses\`
  - Mac: `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses/`
  - Linux: `~/.fusion/BlackmagicDesign/DaVinci Resolve/Fuses/`
- **README** for the plugin with setup instructions

### Phase 4: Quality of Life

- **Auto-launch service from Fuse**: If the service isn't running when the Fuse tries to connect, attempt to launch it automatically via `os.execute()`
- **Progress feedback**: Show processing status in the Fusion console
- **Batch/sequence mode**: Process entire frame sequences efficiently (the service already supports batch inference)
- **Caching**: Skip re-processing frames that haven't changed (hash-based or manifest)
- **Error handling**: Graceful degradation when service is unavailable, clear error messages in Fusion console

## Technical Considerations

### Existing Code We Can Reuse
- `CorridorKeyEngine` — the full inference pipeline (model loading, pre/post-processing)
- `device_utils.py` — GPU detection and setup
- `frame_io.py` — EXR read/write with proper color handling
- `GPUJobQueue` — serialized GPU job scheduling
- `backend.py` — model download and checkpoint discovery

### Challenges
- **Lua ↔ EXR**: Fusion's Lua API can read/write images but EXR interop may need careful handling. We may use PNG/TIFF as an intermediate format if EXR is problematic from Lua.
- **Latency**: Each frame requires disk I/O + HTTP + inference. Not real-time — but comparable to other AI tools in Resolve (e.g., Magic Mask). Expect 1-5 seconds per frame depending on resolution and GPU.
- **VRAM management**: The service holds the model in VRAM. If Resolve is also using the GPU heavily, VRAM contention is possible. May need a "low VRAM" mode that unloads between frames.
- **Color space**: CorridorKey expects linear float input. Fusion works in linear by default — this is a natural fit. Need to ensure no double-gamma is applied.

### Dependencies (additional to existing)
- `fastapi` + `uvicorn` (or `flask`) for the HTTP service
- `python-multipart` if using multipart uploads

## File Structure

```
resolve_plugin/
├── service.py              # FastAPI backend wrapping CorridorKeyEngine
├── Fuses/
│   └── CorridorKey.fuse    # Lua Fuse node for Fusion
├── Start_CorridorKey_Service.bat
├── start_corridorkey_service.sh
├── install_fuse.bat
├── install_fuse.sh
└── README.md
```

## Execution Order

1. Build the HTTP service (`resolve_plugin/service.py`)
2. Test the service standalone with curl/Postman
3. Build the Fuse node (`CorridorKey.fuse`)
4. Test end-to-end in DaVinci Resolve
5. Build launcher/installer scripts
6. Add quality-of-life features (auto-launch, caching, batch mode)
7. Write user-facing README
