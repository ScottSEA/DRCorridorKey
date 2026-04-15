# OFX Plugin — Workstation Build & Test Guide

Everything below assumes a Windows workstation with an NVIDIA GPU and DaVinci Resolve installed. Adjust paths for Linux/macOS as needed.

**Assume nothing is configured.** This guide starts from a bare Windows machine with a GPU.

---

## Step 0: Machine setup

Run these in an **Administrator PowerShell**. Skip any tool you already have.

### 0a. Verify your GPU

```powershell
# Check that Windows sees your NVIDIA GPU
nvidia-smi
```

If this fails, install the latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx). Reboot after installing.

### 0b. Install Git

```powershell
winget install Git.Git
```

Close and reopen your terminal after installing so `git` is on PATH.

### 0c. Install uv (Python manager)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal so `uv` is on PATH. Verify:

```powershell
uv --version
```

### 0d. Install CUDA Toolkit

Download and install **CUDA 12.8+** from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

- Select: Windows → x86_64 → exe (local)
- Use the default install options
- Reboot if prompted

Verify:

```powershell
nvcc --version
```

### 0e. Install Visual Studio 2022 (for C++ OFX build only)

Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/). During install, select the **"Desktop development with C++"** workload. This installs MSVC, CMake, and the Windows SDK.

Verify (open a new "Developer Command Prompt for VS 2022"):

```powershell
cl
cmake --version
```

> **Note:** Steps 0d and 0e are only needed for building the C++ OFX plugin (Steps 7-9). You can test the Fuse plugin (Steps 1-2) without them.

### 0f. Install curl (for testing)

Windows 10/11 ships with curl. Verify:

```powershell
curl --version
```

If missing: `winget install cURL.cURL`

### 0g. Install DaVinci Resolve

Download from [blackmagicdesign.com/products/davinciresolve](https://www.blackmagicdesign.com/products/davinciresolve/). The free version works — Studio is not required for plugin development.

- Run the installer with defaults
- Launch Resolve once to complete initial setup (it will ask about project database location — defaults are fine)
- Close Resolve before proceeding

---

## Step 1: Clone and set up the repo

```powershell
git clone https://github.com/ScottSEA/DRCorridorKey.git
cd DRCorridorKey
```

You need two branches:

```powershell
# The Fuse + HTTP service (ready to test now)
git checkout feature/resolve-plugin

# The OFX plugin TDD work (for building the native plugin)
git checkout feature/ofx-plugin
```

---

## Step 2: Test the Fuse plugin (feature/resolve-plugin)

This is the quickest way to verify CorridorKey works in Resolve on your machine.

```powershell
git checkout feature/resolve-plugin

# Install Python deps
uv sync --extra resolve --extra cuda

# Start the service (keep this terminal open)
uv run python -m resolve_plugin --preload

# In another terminal — verify it's working
curl http://localhost:5309/health
```

Install and test the Fuse in Resolve:

```powershell
resolve_plugin\install_fuse.bat
```

1. Restart Resolve
2. Open the **Fusion** page
3. Add Tool → Fuses → Keying → **CorridorKey**
4. Connect a green screen plate to `Image` and a rough matte to `AlphaHint`
5. Click **Warmup Model** in the Inspector, then scrub to process a frame

---

## Step 3: Run the OFX reference tests (feature/ofx-plugin)

```powershell
git checkout feature/ofx-plugin

# Install the project with CUDA support (this also gets torch, numpy, opencv, etc.)
uv sync --extra cuda

# Install test runner
uv pip install pytest

# Run all reference implementation tests
uv run python -m pytest ofx_plugin/tests/ -v
```

You should see **102 passed, 4 skipped**. The 4 skipped tests are the TorchScript export tests — they need the model checkpoint.

---

## Step 4: Download the model checkpoint

The checkpoint downloads automatically on first run, but you can trigger it manually:

```powershell
uv sync --extra cuda
uv run python -c "from CorridorKeyModule.backend import _discover_checkpoint, TORCH_EXT; print(_discover_checkpoint(TORCH_EXT))"
```

This downloads `CorridorKey.pth` (~300MB) to `CorridorKeyModule/checkpoints/`.

---

## Step 5: Run the export tests (now with checkpoint)

```powershell
uv run python -m pytest ofx_plugin/tests/test_export.py -v
```

All 4 export tests should now pass:
- ✅ Export produces a `.pt` file
- ✅ Exported model loads back with `torch.jit.load`
- ✅ Forward pass produces correct output shapes
- ✅ Exported output matches original model

---

## Step 6: Export the TorchScript model

```powershell
# Export at reduced resolution for faster testing
uv run python -m ofx_plugin.core.export --output ofx_plugin/cpp/corridorkey.pt --img-size 256

# Once validated, export at full resolution (slower, ~2 min)
uv run python -m ofx_plugin.core.export --output ofx_plugin/cpp/corridorkey.pt --img-size 2048
```

This creates `corridorkey.pt` — the file the C++ OFX plugin loads.

---

## Step 7: Download build dependencies

### libtorch

Download the C++ distribution of PyTorch:

```powershell
# From https://pytorch.org/get-started/locally/
# Select: Stable → Windows → C++/Java → CUDA 12.8
# Download the zip and extract to e.g. C:\libtorch
```

### OpenFX SDK

```powershell
git clone https://github.com/AcademySoftwareFoundation/openfx.git C:\openfx
```

---

## Step 8: Build the C++ OFX plugin

```powershell
cd DRCorridorKey\ofx_plugin\cpp

# Configure CMake
cmake -B build ^
  -DCMAKE_PREFIX_PATH="C:\libtorch" ^
  -DOFX_SDK_DIR="C:\openfx" ^
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Install (creates the .ofx.bundle directory)
cmake --install build --prefix install
```

This produces:

```
install/
  CorridorKey.ofx.bundle/
    Contents/
      Win64/
        CorridorKey.ofx        ← the plugin binary
      Resources/
        corridorkey.pt         ← the model (if present)
```

---

## Step 9: Install the OFX plugin in Resolve

Copy the bundle to Resolve's OFX directory:

```powershell
xcopy /E /I install\CorridorKey.ofx.bundle ^
  "%CommonProgramFiles%\OFX\Plugins\CorridorKey.ofx.bundle"
```

Also copy the model file if it wasn't installed automatically:

```powershell
copy corridorkey.pt ^
  "%CommonProgramFiles%\OFX\Plugins\CorridorKey.ofx.bundle\Contents\Resources\"
```

Restart Resolve. The plugin should appear in:
- **Edit** page → Effects → OpenFX → CorridorKey
- **Color** page → OpenFX → CorridorKey
- **Fusion** page → Add Tool → OpenFX → CorridorKey

---

## Step 10: Test the OFX plugin

1. Import a green screen clip into Resolve
2. Apply the CorridorKey OFX effect
3. Connect or generate an alpha hint
4. Verify the keyed output matches the Fuse version

---

## Troubleshooting

**CMake can't find Torch:**
→ Set `CMAKE_PREFIX_PATH` to the libtorch directory (the one containing `share/cmake/Torch/`).

**Linker errors about torch symbols:**
→ Make sure you downloaded the CUDA-enabled libtorch (not CPU-only) if you want GPU inference.

**Plugin doesn't appear in Resolve:**
→ Check the bundle is in `C:\Program Files\Common Files\OFX\Plugins\`. The directory structure must be exactly `CorridorKey.ofx.bundle/Contents/Win64/CorridorKey.ofx`.

**Model loading fails at runtime:**
→ Verify `corridorkey.pt` is in the `Resources` directory of the bundle. Check Resolve's console log for the error message.

**VRAM out of memory:**
→ The model uses ~600MB in fp16. Close other GPU-heavy apps. Try exporting at a smaller `--img-size` (e.g. 1024) for testing.

**Export tests still skip:**
→ Make sure the checkpoint is in `CorridorKeyModule/checkpoints/CorridorKey.pth`. Run the download step again.

---

## What's left after this guide

The `plugin_main.cpp` is a skeleton — it registers with the OFX host but doesn't implement the full OFX action handlers (describe, createInstance, render). To complete it:

1. Include the real OFX SDK headers (`ofxImageEffect.h`, `ofxsImageEffect.h`)
2. Implement `describeAction()` — declare inputs, outputs, parameters
3. Implement `createInstanceAction()` — load model
4. Implement `renderAction()` — convert OFX buffers → tensors → process → write back
5. Test in Resolve, then Nuke/Natron for cross-host compatibility

The reference implementations in `ofx_plugin/core/` and the 102 tests define exactly what the C++ code must do. The C++ headers in `ofx_plugin/cpp/src/` already implement the tensor math — what's missing is the OFX host glue.
