# OFX Plugin Port — Technical Breakdown

## Goal

Replace the current Fuse + HTTP service architecture with a single self-contained OFX binary plugin. This would make CorridorKey appear as a native effect in Resolve's Edit, Color, and Fusion pages — no Python, no service, no setup.

---

## Architecture Comparison

| | Current (Fuse + Service) | OFX Plugin |
|---|---|---|
| **Language** | Lua + Python | C++ |
| **Inference** | PyTorch (Python) | libtorch (C++ PyTorch API) |
| **Distribution** | Manual install + service | Single `.ofx.bundle` file |
| **Pages** | Fusion only | Edit, Color, Fusion |
| **Setup** | Install Fuse, start service, install deps | Drop plugin in folder, restart Resolve |
| **Perf** | HTTP + disk I/O overhead per frame | Direct GPU memory, zero overhead |

---

## What the Model Looks Like

| Property | Value |
|----------|-------|
| **Architecture** | Hiera (vision transformer) backbone + dual decoder heads + CNN refiner |
| **Parameters** | ~75M (~300MB checkpoint) |
| **Backbone** | `hiera_base_plus_224` — 24 transformer blocks, 4 feature scales |
| **Input** | 4-channel (RGB + alpha hint), resized to 2048×2048 |
| **Output** | Alpha [H,W,1] + Foreground RGB [H,W,3], both float32 |
| **Precision** | float16 inference with autocast |
| **Custom CUDA kernels** | None — all standard PyTorch ops |

---

## Work Breakdown

### Phase 1: Model Export (~1-2 weeks)

**Convert the Python model to TorchScript or ONNX for libtorch consumption.**

- Export `GreenFormer` via `torch.jit.trace()` or `torch.export()`
- The core forward pass uses only standard ops — **no blockers** for TorchScript
- Hiera attention uses `F.scaled_dot_product_attention` (supported in libtorch)
- Handle positional embedding resize (bicubic interpolation of PE weights)
- Validate exported model produces identical outputs to Python version
- Bundle the `.pt` TorchScript file with the plugin

**Risk:** Hiera's unroll/reroll mask-unit attention has dynamic shapes. May need `torch.jit.script` for those paths or a fixed input size.

### Phase 2: C++ Preprocessing (~1 week)

**Reimplement `_preprocess_input` in C++.**

The preprocessing is straightforward tensor math:
- Bilinear resize to model input size (2048×2048)
- sRGB ↔ linear conversion (piecewise `pow` + `where` — ~10 lines of C++)
- ImageNet normalisation (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)
- Concatenate RGB + mask → 4 channels

All of this maps directly to libtorch tensor ops.

### Phase 3: C++ Postprocessing (~2-3 weeks)

**Reimplement `_postprocess_torch` / `_postprocess_opencv` in C++.**

This is the most complex piece:

| Operation | Difficulty | Notes |
|-----------|-----------|-------|
| Bilinear resize back to original size | Easy | `torch::nn::functional::interpolate` |
| Despill | Easy | Simple channel math (~20 lines) |
| sRGB ↔ linear | Easy | Same piecewise function as preprocessing |
| Premultiply alpha | Easy | Multiply RGB × alpha |
| Sigmoid | Easy | `torch::sigmoid` |
| **Matte cleanup (despeckle)** | **Hard** | Uses `connected_components`, `bincount`, `unique`, morphological ops |
| Checkerboard composite | Easy | Procedural pattern + blend |

**The matte cleanup is the hard part.** Options:
1. **Use OpenCV C++ API** — `cv::connectedComponentsWithStats`, `cv::dilate`, `cv::GaussianBlur` all exist in C++. Just need to convert tensors ↔ cv::Mat.
2. **Keep it in libtorch** — reimplement with `max_pool2d` + custom logic. More GPU-friendly but more code.
3. **Skip it initially** — make despeckle optional and add it in a later version.

### Phase 4: OFX Plugin Shell (~2 weeks)

**Build the OFX host integration.**

- OFX plugin skeleton (register, describe, create instance)
- Image I/O: convert OFX image buffers ↔ libtorch tensors (RGBA float ↔ NCHW tensor)
- Parameter UI: sliders for despill, despeckle, refiner scale, etc.
- Model lifecycle: load TorchScript model once on plugin init, keep in GPU memory
- Multi-resolution support: handle Resolve sending different resolutions
- Thread safety: libtorch inference must be serialised (one frame at a time on GPU)

**Key OFX considerations for Resolve:**
- Resolve supports OFX on Edit and Color pages (plus Fusion)
- GPU rendering via CUDA — plugin gets raw GPU buffer pointers
- Must handle both float32 and uint8 pixel formats
- Bundle as `.ofx.bundle` directory structure (platform-specific)

### Phase 5: Build System & Packaging (~1 week)

- CMake build system linking libtorch + OpenFX SDK + OpenCV (optional)
- Cross-platform builds: Windows (MSVC), Linux (GCC), macOS (Clang + Metal?)
- Bundle layout: `CorridorKey.ofx.bundle/Contents/{Win64,Linux64,MacOS}/CorridorKey.ofx`
- Ship model checkpoint inside the bundle or download on first use
- Installer for each platform

### Phase 6: Testing & Polish (~2 weeks)

- Verify output matches Python version (numerical diff < threshold)
- Test in Resolve (Edit, Color, Fusion pages)
- Test in Nuke, Natron (OFX is cross-host)
- VRAM management (model is ~600MB in fp16 on GPU)
- Error handling (no GPU, wrong CUDA version, missing model file)
- Performance benchmarking vs Python service

---

## Total Estimate

| Phase | Effort |
|-------|--------|
| Model export | 1-2 weeks |
| C++ preprocessing | 1 week |
| C++ postprocessing | 2-3 weeks |
| OFX plugin shell | 2 weeks |
| Build system & packaging | 1 week |
| Testing & polish | 2 weeks |
| **Total** | **9-11 weeks** |

This assumes one experienced C++ developer who knows libtorch and OFX.

---

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hiera dynamic shapes block TorchScript export | Can't export model | Use fixed input size (2048×2048), or use ONNX Runtime instead |
| libtorch version must match training torch version | Model won't load | Pin libtorch to torch 2.8.0 |
| VRAM contention with Resolve | OOM crashes | Implement model unload when not in use, or fp16 quantization |
| OFX GPU buffer format varies by host | Rendering artifacts | Test extensively in Resolve + Nuke |
| macOS — no CUDA | No GPU inference on Mac | Use MPS via libtorch (experimental) or CPU fallback |
| Model checkpoint is ~300MB | Large plugin download | Host on HuggingFace, download on first use |

---

## Alternative: ONNX Runtime Instead of libtorch

Instead of libtorch, the model could be exported to ONNX and run via ONNX Runtime C++ API. This has tradeoffs:

| | libtorch | ONNX Runtime |
|---|---|---|
| **Export difficulty** | TorchScript trace/script | `torch.onnx.export` |
| **Runtime size** | ~800MB (large) | ~200MB (smaller) |
| **GPU support** | CUDA, MPS | CUDA, DirectML, CoreML, TensorRT |
| **Operator coverage** | Full PyTorch | Most standard ops, some gaps |
| **Hiera compatibility** | Native | May need op workarounds |
| **Performance** | Good | Often better (TensorRT backend) |

ONNX Runtime would reduce the plugin size and potentially improve performance via TensorRT, but adds export complexity.

---

## Recommendation

Start with **libtorch + TorchScript** — it's the path of least resistance since the model is already in PyTorch. If TorchScript export hits dynamic-shape issues with Hiera, fall back to ONNX.

The Fuse + Service architecture we built can serve as the **reference implementation** and test harness during OFX development — run both side by side and compare outputs.
