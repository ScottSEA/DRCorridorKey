/**
 * @file plugin_main.cpp
 * @brief OFX plugin entry point and registration for CorridorKey.
 *
 * This file implements the OFX plugin interface functions that the
 * host (DaVinci Resolve, Nuke, etc.) calls to discover and create
 * the CorridorKey effect.
 *
 * NOTE: This is a skeleton — it uses placeholder OFX API calls.
 * The full implementation requires the OpenFX SDK headers and
 * the OFX Support Library.  Build with CMake and the SDK to compile.
 *
 * OFX plugin lifecycle:
 *   1. Host loads the .ofx shared library
 *   2. Host calls OfxGetNumberOfPlugins() and OfxGetPlugin()
 *   3. Host calls the plugin's describe/createInstance actions
 *   4. Host calls render() for each frame
 *   5. Host destroys the instance when done
 */

// When building with the real OFX SDK, include these:
// #include "ofxImageEffect.h"
// #include "ofxsImageEffect.h"

#include <iostream>

// ── Plugin metadata ─────────────────────────────────────────────────

static const char* PLUGIN_ID = "com.corridordigital.corridorkey";
static const char* PLUGIN_NAME = "CorridorKey";
static const int   PLUGIN_VERSION_MAJOR = 0;
static const int   PLUGIN_VERSION_MINOR = 1;

// ── OFX entry points (skeleton) ─────────────────────────────────────

/**
 * Called by the host to query how many plugins this binary contains.
 * CorridorKey is a single plugin, so we always return 1.
 */
extern "C" int OfxGetNumberOfPlugins(void) {
    return 1;
}

/**
 * Called by the host to get the plugin descriptor for index `nth`.
 *
 * TODO: Return a real OfxPlugin struct when building with the OFX SDK.
 * This skeleton just prints a message and returns nullptr.
 */
extern "C" void* OfxGetPlugin(int nth) {
    if (nth != 0) return nullptr;

    std::cout << "[CorridorKey OFX] Plugin registered: "
              << PLUGIN_ID << " v"
              << PLUGIN_VERSION_MAJOR << "."
              << PLUGIN_VERSION_MINOR << std::endl;

    // TODO: Build and return OfxPlugin struct with:
    //   - pluginApi = kOfxImageEffectPluginApi
    //   - apiVersion = 1
    //   - pluginIdentifier = PLUGIN_ID
    //   - pluginVersionMajor/Minor
    //   - setHost callback
    //   - mainEntry callback → dispatches describe, createInstance, render, etc.

    return nullptr;  // Skeleton — will crash if host tries to use this
}

// ── Action handlers (to be implemented) ─────────────────────────────

/**
 * Describe action — declares inputs, outputs, and parameters.
 *
 * Inputs:
 *   - "Source" (Image) — green screen plate
 *   - "AlphaHint" (Image) — rough matte
 *
 * Parameters:
 *   - "despillStrength" (Double, 0-1, default 1.0)
 *   - "autoDespeckle" (Boolean, default true)
 *   - "despeckleSize" (Int, default 400)
 *   - "refinerScale" (Double, 0-2, default 1.0)
 *   - "inputIsLinear" (Boolean, default true)
 *
 * Output:
 *   - RGBA image (foreground + alpha)
 */

/**
 * Render action — called per frame.
 *
 * 1. Read input image from OFX source clip
 * 2. Read alpha hint from OFX mask clip
 * 3. Convert OFX pixel buffers → libtorch tensors
 * 4. Run CorridorKeyEffect::process_frame()
 * 5. Convert output tensors → OFX output pixel buffer
 */
