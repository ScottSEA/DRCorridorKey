/**
 * @file corridorkey_effect.h
 * @brief OFX image effect declaration for CorridorKey.
 *
 * Declares the CorridorKey OFX effect class which:
 *   - Loads the TorchScript model on first use
 *   - Converts OFX image buffers to/from libtorch tensors
 *   - Runs the preprocess → model → postprocess pipeline
 *   - Writes results back to OFX output buffers
 */

#pragma once

#include <string>
#include <memory>

#include <torch/script.h>

// Forward declarations — full OFX headers are included in the .cpp
struct OfxImageEffectHandle;

namespace corridorkey {

/**
 * CorridorKey OFX Image Effect.
 *
 * Lifecycle:
 *   1. Host creates the effect → constructor loads model
 *   2. Host calls render() per frame → preprocess + infer + postprocess
 *   3. Host destroys the effect → model unloaded
 *
 * The model is loaded lazily on first render() to avoid slowing down
 * project load.
 */
class CorridorKeyEffect {
public:
    CorridorKeyEffect();
    ~CorridorKeyEffect();

    // Non-copyable (model is GPU-resident)
    CorridorKeyEffect(const CorridorKeyEffect&) = delete;
    CorridorKeyEffect& operator=(const CorridorKeyEffect&) = delete;

    /**
     * Load the TorchScript model from a .pt file.
     *
     * @param model_path  Path to the corridorkey.pt TorchScript file.
     * @param device      "cuda" or "cpu"
     * @throws std::runtime_error if the file can't be loaded
     */
    void load_model(const std::string& model_path, const std::string& device = "cpu");

    /**
     * Check if the model is loaded and ready for inference.
     */
    bool is_ready() const;

    /**
     * Run inference on a single frame.
     *
     * @param image         [H, W, 3] float32 RGB input
     * @param mask          [H, W] float32 alpha hint
     * @param model_size    Model input resolution (default 2048)
     * @param is_linear     Whether input is in linear color space
     * @param despill_str   Despill strength (0-1)
     * @return              Tuple of (alpha [H,W], fg [H,W,3], rgba [H,W,4])
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_frame(
        torch::Tensor image,
        torch::Tensor mask,
        int model_size = 2048,
        bool is_linear = true,
        float despill_str = 1.0f
    );

private:
    torch::jit::script::Module model_;
    bool model_loaded_ = false;
    torch::Device device_ = torch::kCPU;
};

}  // namespace corridorkey
