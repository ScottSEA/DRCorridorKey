/**
 * @file corridorkey_effect.cpp
 * @brief OFX image effect implementation for CorridorKey.
 *
 * Implements model loading and the per-frame inference pipeline:
 *   preprocess → model forward → postprocess
 */

#include "corridorkey_effect.h"
#include "preprocess.h"
#include "postprocess.h"

#include <torch/script.h>
#include <stdexcept>
#include <iostream>

namespace corridorkey {

CorridorKeyEffect::CorridorKeyEffect() = default;
CorridorKeyEffect::~CorridorKeyEffect() = default;

void CorridorKeyEffect::load_model(
    const std::string& model_path,
    const std::string& device
) {
    try {
        // Select device
        if (device == "cuda" && torch::cuda::is_available()) {
            device_ = torch::kCUDA;
        } else {
            device_ = torch::kCPU;
        }

        // Load the TorchScript model
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        model_loaded_ = true;

        std::cout << "[CorridorKey] Model loaded from " << model_path
                  << " on " << (device_ == torch::kCUDA ? "CUDA" : "CPU")
                  << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Failed to load CorridorKey model: ") + e.what()
        );
    }
}

bool CorridorKeyEffect::is_ready() const {
    return model_loaded_;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CorridorKeyEffect::process_frame(
    torch::Tensor image,
    torch::Tensor mask,
    int model_size,
    bool is_linear,
    float despill_str
) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded — call load_model() first");
    }

    // Save original dimensions for postprocessing
    int orig_h = image.size(0);
    int orig_w = image.size(1);

    // ── Preprocess ──────────────────────────────────────────────────
    // preprocess returns [1, 4, model_size, model_size]
    auto input = preprocess(
        image.to(device_),
        mask.to(device_),
        model_size,
        is_linear
    );

    // ── Model inference ─────────────────────────────────────────────
    torch::NoGradGuard no_grad;

    // The model returns a dict with "alpha" and "fg" keys
    auto output = model_.forward({input}).toGenericDict();
    auto alpha_raw = output.at("alpha").toTensor();  // [1, 1, H, W]
    auto fg_raw = output.at("fg").toTensor();        // [1, 3, H, W]

    // Apply sigmoid (model outputs logits)
    alpha_raw = torch::sigmoid(alpha_raw);
    fg_raw = torch::sigmoid(fg_raw);

    // ── Postprocess ─────────────────────────────────────────────────
    auto [alpha, fg, rgba] = postprocess(
        alpha_raw, fg_raw,
        orig_h, orig_w,
        despill_str
    );

    // Move back to CPU and remove batch dimension
    alpha = alpha.squeeze(0).squeeze(0).cpu();  // [H, W]
    fg = fg.squeeze(0).permute({1, 2, 0}).cpu();  // [H, W, 3]
    rgba = rgba.squeeze(0).permute({1, 2, 0}).cpu();  // [H, W, 4]

    return {alpha, fg, rgba};
}

}  // namespace corridorkey
