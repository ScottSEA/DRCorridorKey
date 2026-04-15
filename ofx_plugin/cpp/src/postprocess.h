/**
 * @file postprocess.h
 * @brief Postprocessing pipeline for CorridorKey model output (libtorch).
 *
 * Transforms raw model output (alpha + fg at model resolution) back
 * to the original resolution with correct color space and compositing:
 *   1. Resize alpha + fg back to original size
 *   2. Despill green from foreground
 *   3. Convert fg from sRGB → linear
 *   4. Premultiply fg by alpha
 *   5. Pack RGBA
 *
 * Matte cleanup (despeckle) is omitted from this header — it requires
 * OpenCV's connectedComponentsWithStats.  See corridorkey_effect.cpp
 * for the full pipeline with optional OpenCV integration.
 *
 * Matches ofx_plugin/core/postprocess.py.
 */

#pragma once

#include <torch/torch.h>
#include "color.h"

namespace corridorkey {

/**
 * Despill green from an RGB image.
 *
 * Caps green channel at avg(R, B), redistributes excess equally
 * to R and B to preserve luminance.
 *
 * @param image    [B, 3, H, W] float32 RGB
 * @param strength Despill intensity (0 = off, 1 = full)
 * @return         [B, 3, H, W] despilled image
 */
inline torch::Tensor despill(torch::Tensor image, float strength) {
    if (strength <= 0.0f) return image;

    auto r = image.select(1, 0);  // [B, H, W]
    auto g = image.select(1, 1);
    auto b = image.select(1, 2);

    auto limit = (r + b) / 2.0f;
    auto spill = (g - limit).clamp_min(0.0f);

    auto g_new = g - spill;
    auto r_new = r + spill * 0.5f;
    auto b_new = b + spill * 0.5f;

    auto despilled = torch::stack({r_new, g_new, b_new}, /*dim=*/1);

    if (strength < 1.0f) {
        return image * (1.0f - strength) + despilled * strength;
    }
    return despilled;
}

/**
 * Full postprocessing pipeline (without matte cleanup).
 *
 * @param alpha           [B, 1, H, W] raw model alpha
 * @param fg              [B, 3, H, W] raw model foreground (sRGB)
 * @param original_height Target output height
 * @param original_width  Target output width
 * @param despill_str     Despill strength (0-1)
 * @return                Tuple of (alpha [B,1,H,W], fg [B,3,H,W], rgba [B,4,H,W])
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> postprocess(
    torch::Tensor alpha,
    torch::Tensor fg,
    int original_height,
    int original_width,
    float despill_str = 1.0f
) {
    auto interp_opts = torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{original_height, original_width})
        .mode(torch::kBilinear)
        .align_corners(false);

    // 1. Resize back to original resolution
    alpha = torch::nn::functional::interpolate(alpha, interp_opts);
    fg = torch::nn::functional::interpolate(fg, interp_opts);

    // 2. Despill
    fg = despill(fg, despill_str);

    // 3. sRGB → linear
    auto fg_linear = srgb_to_linear(fg);

    // 4. Premultiply
    auto fg_premul = fg_linear * alpha;

    // 5. Pack RGBA
    auto rgba = torch::cat({fg_premul, alpha}, /*dim=*/1);  // [B, 4, H, W]

    return {alpha, fg, rgba};
}

}  // namespace corridorkey
