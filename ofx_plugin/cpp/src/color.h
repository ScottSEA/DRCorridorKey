/**
 * @file color.h
 * @brief sRGB ↔ linear color conversion for libtorch tensors.
 *
 * Pure-header implementation of the IEC 61966-2-1 piecewise sRGB
 * transfer function.  Matches the Python reference in
 * ofx_plugin/core/color.py (validated by cross-validation tests).
 *
 * Usage:
 *     auto linear = corridorkey::srgb_to_linear(srgb_tensor);
 *     auto srgb = corridorkey::linear_to_srgb(linear_tensor);
 */

#pragma once

#include <torch/torch.h>

namespace corridorkey {

/**
 * Convert sRGB values to linear light.
 *
 * Piecewise transfer function:
 *   if srgb <= 0.04045:  linear = srgb / 12.92
 *   else:                linear = ((srgb + 0.055) / 1.055) ^ 2.4
 *
 * @param x  Tensor of any shape, values in [0, 1].
 *           Negative values are clamped to 0.
 * @return   Tensor of the same shape in linear space.
 */
inline torch::Tensor srgb_to_linear(torch::Tensor x) {
    x = x.clamp_min(0.0f);
    auto mask = x <= 0.04045f;
    auto low = x / 12.92f;
    auto high = torch::pow((x + 0.055f) / 1.055f, 2.4f);
    return torch::where(mask, low, high);
}

/**
 * Convert linear light values to sRGB.
 *
 * Piecewise transfer function:
 *   if linear <= 0.0031308:  srgb = linear * 12.92
 *   else:                    srgb = 1.055 * linear^(1/2.4) - 0.055
 *
 * @param x  Tensor of any shape, values in [0, 1].
 *           Negative values are clamped to 0.
 * @return   Tensor of the same shape in sRGB space.
 */
inline torch::Tensor linear_to_srgb(torch::Tensor x) {
    x = x.clamp_min(0.0f);
    auto mask = x <= 0.0031308f;
    auto low = x * 12.92f;
    auto high = 1.055f * torch::pow(x, 1.0f / 2.4f) - 0.055f;
    return torch::where(mask, low, high);
}

}  // namespace corridorkey
