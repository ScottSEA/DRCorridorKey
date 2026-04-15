/**
 * @file preprocess.h
 * @brief Preprocessing pipeline for the CorridorKey model (libtorch).
 *
 * Transforms a raw image + mask into the 4-channel tensor the model
 * expects:
 *   1. Resize to model_size × model_size
 *   2. If linear input, convert to sRGB
 *   3. Normalize with ImageNet mean/std
 *   4. Concatenate image (3ch) + mask (1ch) → 4ch
 *   5. Convert HWC → NCHW layout for PyTorch
 *
 * Matches ofx_plugin/core/preprocess.py.
 */

#pragma once

#include <torch/torch.h>
#include "color.h"

namespace corridorkey {

// ImageNet normalization constants (RGB order)
constexpr float IMAGENET_MEAN[] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[]  = {0.229f, 0.224f, 0.225f};

/**
 * Normalize a [B, 3, H, W] tensor with ImageNet mean/std.
 *
 * Each channel c: normalized[c] = (input[c] - mean[c]) / std[c]
 */
inline torch::Tensor imagenet_normalize(torch::Tensor x) {
    // Create mean/std tensors shaped [1, 3, 1, 1] for broadcasting
    auto mean = torch::tensor({IMAGENET_MEAN[0], IMAGENET_MEAN[1], IMAGENET_MEAN[2]})
                    .view({1, 3, 1, 1}).to(x.device());
    auto std = torch::tensor({IMAGENET_STD[0], IMAGENET_STD[1], IMAGENET_STD[2]})
                   .view({1, 3, 1, 1}).to(x.device());
    return (x - mean) / std;
}

/**
 * Full preprocessing pipeline.
 *
 * @param image       [H, W, 3] float32 RGB in [0, 1]
 * @param mask        [H, W] float32 alpha hint in [0, 1]
 * @param model_size  Target square resolution for the model
 * @param is_linear   If true, convert image from linear → sRGB first
 * @return            [1, 4, model_size, model_size] float32 tensor
 */
inline torch::Tensor preprocess(
    torch::Tensor image,
    torch::Tensor mask,
    int model_size,
    bool is_linear
) {
    // 1. Resize — use interpolate on NCHW layout
    // Convert HWC → CHW → NCHW
    auto img = image.permute({2, 0, 1}).unsqueeze(0);  // [1, 3, H, W]
    auto msk = mask.unsqueeze(0).unsqueeze(0);          // [1, 1, H, W]

    img = torch::nn::functional::interpolate(
        img,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{model_size, model_size})
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    msk = torch::nn::functional::interpolate(
        msk,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{model_size, model_size})
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    // 2. Color space conversion
    if (is_linear) {
        img = linear_to_srgb(img);
    }

    // 3. ImageNet normalization
    img = imagenet_normalize(img);

    // 4. Concatenate image + mask → 4 channels [1, 4, H, W]
    return torch::cat({img, msk}, /*dim=*/1);
}

}  // namespace corridorkey
