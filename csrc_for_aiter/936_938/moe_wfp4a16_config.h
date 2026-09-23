// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

template <typename T>
struct GemmParams_w4a16
{
    GemmParams_w4a16(const T *ptr_A,
                     const uint32_t *ptr_B0,
                     T *ptr_C,
                     uint32_t *ptr_B_zeros,
                     T *ptr_B_scale,
                     const uint8_t *ptr_B_scale_u8,
                     const float *topk_weights,
                     const int32_t *sorted_token_ids,
                     const int32_t *expert_ids,
                     const int32_t num_tokens_post_pad,
                     const int32_t *num_tokens_post_pad_ptr,
                     uint32_t size_m,
                     uint32_t size_n,
                     uint32_t size_k,
                     uint32_t sorted_token_lens,
                     uint32_t top_k,
                     uint32_t delta,
                     bool is_marlin)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_B_zeros(ptr_B_zeros),
          ptr_B_scale(ptr_B_scale),
          ptr_B_scale_u8(ptr_B_scale_u8),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          delta(delta),
          is_marlin(is_marlin) {}

    const T *ptr_A;
    const uint32_t *ptr_B0;
    T *ptr_C;
    uint32_t *ptr_B_zeros;
    T *ptr_B_scale;
    const uint8_t *ptr_B_scale_u8;
    const float *topk_weights;
    const int32_t *sorted_token_ids;
    const int32_t *expert_ids;
    const int32_t num_tokens_post_pad;
    const int32_t *num_tokens_post_pad_ptr;
    uint32_t size_m;
    uint32_t size_n;
    uint32_t size_k;
    uint32_t sorted_token_lens;
    uint32_t top_k;
    uint32_t delta;
    bool is_marlin;
};

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          typename T, bool FP4_E2M1 = false, bool FP4_K_SPLIT = false>
void launch_moe_w4a16_first_stage_decode(const GemmParams_w4a16<T> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          typename T, bool FP4_E2M1 = false>
void launch_moe_w4a16_second_stage_decode(const GemmParams_w4a16<T> &params);

template <typename scalar_t>
using KernelFunc_w4a16 = std::function<void(const GemmParams_w4a16<scalar_t> &)>;

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm1_prefill_wfp4a16 = {
    {514, [](const GemmParams_w4a16<scalar_t> &p)
     { launch_moe_w4a16_first_stage_decode<128, 128, 64, 64, 64, 64, 1, scalar_t, true>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm2_prefill_wfp4a16 = {
    {511, [](const GemmParams_w4a16<scalar_t> &p)
     { launch_moe_w4a16_second_stage_decode<128, 128, 64, 64, 64, 64, 1, scalar_t, true>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm1_decode_wfp4a16 = {
    {114, [](const GemmParams_w4a16<scalar_t> &p)
     { launch_moe_w4a16_first_stage_decode<16, 64, 64, 16, 32, 64, 1, scalar_t, true>(p); }},
    {115, [](const GemmParams_w4a16<scalar_t> &p)
     { launch_moe_w4a16_first_stage_decode<16, 32, 128, 16, 32, 64, 1, scalar_t, true, true>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm2_decode_wfp4a16 = {
    {111, [](const GemmParams_w4a16<scalar_t> &p)
     { launch_moe_w4a16_second_stage_decode<16, 64, 64, 16, 32, 64, 1, scalar_t, true>(p); }},
    {112, [](const GemmParams_w4a16<scalar_t> &p)
     { launch_moe_w4a16_second_stage_decode<16, 128, 64, 16, 32, 64, 1, scalar_t, true>(p); }},
    {113, [](const GemmParams_w4a16<scalar_t> &p)
     { launch_moe_w4a16_second_stage_decode<16, 256, 64, 16, 32, 64, 1, scalar_t, true>(p); }},
};
