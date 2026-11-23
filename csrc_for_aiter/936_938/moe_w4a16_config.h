// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

#include "moe_channel_mode.h"
#include "moe_w4a16_prefill_pipeline.h"

template <typename T>
struct GemmParams_w4a16
{
    GemmParams_w4a16(const T *ptr_A,
                     const uint32_t *ptr_B0,
                     T *ptr_C,
                     uint32_t *ptr_B_zeros,
                     T *ptr_B_scale,
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
          is_marlin(is_marlin)
    {
    }

    const T *ptr_A;                         // input
    const uint32_t *ptr_B0;                 // weight
    T *ptr_C;                               // output
    uint32_t *ptr_B_zeros;                  // input scale
    T *ptr_B_scale;                         // weight scale
    const float *topk_weights;              // topk weights
    const int32_t *sorted_token_ids;        // sorted token ids
    const int32_t *expert_ids;              // expert ids
    const int32_t num_tokens_post_pad;      // num tokens after padding
    const int32_t *num_tokens_post_pad_ptr; // num tokens after padding
    uint32_t size_m;                        // input size m
    uint32_t size_n;                        // output size n
    uint32_t size_k;                        // input size k
    uint32_t sorted_token_lens;             // sorted token length
    uint32_t top_k;                         // top k
    uint32_t delta;                         // delta
    bool is_marlin;                         // 是否使用weight重排
};

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w4a16_first_stage_decode(const GemmParams_w4a16<T> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w4a16_second_stage_decode(const GemmParams_w4a16<T> &params);

// BM routing rows, BN physical columns per chunk, NLoop chunks per CTA (A stays in
// LDS when NLoop > 1). The pipeline derives wave layout and scheduling at compile time.
// GS selects the scale layout. Legacy decode always uses group-32 scales.
template <int BM, int BN, int NLoop, W4A16Pipeline Pipeline, W4A16Epilogue Epilogue, typename T, int GS = 32>
void launch_moe_w4a16_prefill(const GemmParams_w4a16<T> &params);

template <int BM, int BN, typename T, int GS = 32>
void launch_moe_w4a16_g2_bstat(const GemmParams_w4a16<T> &params);

template <int BM, int BN, int PrefetchDepth, W4A16Epilogue Epilogue, typename T, int GS = 32>
void launch_moe_w4a16_decode(const GemmParams_w4a16<T> &params);

// The unfused twin also describes the epilogue: a nonzero twin means the C
// buffer holds half of GEMM1's output columns.
template <typename T>
struct W4A16Mode
{
    int id;
    int unfused_mode;
    void (*launch)(const GemmParams_w4a16<T> &);
};

template <int Stage, int BM, int BN, int BK, int WM, int WN, int WK, int ST, typename T>
void launch_w4a16_legacy(const GemmParams_w4a16<T> &p)
{
    TORCH_CHECK(p.size_n > 0 && p.size_n % BN == 0,
                "moe_w4a16 legacy decode: N must be a positive multiple of ", BN, ", got ", p.size_n);
    // Stage 1 pipelines load 4 blocks for K=256 and 8 blocks otherwise (BK=64).
    const bool valid_k = ST == 1 ? (p.size_k == 256 || p.size_k % (8 * BK) == 0)
                                 : p.size_k % (2 * BK) == 0;
    TORCH_CHECK(p.size_k > 0 && valid_k, "moe_w4a16 legacy decode: unsupported K=", p.size_k,
                " for BK=", BK, ", ST=", ST);
    if constexpr (Stage == 1)
        launch_moe_w4a16_first_stage_decode<BM, BN, BK, WM, WN, WK, ST>(p);
    else
        launch_moe_w4a16_second_stage_decode<BM, BN, BK, WM, WN, WK, ST>(p);
}

template <typename T>
bool w4a16_bstat_output_fits(const GemmParams_w4a16<T> &p)
{
    constexpr uint64_t max_bytes = std::numeric_limits<int32_t>::max() - 1;
    const uint64_t rows = static_cast<uint64_t>(p.size_m) * p.top_k;
    return p.size_n == 0 || rows <= max_bytes / sizeof(T) / p.size_n;
}

template <int BM, int BN, typename T, int GS>
void launch_w4a16_bstat_or_streamed(const GemmParams_w4a16<T> &p)
{
    if (p.size_k == 256 && w4a16_bstat_output_fits(p))
        launch_moe_w4a16_g2_bstat<BM, BN, T, GS>(p);
    else
        launch_moe_w4a16_prefill<BM, w4a16_prefill::BstatFallbackBN, w4a16_prefill::BstatFallbackNLoop,
                                w4a16_prefill::BstatFallbackPipeline, W4A16Epilogue::DownWeighted, T, GS>(p);
}

#include "moe_w4a16_mode_table.h"

template <typename T, size_t N>
const W4A16Mode<T> *find_w4a16_mode(const W4A16Mode<T> (&table)[N], int bm, int profile)
{
    // Stage and family select the table. Its remaining fields are sorted by BM, then profile.
    const int key = (bm / 16) * 1000 + profile;
    const auto *end = table + N;
    const auto *it = std::lower_bound(table, end, key, [](const W4A16Mode<T> &entry, int value) {
        return entry.id % 100000 < value;
    });
    return it != end && it->id % 100000 == key ? it : nullptr;
}
// NOTE(build-diff): Standalone CMake builds libmoe_w4a16.so with -ffast-math and post-misched enabled, whereas AITER JIT (core.py) builds module_moe_c_kernel with -mllvm -enable-post-misched=0, -mllvm --lsr-drop-solution=1, -fno-offload-uniform-block and without -ffast-math (+2.6%-7.8% slower on some prefill modes in gfx938 A/B).
