// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <unordered_map>
#include <functional>
#include <type_traits>
#include <vector>
#include <algorithm>

// HIPC mode format shared with W8A8: S F K MM VVV (see csrc/moe_c_common.h).
// W4A16 kernels consume moe_align ids, so SORTING is always 1; FAMILY 0 is decode.
#define MOE_W4A16_MODE(STAGE, SORTING, FAMILY, BM, PROFILE) \
    ((STAGE) * 10000000 + (SORTING) * 1000000 + (FAMILY) * 100000 + \
     ((BM) / 16) * 1000 + (PROFILE))

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

// moe_w4a16_prefill.h: BM rows, WARPS warps along N, NN 16-column tiles per warp, NCH column
// chunks per block (A resident in LDS, size_k == 256 only), ST LDS stages, PREFETCH_A fragments,
// GEMM2 = down projection (scales rows by topk_weights).
template <int BM, int WARPS, int NN, int NCH, int ST, bool PREFETCH_A, bool GEMM2, int WPE, int SCM, typename T, bool SILU = false,
          int AH = 1>
void launch_moe_w4a16_prefill(const GemmParams_w4a16<T> &params);

template <typename scalar_t>
using KernelFunc_w4a16 = std::function<void(const GemmParams_w4a16<scalar_t> &)>;

// Prefill modes (FAMILY 1).  Profile segments: 2xx unfused, 3xx GEMM1 with fused silu_and_mul
// (output [M, topk, N / 2]); a fused mode minus 100 is its unfused twin.
inline bool w4a16_prefill_fused_silu(int64_t mode)
{
    return mode / 100000 % 10 == 1 && mode % 1000 / 100 == 3;
}

// Kernel registration.  BM is written once: it is both the BM field of the map key and the first
// template argument of the launcher, so a key can never disagree with its kernel.  The map keys
// (all four maps) are the only list of valid modes; moe_c_w4a16_modes() exports them to the tuner.
#define W4A16_UNPAREN(...) __VA_ARGS__
#define W4A16_DECODE(STAGE, PROFILE, BM, BN, BK, WM, WN, WK, ST, FN)                              \
    {MOE_W4A16_MODE(STAGE, 1, 0, BM, PROFILE), [](const GemmParams_w4a16<scalar_t> &p)             \
     { FN<BM, BN, BK, WM, WN, WK, ST, scalar_t>(p); }}
// ARGS: launch_moe_w4a16_prefill template arguments after BM, as "(WARPS, NN, NCH, ST, PREFETCH_A,
// GEMM2, WPE, SCM, scalar_t[, SILU[, AH]])"; FP16_ARGS replaces them for fp16 where the bf16
// instance spills registers.
#define W4A16_PREFILL(STAGE, PROFILE, BM, ARGS) W4A16_PREFILL_FP16(STAGE, PROFILE, BM, ARGS, ARGS)
#define W4A16_PREFILL_FP16(STAGE, PROFILE, BM, ARGS, FP16_ARGS)                                   \
    {MOE_W4A16_MODE(STAGE, 1, 1, BM, PROFILE), [](const GemmParams_w4a16<scalar_t> &p)             \
     {                                                                                            \
       if constexpr (std::is_same_v<scalar_t, half>)                                              \
         launch_moe_w4a16_prefill<BM, W4A16_UNPAREN FP16_ARGS>(p);                                \
       else                                                                                       \
         launch_moe_w4a16_prefill<BM, W4A16_UNPAREN ARGS>(p);                                     \
     }}

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm1_prefill_w4a16 = {
    W4A16_PREFILL(1, 201, 48, (4, 2, 1, 2, true, false, 0, 4, scalar_t)),
    W4A16_PREFILL(1, 201, 64, (8, 2, 1, 2, true, false, 0, 4, scalar_t)),
    W4A16_PREFILL(1, 202, 64, (8, 4, 1, 2, false, false, 4, 4, scalar_t)),
    // BM80 / BM128: A staged per half K step (two blocks per CU).
    W4A16_PREFILL(1, 201, 80, (8, 2, 1, 2, false, false, 4, 4, scalar_t, false, 2)),
    W4A16_PREFILL(1, 201, 128, (8, 2, 1, 2, false, false, 4, 4, scalar_t, false, 2)),
    // Larger row tiles reuse each dequantized B fragment across more MMACs.
    // Half-step staging keeps BM256 within the 64 KiB LDS budget.
    W4A16_PREFILL(1, 201, 160, (8, 2, 1, 2, false, false, 4, 4, scalar_t, false, 2)),
    W4A16_PREFILL(1, 201, 192, (8, 2, 1, 2, false, false, 4, 4, scalar_t, false, 2)),
    W4A16_PREFILL(1, 201, 256, (8, 2, 1, 2, false, false, 4, 4, scalar_t, false, 2)),
    // These LDS sizes permit one block/CU, so a two-wave register budget avoids
    // forcing the larger accumulator tiles to spill for unattainable occupancy.
    W4A16_PREFILL(1, 202, 160, (8, 2, 1, 2, false, false, 2, 4, scalar_t, false, 2)),
    W4A16_PREFILL(1, 202, 192, (8, 2, 1, 2, false, false, 2, 4, scalar_t, false, 2)),
    W4A16_PREFILL(1, 202, 256, (8, 2, 1, 2, false, false, 2, 4, scalar_t, false, 2)),
    // Fused silu_and_mul twins (profile + 100).
    W4A16_PREFILL(1, 301, 48, (4, 2, 1, 2, true, false, 0, 4, scalar_t, true)),
    W4A16_PREFILL(1, 302, 64, (8, 4, 1, 2, false, false, 4, 4, scalar_t, true)),
    W4A16_PREFILL(1, 301, 80, (8, 2, 1, 2, false, false, 4, 4, scalar_t, true, 2)),
    W4A16_PREFILL(1, 301, 128, (8, 2, 1, 2, false, false, 4, 4, scalar_t, true, 2)),
    W4A16_PREFILL(1, 301, 160, (8, 2, 1, 2, false, false, 4, 4, scalar_t, true, 2)),
    W4A16_PREFILL(1, 301, 192, (8, 2, 1, 2, false, false, 4, 4, scalar_t, true, 2)),
    W4A16_PREFILL(1, 301, 256, (8, 2, 1, 2, false, false, 4, 4, scalar_t, true, 2)),
    W4A16_PREFILL(1, 302, 160, (8, 2, 1, 2, false, false, 2, 4, scalar_t, true, 2)),
    W4A16_PREFILL(1, 302, 192, (8, 2, 1, 2, false, false, 2, 4, scalar_t, true, 2)),
    W4A16_PREFILL(1, 302, 256, (8, 2, 1, 2, false, false, 2, 4, scalar_t, true, 2)),
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm2_prefill_w4a16 = {
    W4A16_PREFILL(2, 201, 48, (8, 2, 4, 2, true, true, 0, 4, scalar_t)),
    W4A16_PREFILL(2, 201, 64, (8, 2, 2, 2, true, true, 0, 4, scalar_t)),
    W4A16_PREFILL(2, 202, 64, (8, 2, 4, 2, true, true, 0, 4, scalar_t)),
    // 203..205: NN4 / full-width variants (bf16); the fp16 NN4 instances spill registers, so fp16
    // falls back to the 202 instance (or the 256-VGPR-budget variant for 205).
    W4A16_PREFILL_FP16(2, 203, 64, (8, 4, 4, 2, false, true, 4, 2, scalar_t),
                                   (8, 2, 4, 2, true, true, 0, 4, scalar_t)),
    W4A16_PREFILL_FP16(2, 204, 64, (8, 4, 2, 2, false, true, 4, 2, scalar_t),
                                   (8, 2, 4, 2, true, true, 0, 4, scalar_t)),
    // NN2, all 2048 columns per block (8 chunks over the resident A).
    W4A16_PREFILL_FP16(2, 205, 64, (8, 2, 8, 2, false, true, 4, 4, scalar_t),
                                   (8, 2, 8, 2, false, true, 3, 4, scalar_t)),
    // Resident-A variants: BM128 uses the full 64 KiB LDS allocation.
    W4A16_PREFILL(2, 205, 80, (8, 2, 8, 2, false, true, 3, 4, scalar_t)),
    W4A16_PREFILL(2, 205, 128, (8, 2, 8, 2, false, true, 2, 4, scalar_t)),
    W4A16_PREFILL(2, 203, 128, (8, 4, 4, 2, false, true, 2, 2, scalar_t)),
    // Beyond BM128, stream A in half steps to stay within 64 KiB LDS.
    W4A16_PREFILL(2, 206, 256, (8, 2, 1, 2, false, true, 2, 4, scalar_t, false, 2)),
    // Restore sixteen resident waves with one BM128 block (1024 threads).
    W4A16_PREFILL(2, 207, 128, (16, 2, 4, 2, false, true, 4, 4, scalar_t)),
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm1_decode_w4a16 = {
    W4A16_DECODE(1, 101, 16, 16, 512, 16, 16, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 102, 16, 32, 512, 16, 32, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 103, 16, 32, 256, 16, 16, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 104, 16, 256, 128, 16, 64, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 105, 16, 128, 256, 16, 64, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 101, 32, 128, 256, 32, 64, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 102, 32, 256, 128, 32, 64, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 101, 48, 128, 128, 48, 32, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 102, 48, 256, 128, 48, 64, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 101, 64, 128, 128, 64, 32, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 102, 64, 256, 128, 64, 64, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 101, 128, 128, 128, 64, 64, 128, 2, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 102, 128, 256, 64, 64, 64, 64, 1, launch_moe_w4a16_first_stage_decode),
    W4A16_DECODE(1, 103, 128, 128, 64, 64, 64, 64, 1, launch_moe_w4a16_first_stage_decode),
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a16<scalar_t>> kernel_maps_gemm2_decode_w4a16 = {
    W4A16_DECODE(2, 101, 16, 64, 128, 16, 16, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 102, 16, 128, 128, 16, 32, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 103, 16, 256, 128, 16, 64, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 101, 32, 256, 128, 32, 64, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 102, 32, 128, 256, 32, 64, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 101, 48, 128, 128, 48, 32, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 102, 48, 256, 128, 48, 64, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 101, 64, 128, 128, 64, 32, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 102, 64, 256, 128, 64, 64, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 101, 128, 128, 128, 64, 64, 128, 2, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 102, 128, 256, 64, 64, 64, 64, 1, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 103, 128, 128, 64, 64, 64, 64, 1, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 104, 128, 384, 64, 64, 64, 64, 1, launch_moe_w4a16_second_stage_decode),
    W4A16_DECODE(2, 101, 256, 128, 128, 64, 64, 128, 2, launch_moe_w4a16_second_stage_decode),
};

// Every registered mode of one stage (decode and prefill), sorted.
inline std::vector<int64_t> w4a16_registered_modes(int stage)
{
    std::vector<int64_t> modes;
    auto add = [&](const auto &map) { for (const auto &kv : map) modes.push_back(kv.first); };
    if (stage == 1) { add(kernel_maps_gemm1_decode_w4a16<half>); add(kernel_maps_gemm1_prefill_w4a16<half>); }
    else            { add(kernel_maps_gemm2_decode_w4a16<half>); add(kernel_maps_gemm2_prefill_w4a16<half>); }
    std::sort(modes.begin(), modes.end());
    return modes;
}
