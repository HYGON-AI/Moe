// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef MOE_W8A8_CONFIGS_HIP_H
#define MOE_W8A8_CONFIGS_HIP_H

#include <unordered_map>
#include <functional>

#define MOE_W8A8_MODE(STAGE, SORTING, FAMILY, BM, PROFILE) \
    ((STAGE) * 10000000 + (SORTING) * 1000000 + (FAMILY) * 100000 + \
     ((BM) / 16) * 1000 + (PROFILE))

template <typename T, typename T_hidden = bhalf_t>
struct GemmParams
{
    GemmParams(const T *ptr_A,
               const T *ptr_B0,
               T_hidden *ptr_C,
               float *ptr_A_scale,
               float *ptr_B_scale,
               const float *topk_weights,
               const int32_t *sorted_token_ids,
               const int32_t *expert_ids,
               const int32_t num_tokens_post_pad,
               const int32_t *num_tokens_post_pad_ptr,
               uint32_t size_m,
               uint32_t size_n,
               uint32_t size_k,
               uint32_t stride_asm,
               uint32_t stride_ask,
               uint32_t stride_bse,
               uint32_t stride_bsn,
               uint32_t stride_bsk,
               uint32_t sorted_token_lens,
               uint32_t top_k,
               uint32_t real_topk,
               bool is_marlin,
               bool tensorwise_scale = false,
               uint32_t real_size_k_in = 0)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
          ptr_B_scale(ptr_B_scale),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          real_size_k(real_size_k_in == 0 ? size_k : real_size_k_in),
          stride_asm(stride_asm),
          stride_ask(stride_ask),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          real_topk(real_topk),
          is_marlin(is_marlin),
          tensorwise_scale(tensorwise_scale)
    {
    }

    const T *ptr_A;                         // input
    const T *ptr_B0;                        // weight
    T_hidden *ptr_C;                        // output
    float *ptr_A_scale;                     // input scale
    float *ptr_B_scale;                     // weight scale
    const float *topk_weights;              // topk weights
    const int32_t *sorted_token_ids;        // sorted token ids
    const int32_t *expert_ids;              // expert ids
    const int32_t num_tokens_post_pad;      // num tokens after padding
    const int32_t *num_tokens_post_pad_ptr; // num tokens after padding
    uint32_t size_m;                        // input size m
    uint32_t size_n;                        // output size n
    uint32_t size_k;                        // input size k
    uint32_t real_size_k;                   // real A row stride before static K padding
    uint32_t stride_asm;                    // input scale stride m
    uint32_t stride_ask;                    // input scale stride k
    uint32_t stride_bse;                    // weight scale stride expert
    uint32_t stride_bsn;                    // weight scale stride n
    uint32_t stride_bsk;                    // weight scale stride k
    uint32_t sorted_token_lens;             // sorted token length
    uint32_t top_k;                         // top k
    uint32_t real_topk;                     // real_topk
    bool is_marlin;                         // 是否使用weight重排
    bool tensorwise_scale;                  // W8A8 tensorwise: B scale shape is (E, 1, 1)
};

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_decode(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_decode(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_up_int8_bk64_nloop(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_down_int8_bk64_nloop(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_decode_fp8(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_decode_fp8(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill_fp8(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill_fp8_nloop(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill_fp8(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill_fp8_nloop(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_up_fp8_bk64_nloop(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_down_fp8_bk64_nloop(const GemmParams<T, T_hidden> &params);

template <typename scalar_t>
using KernelFunc = std::function<void(const GemmParams<char, scalar_t> &)>;

#define MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP_ENTRY(BM, N_LOOP_NUM) \
    {MOE_W8A8_MODE(1, 0, 2, BM, 100 + (N_LOOP_NUM)), [](const GemmParams<char, scalar_t> &p) \
     { launch_moe_w8a8_prefill_up_int8_bk64_nloop<BM, 128, 64, BM, 32, 64, 2, N_LOOP_NUM>(p); }}

#define MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP12_ENTRIES(BM) \
    MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP_ENTRY(BM, 1), \
    MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP_ENTRY(BM, 2)

#define MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP_ENTRY(BM, N_LOOP_NUM) \
    {MOE_W8A8_MODE(2, 0, 2, BM, 100 + (N_LOOP_NUM)), [](const GemmParams<char, scalar_t> &p) \
     { launch_moe_w8a8_prefill_down_int8_bk64_nloop<BM, 128, 64, BM, 32, 64, 2, N_LOOP_NUM>(p); }}

#define MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP4_ENTRIES \
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP_ENTRY(32, 4), \
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP_ENTRY(48, 4), \
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP_ENTRY(64, 4)

#define MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP12_ENTRIES(BM) \
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP_ENTRY(BM, 1), \
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP_ENTRY(BM, 2)

template <typename scalar_t>
// These maps consume CK-sorting packed token metadata.  Keep the layout in
// the name: a moe_align mode must never silently alias one of these entries.
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_decode_ck_sorting = {

    {MOE_W8A8_MODE(1, 0, 0, 16, 101), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode<16, 64, 1024, 16, 64, 128, 1>(p); }},
    {MOE_W8A8_MODE(1, 0, 0, 16, 102), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode<16, 16, 512, 16, 16, 128, 4>(p); }},
    {MOE_W8A8_MODE(1, 0, 0, 16, 103), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode<16, 16, 1024, 16, 16, 256, 4>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_decode_ck_sorting = {
    {MOE_W8A8_MODE(2, 0, 0, 16, 101), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 32, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 0, 16, 102), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 64, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 0, 16, 103), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 64, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 0, 16, 104), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 128, 64, 2>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_prefill_ck_sorting = {

    {MOE_W8A8_MODE(1, 0, 1, 32, 201), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 128, 64, 32, 128, 64, 1>(p); }},

    {MOE_W8A8_MODE(1, 0, 1, 32, 202), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 128, 64, 32, 128, 64, 2>(p); }},
    {MOE_W8A8_MODE(1, 0, 1, 32, 203), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 256, 64, 32, 128, 64, 2>(p); }},

    {MOE_W8A8_MODE(1, 0, 1, 32, 204), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 32, 64, 64, 1>(p); }},

    MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP12_ENTRIES(32),
    MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP12_ENTRIES(48),
    MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP12_ENTRIES(64),

};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_prefill_ck_sorting = {
    {MOE_W8A8_MODE(2, 0, 1, 32, 201), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill<32, 128, 64, 32, 32, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 1, 64, 202), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill<64, 128, 64, 64, 32, 64, 2>(p); }},

    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP4_ENTRIES,
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP12_ENTRIES(32),
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP12_ENTRIES(48),
    MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP12_ENTRIES(64),

};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_decode_fp8_ck_sorting = {
    {MOE_W8A8_MODE(1, 0, 0, 16, 101), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_fp8<16, 64, 1024, 16, 64, 128, 1>(p); }},
    {MOE_W8A8_MODE(1, 0, 0, 16, 102), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_fp8<16, 16, 512, 16, 16, 128, 4>(p); }},
    {MOE_W8A8_MODE(1, 0, 0, 16, 103), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_fp8<16, 64, 256, 16, 64, 128, 2>(p); }},
    {MOE_W8A8_MODE(1, 0, 0, 32, 101), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_fp8<32, 128, 64, 32, 32, 64, 2>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_decode_fp8_ck_sorting = {
    {MOE_W8A8_MODE(2, 0, 0, 16, 101), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 128, 64, 16, 32, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 0, 16, 102), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 256, 64, 16, 64, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 0, 16, 103), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 512, 64, 16, 64, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 0, 16, 104), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 512, 64, 16, 128, 64, 2>(p); }},
    {MOE_W8A8_MODE(2, 0, 0, 32, 101), [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<32, 128, 64, 32, 32, 64, 2>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_prefill_fp8_moe_align = {
        {MOE_W8A8_MODE(1, 1, 1, 32, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 64, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 80, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 96, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 128, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 32, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 32, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 5>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 48, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 48, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 48, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 5>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 64, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 64, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 5>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 80, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 80, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 80, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 5>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 96, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 96, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 96, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 5>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 128, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 1, 1, 128, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_first_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 5>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_prefill_fp8_ck_sorting = {
        {MOE_W8A8_MODE(1, 0, 2, 32, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 48, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 64, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 32, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 32, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<32, 128, 64, 32, 32, 64, 2, 5>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 48, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 48, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<48, 128, 64, 48, 32, 64, 2, 5>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 64, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(1, 0, 2, 64, 105), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_up_fp8_bk64_nloop<64, 128, 64, 64, 32, 64, 2, 5>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_prefill_fp8_moe_align = {
        {MOE_W8A8_MODE(2, 1, 1, 32, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 64, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 80, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 96, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 128, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 32, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 32, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 48, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 48, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 48, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 64, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 64, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 80, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 80, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 96, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 96, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 128, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 1, 1, 128, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_second_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 4>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_prefill_fp8_ck_sorting = {
        {MOE_W8A8_MODE(2, 0, 2, 32, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<32, 128, 64, 32, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 48, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<48, 128, 64, 48, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 64, 104), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<64, 128, 64, 64, 32, 64, 2, 4>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 32, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 32, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 48, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 48, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 64, 101), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},

        {MOE_W8A8_MODE(2, 0, 2, 64, 102), [](const GemmParams<char, scalar_t> &p)
         { launch_moe_w8a8_prefill_down_fp8_bk64_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},
};

#undef MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP12_ENTRIES
#undef MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP4_ENTRIES
#undef MOE_W8A8_INT8_PREFILL_DOWN_BK64_NLOOP_ENTRY
#undef MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP12_ENTRIES
#undef MOE_W8A8_INT8_PREFILL_UP_BK64_NLOOP_ENTRY
#undef MOE_W8A8_MODE

#endif
