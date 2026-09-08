// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef MOE_W8A8_CONFIGS_HIP_H
#define MOE_W8A8_CONFIGS_HIP_H

#include <unordered_map>
#include <functional>
#define MOE_THRESHOLD 128

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
void launch_moe_w8a8_prefill_up_int8_bk128_nloop(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_down_int8_bk128_nloop(const GemmParams<T, T_hidden> &params);

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
void launch_moe_w8a8_prefill_up_fp8_bk128_nloop(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_down_fp8_bk128_nloop(const GemmParams<T, T_hidden> &params);

template <typename scalar_t>
using KernelFunc = std::function<void(const GemmParams<char, scalar_t> &)>;

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_decode = {

    {98, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode<16, 64, 1024, 16, 64, 128, 1>(p); }},
    {121, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode<16, 16, 512, 16, 16, 128, 4>(p); }},
    {122, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode<16, 16, 1024, 16, 16, 256, 4>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_decode = {
    {38, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 32, 64, 2>(p); }},
    {42, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 64, 64, 2>(p); }},
    {43, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 64, 64, 2>(p); }},
    {46, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 128, 64, 2>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_prefill = {

    {146, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 128, 64, 32, 128, 64, 1>(p); }},

    {159, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 128, 64, 32, 128, 64, 2>(p); }},
    {160, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 256, 64, 32, 128, 64, 2>(p); }},

    {183, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 32, 64, 64, 1>(p); }},

    {200321, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_int8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},
    {200322, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_int8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},
    {200481, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_int8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},
    {200482, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_int8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},
    {200641, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_int8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},
    {200642, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_int8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},

};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_prefill = {
    {86, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill<32, 128, 64, 32, 32, 64, 2>(p); }},
    {166, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill<64, 128, 64, 64, 32, 64, 2>(p); }},

    {200324, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 4>(p); }},
    {200484, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 4>(p); }},
    {200644, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 4>(p); }},
    {201321, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},
    {201322, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},
    {201481, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},
    {201482, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},
    {201641, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},
    {201642, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_int8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},

};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_decode_fp8 = {
    {60098, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_fp8<16, 64, 1024, 16, 64, 128, 1>(p); }},
    {60121, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_fp8<16, 16, 512, 16, 16, 128, 4>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_decode_fp8 = {
    {60038, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 128, 64, 16, 32, 64, 2>(p); }},
    {60042, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 256, 64, 16, 64, 64, 2>(p); }},
    {60043, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 512, 64, 16, 64, 64, 2>(p); }},
    {60046, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_fp8<16, 512, 64, 16, 128, 64, 2>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_prefill_fp8 = {
    {80322, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},
    {80642, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},
    {80804, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 4>(p); }},
    {80964, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 4>(p); }},
    {81282, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 2>(p); }},
    {80321, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},
    {80325, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 5>(p); }},
    {80481, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},
    {80482, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},
    {80485, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 5>(p); }},
    {80641, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},
    {80645, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 5>(p); }},
    {80801, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 1>(p); }},
    {80802, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 2>(p); }},
    {80805, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 5>(p); }},
    {80961, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 1>(p); }},
    {80962, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 2>(p); }},
    {80965, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 5>(p); }},
    {81281, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 1>(p); }},
    {81285, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 5>(p); }},
    {100322, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},
    {100482, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},
    {100642, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},
    {100321, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},
    {100325, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 5>(p); }},
    {100481, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},
    {100485, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 5>(p); }},
    {100641, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},
    {100645, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_up_fp8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 5>(p); }},
    {146, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 128, 64, 32, 128, 64, 1>(p); }},
    {71146, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 128, 64, 32, 128, 64, 1>(p); }},
    {159, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 128, 64, 32, 64, 64, 2>(p); }},
    {71159, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 128, 64, 32, 64, 64, 2>(p); }},
    {160, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 256, 64, 32, 128, 64, 2>(p); }},
    {71160, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 256, 64, 32, 128, 64, 2>(p); }},
    {162, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<48, 128, 64, 48, 32, 64, 2>(p); }},
    {165, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<80, 128, 64, 80, 32, 64, 2>(p); }},
    {166, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
    {183, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 64, 128, 32, 64, 64, 1>(p); }},
    {71183, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_fp8<32, 64, 128, 32, 64, 64, 1>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_prefill_fp8 = {
    {80324, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 4>(p); }},
    {80642, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},
    {80804, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 4>(p); }},
    {80964, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 4>(p); }},
    {81282, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 2>(p); }},
    {80321, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},
    {80322, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},
    {80481, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},
    {80482, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},
    {80484, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<48, 128, 64, 48, 32, 64, 2, 4>(p); }},
    {80641, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},
    {80644, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<64, 128, 64, 64, 32, 64, 2, 4>(p); }},
    {80801, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 1>(p); }},
    {80802, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<80, 128, 64, 80, 32, 64, 2, 2>(p); }},
    {80961, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 1>(p); }},
    {80962, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<96, 128, 64, 96, 32, 64, 2, 2>(p); }},
    {81281, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 1>(p); }},
    {81284, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8_nloop<128, 128, 64, 128, 32, 64, 2, 4>(p); }},
    {100324, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 4>(p); }},
    {100484, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 4>(p); }},
    {100644, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 4>(p); }},
    {100321, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 1>(p); }},
    {100322, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<32, 128, 64, 32, 32, 64, 2, 2>(p); }},
    {100481, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 1>(p); }},
    {100482, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<48, 128, 64, 48, 32, 64, 2, 2>(p); }},
    {100641, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 1>(p); }},
    {100642, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_prefill_down_fp8_bk128_nloop<64, 128, 64, 64, 32, 64, 2, 2>(p); }},
    {86, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
    {71086, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
    {165, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8<48, 128, 64, 48, 32, 64, 2>(p); }},
    {168, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8<80, 128, 64, 80, 32, 64, 2>(p); }},
    {71168, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8<80, 128, 64, 80, 32, 64, 2>(p); }},
    {169, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
};

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_decode_n160_fp8(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_decode_n160_fp8(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill_n160_fp8(const GemmParams<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill_n160_fp8(const GemmParams<T, T_hidden> &params);

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_decode_n160_fp8 = {
    {60062, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 32, 512, 16, 32, 64, 2>(p); }},
    {60069, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 128, 256, 16, 128, 64, 2>(p); }},
    {60072, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 16, 128, 16, 16, 64, 4>(p); }},
    {60075, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 32, 64, 16, 32, 64, 4>(p); }},
    {60077, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 32, 256, 16, 32, 64, 4>(p); }},
    {60081, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 64, 256, 16, 64, 64, 4>(p); }},
    {60092, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 32, 256, 16, 32, 128, 1>(p); }},
    {60098, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 64, 1024, 16, 64, 128, 1>(p); }},
    {60121, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 16, 512, 16, 16, 128, 4>(p); }},
    {60125, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_decode_n160_fp8<16, 32, 512, 16, 32, 128, 4>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_decode_n160_fp8 = {
    {60038, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_n160_fp8<16, 128, 64, 16, 32, 64, 2>(p); }},
    {60039, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_n160_fp8<16, 256, 64, 16, 32, 64, 2>(p); }},
    {60042, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_n160_fp8<16, 256, 64, 16, 64, 64, 2>(p); }},
    {60043, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_n160_fp8<16, 512, 64, 16, 64, 64, 2>(p); }},
    {60046, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_decode_n160_fp8<16, 512, 64, 16, 128, 64, 2>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm1_prefill_n160_fp8 = {
    {146, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<32, 128, 64, 32, 128, 64, 1>(p); }},
    {159, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<32, 128, 64, 32, 32, 64, 2>(p); }},
    {70159, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<32, 128, 64, 32, 32, 64, 2>(p); }},
    {160, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<32, 256, 64, 32, 128, 64, 2>(p); }},
    {70160, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<32, 256, 64, 32, 128, 64, 2>(p); }},
    {162, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<48, 128, 64, 48, 32, 64, 2>(p); }},
    {70162, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<48, 128, 64, 48, 32, 64, 2>(p); }},
    {165, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<80, 128, 64, 80, 32, 64, 2>(p); }},
    {70165, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<80, 128, 64, 80, 32, 64, 2>(p); }},
    {166, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
    {70166, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
    {183, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<32, 64, 128, 32, 64, 64, 1>(p); }},
    {70183, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_first_stage_prefill_n160_fp8<32, 64, 128, 32, 64, 64, 1>(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc<scalar_t>> kernel_maps_gemm2_prefill_n160_fp8 = {
    {86, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<32, 128, 64, 32, 32, 64, 2>(p); }},
    {70086, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<32, 128, 64, 32, 32, 64, 2>(p); }},
    {165, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<48, 128, 64, 48, 32, 64, 2>(p); }},
    {70165, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<48, 128, 64, 48, 32, 64, 2>(p); }},
    {168, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<80, 128, 64, 80, 32, 64, 2>(p); }},
    {70168, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<80, 128, 64, 80, 32, 64, 2>(p); }},
    {169, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
    {70169, [](const GemmParams<char, scalar_t> &p)
     { launch_moe_w8a8_second_stage_prefill_n160_fp8<96, 128, 64, 96, 32, 64, 2>(p); }},
};

#endif
