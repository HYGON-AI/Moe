#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

template <typename T_hidden = bhalf_t>
struct GemmParams_wfp4a8
{
    GemmParams_wfp4a8(const char *ptr_A,
                      const char *ptr_B0,
                      T_hidden *ptr_C,
                      float *ptr_A_scale,
                      float *ptr_B_scale,
                      const uint8_t *ptr_B_scale_u8,
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
                      bool is_marlin)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
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
          stride_asm(stride_asm),
          stride_ask(stride_ask),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          real_topk(real_topk),
          is_marlin(is_marlin) {}

    const char *ptr_A;
    const char *ptr_B0;
    T_hidden *ptr_C;
    float *ptr_A_scale;
    float *ptr_B_scale;
    const uint8_t *ptr_B_scale_u8;
    const float *topk_weights;
    const int32_t *sorted_token_ids;
    const int32_t *expert_ids;
    const int32_t num_tokens_post_pad;
    const int32_t *num_tokens_post_pad_ptr;
    uint32_t size_m;
    uint32_t size_n;
    uint32_t size_k;
    uint32_t stride_asm;
    uint32_t stride_ask;
    uint32_t stride_bse;
    uint32_t stride_bsn;
    uint32_t stride_bsk;
    uint32_t sorted_token_lens;
    uint32_t top_k;
    uint32_t real_topk;
    bool is_marlin;
};

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode(const GemmParams_wfp4a8<T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise(const GemmParams_wfp4a8<T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode(const GemmParams_wfp4a8<T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise(const GemmParams_wfp4a8<T_hidden> &params);

template <int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1Nloop(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_9537(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_groupwise_9523(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_9524(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_9121(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise_9525(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_qgroup_9537(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_groupwise_qgroup_9523(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9524(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9121(const GemmParams_wfp4a8<T_hidden> &params);

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise_qgroup_9525(const GemmParams_wfp4a8<T_hidden> &params);

template <int FIXED_SIZE_K, int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_fixed_k(const GemmParams_wfp4a8<T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_K192(const GemmParams_wfp4a8<T_hidden> &params);

template <typename scalar_t>
using KernelFunc_wfp4a8 = std::function<void(const GemmParams_wfp4a8<scalar_t> &)>;

#define WFP4A8_GEMM1_NLOOP_ENTRY(MODE, BM, BK, N_LOOP) \
    {MODE, [](const GemmParams_wfp4a8<scalar_t> &p) { launch_moe_wfp4a8_first_stage_prefill_GEMM1Nloop<N_LOOP, BM, 128, BK, BM, 32, BK, 2>(p); }}

#define WFP4A8_GEMM2_RUNTIME_K_ENTRY(MODE, BM, BK, N_LOOP) \
    {MODE, [](const GemmParams_wfp4a8<scalar_t> &p) { launch_moe_wfp4a8_second_stage_prefill_fixed_k<0, N_LOOP, BM, 128, BK, BM, 32, BK, 2>(p); }}

#define WFP4A8_GEMM2_FIXED_K64_ENTRY(MODE, FIXED_K, BM, BK, N_LOOP) \
    {MODE, [](const GemmParams_wfp4a8<scalar_t> &p) { launch_moe_wfp4a8_second_stage_prefill_fixed_k<FIXED_K, N_LOOP, BM, 128, BK, BM, 32, BK, 2>(p); }}

#define WFP4A8_GEMM2_K192_ENTRY(MODE, BM, BK, N_LOOP) \
    WFP4A8_GEMM2_FIXED_K64_ENTRY(MODE, 192, BM, BK, N_LOOP)

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm1_decode_wfp4a8_channelwise = {
    {101, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 128, 64, 16, 32, 64, 4>(p); }},
    {121, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 32, 512, 16, 32, 128, 4>(p); }},
    {122, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 128, 64, 16, 32, 64, 4>(p); }},
    {123, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 32, 64, 16, 32, 64, 4>(p); }},
    {124, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 32, 512, 16, 32, 128, 2>(p); }},
    {125, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 128, 64, 16, 32, 64, 2>(p); }},
    {126, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 32, 256, 16, 32, 64, 2>(p); }},
    {127, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode<16, 32, 256, 16, 32, 64, 4>(p); }},
    WFP4A8_GEMM1_NLOOP_ENTRY(31316, 16, 64, 3),
    WFP4A8_GEMM1_NLOOP_ENTRY(31332, 32, 64, 3),
    WFP4A8_GEMM1_NLOOP_ENTRY(31348, 48, 64, 3),
    WFP4A8_GEMM1_NLOOP_ENTRY(31364, 64, 64, 3),
    WFP4A8_GEMM1_NLOOP_ENTRY(374116, 16, 64, 1),
    WFP4A8_GEMM1_NLOOP_ENTRY(374132, 32, 64, 1),
    WFP4A8_GEMM1_NLOOP_ENTRY(374332, 32, 64, 3),
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm2_decode_wfp4a8_channelwise = {
    {102, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode<16, 128, 64, 16, 32, 64, 2>(p); }},
    {32, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode<16, 128, 64, 16, 32, 64, 2>(p); }},
    {86, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode<32, 128, 64, 32, 32, 64, 2>(p); }},
    {118, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode<48, 128, 64, 48, 32, 64, 2>(p); }},
    {166, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode<64, 128, 64, 64, 32, 64, 2>(p); }},
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41316, 16, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41332, 32, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41348, 48, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41364, 64, 64, 4),
    WFP4A8_GEMM2_K192_ENTRY(384116, 16, 64, 1),
    WFP4A8_GEMM2_K192_ENTRY(384216, 16, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(384232, 32, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(384248, 48, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(394232, 32, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(394432, 32, 64, 4),
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm1_prefill_wfp4a8_channelwise = {
    {501, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_prefill_GEMM1Nloop<3, 64, 128, 64, 64, 32, 64, 2>(p); }},
    {503, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_prefill_GEMM1Nloop<4, 64, 128, 64, 64, 32, 64, 2>(p); }},
    WFP4A8_GEMM1_NLOOP_ENTRY(160, 64, 64, 4),
    WFP4A8_GEMM1_NLOOP_ENTRY(290, 48, 64, 4),
    WFP4A8_GEMM1_NLOOP_ENTRY(31332, 32, 64, 3),
    WFP4A8_GEMM1_NLOOP_ENTRY(31348, 48, 64, 3),
    WFP4A8_GEMM1_NLOOP_ENTRY(374116, 16, 64, 1),
    WFP4A8_GEMM1_NLOOP_ENTRY(374332, 32, 64, 3),
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm2_prefill_wfp4a8_channelwise = {
    {502, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_prefill_K192<64, 128, 64, 64, 32, 64, 2>(p); }},
    {504, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_prefill_fixed_k<0, 4, 64, 128, 64, 64, 32, 64, 2>(p); }},
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(32, 16, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(86, 32, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(118, 48, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(166, 64, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41316, 16, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41332, 32, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41348, 48, 64, 4),
    WFP4A8_GEMM2_RUNTIME_K_ENTRY(41364, 64, 64, 4),
    WFP4A8_GEMM2_K192_ENTRY(384116, 16, 64, 1),
    WFP4A8_GEMM2_K192_ENTRY(384216, 16, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(384232, 32, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(384248, 48, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(394232, 32, 64, 2),
    WFP4A8_GEMM2_K192_ENTRY(394432, 32, 64, 4),
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm1_decode_wfp4a8_groupwise = {
    {201, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_9524(p); }},
    {221, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_9121(p); }},
    {9524, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_9524(p); }},
    {9121, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_9121(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm2_decode_wfp4a8_groupwise = {
    {202, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode_groupwise_9525(p); }},
    {9525, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode_groupwise_9525(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm1_prefill_wfp4a8_groupwise = {
    {601, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_9537(p); }},
    {9537, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_9537(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm2_prefill_wfp4a8_groupwise = {
    {602, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_prefill_groupwise_9523(p); }},
    {9523, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_prefill_groupwise_9523(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm1_decode_wfp4a8_groupwise_qgroup = {
    {201, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9524(p); }},
    {221, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9121(p); }},
    {9524, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9524(p); }},
    {9121, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9121(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm2_decode_wfp4a8_groupwise_qgroup = {
    {202, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode_groupwise_qgroup_9525(p); }},
    {9525, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_decode_groupwise_qgroup_9525(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm1_prefill_wfp4a8_groupwise_qgroup = {
    {601, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_qgroup_9537(p); }},
    {9537, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_qgroup_9537(p); }},
};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>> kernel_maps_gemm2_prefill_wfp4a8_groupwise_qgroup = {
    {602, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_prefill_groupwise_qgroup_9523(p); }},
    {9523, [](const GemmParams_wfp4a8<scalar_t> &p)
     { launch_moe_wfp4a8_second_stage_prefill_groupwise_qgroup_9523(p); }},
};

#undef WFP4A8_GEMM1_NLOOP_ENTRY
#undef WFP4A8_GEMM2_RUNTIME_K_ENTRY
#undef WFP4A8_GEMM2_FIXED_K64_ENTRY
#undef WFP4A8_GEMM2_K192_ENTRY
