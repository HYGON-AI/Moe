// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <torch/all.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <type_traits>
#include <vector>

#include "moe_wna16_utils.h"
#include "moe_w8a16_block_wise.h"
#include "moe_w8a16_awq.h"
#include "moe_w4a16.h"
#include "moe_w4a16_2.h"
#include "moe_w4a16_base.h"
#include "topk_softmax_kernel.h"
#include "moe_w8a8_block_wise.h"
#include "moe_w8a8_block_wise_kernel2.h"
#include "moe_w8a8_block_wise_fp8.h"
#include "moe_w8a8_block_wise_kernel2_fp8.h"
#include "moe_w8a8_opt.h"
#include "moe_w4a16_opt.h"
#include "moe_w8a16_chan_opt.h"

#undef S_BARRIER
#undef vmcnt_wait
#undef vmcnt
#undef lgkmcnt_wait
#undef lgkmcnt_wait_barrier
#undef DIVIDE
#undef DIV_ceil
#undef BOOL_SWITCH
#undef ATOMIC_SWITCH
#include "moe_w16a16_opt.h"
#undef BOOL_SWITCH
#undef ATOMIC_SWITCH
#undef DIVIDE
#undef DIV_ceil
#undef S_BARRIER
#undef vmcnt_wait
#undef vmcnt
#undef lgkmcnt_wait
#undef lgkmcnt_wait_barrier

using at::device_of;

#define BIT_SWITCH(bit, BIT, ...)                           \
[&] {                                                       \
  if (bit == 8) {                                           \
    constexpr static int BIT = 8;                           \
    return __VA_ARGS__();                                   \
      }else if (bit == 4) {                    \
    constexpr static int BIT = 4;          \
    return __VA_ARGS__();                   \
  } \
   else {                                                  \
    TORCH_CHECK(false, "unsupported BIT: ", bit);           \
  }                                                         \
}()

#define BLOCK_M_SWITCH(BLOCK_SIZE_M, BLOCK_SIZE_M_, ...)    \
[&] {                                                       \
  if (BLOCK_SIZE_M == 16) {                                  \
    constexpr static int BLOCK_SIZE_M_ = 16;                 \
    return __VA_ARGS__();                                   \
  }else if(BLOCK_SIZE_M == 32) {\
    constexpr static int BLOCK_SIZE_M_ = 32;                 \
    return __VA_ARGS__(); \
  }else if(BLOCK_SIZE_M == 48) {\
    constexpr static int BLOCK_SIZE_M_ = 48;                 \
    return __VA_ARGS__(); \
  }else if(BLOCK_SIZE_M == 64) {\
    constexpr static int BLOCK_SIZE_M_ = 64;                 \
    return __VA_ARGS__(); \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported BLOCK_SIZE_M: ", BLOCK_SIZE_M); \
  }                                                         \
}()

#define BLOCK_N_SWITCH(BLOCK_SIZE_N, BLOCK_SIZE_N_, ...)    \
[&] {                                                       \
  if (BLOCK_SIZE_N == 64) {                                \
    constexpr static int BLOCK_SIZE_N_ = 64;               \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported BLOCK_SIZE_N: ", BLOCK_SIZE_N); \
  }                                                         \
}()

#define BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, ...)    \
[&] {                                                       \
  if (BLOCK_SIZE_K == 128) {                                \
    constexpr static int BLOCK_SIZE_K_ = 128;               \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported BLOCK_SIZE_K: ", BLOCK_SIZE_K); \
  }                                                         \
}()

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define TOPK_SWITCH(topk, TOPK, ...)   \
[&] {                                       \
  if (topk == 8) {                          \
    constexpr static int TOPK = 8;          \
    return __VA_ARGS__();                   \
  }else if (topk == 1) {                    \
    constexpr static int TOPK = 1;          \
    return __VA_ARGS__();                   \
  }else {                                  \
    TORCH_CHECK(false, "unsupported TOPK: ", topk);\
  }                                         \
}()

#define GROUP_SIZE_N_SWITCH(group_size_n, GROUP_SIZE_N, ...)      \
[&] {                                                       \
  if (group_size_n == 128) {                                   \
    constexpr static int GROUP_SIZE_N = 128;                   \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported GROUP_SIZE_N: ", group_size_n); \
  }                                                         \
}()

#define GROUP_SIZE_K_SWITCH(group_size_k, GROUP_SIZE_K, ...)      \
[&] {                                                       \
  if (group_size_k == 128) {                                   \
    constexpr static int GROUP_SIZE_K = 128;                   \
    return __VA_ARGS__();                                     \
      }else if(group_size_k == 64){                         \
    constexpr static int GROUP_SIZE_K = 64;            \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported GROUP_SIZE_K: ", group_size_k); \
  }                                                         \
}()

#define BLOCK_SIZE_M_LOOPS_SWITCH(block_size_m_loops, BLOCK_SIZE_M_LOOPS, ...) \
[&] {                                                       \
  if (block_size_m_loops == 1) {                            \
    constexpr static int BLOCK_SIZE_M_LOOPS = 1;            \
    return __VA_ARGS__();                                   \
  } else {                                                  \
    TORCH_CHECK(false, "unsupported BLOCK_SIZE_M_LOOPS: ", block_size_m_loops); \
  }                                                         \
}()

#define BLOCK_SIZE_N_LOOPS_SWITCH(block_size_n_loops, BLOCK_SIZE_N_LOOPS, ...) \
[&] {                                                       \
  if (block_size_n_loops ==4) {                            \
    constexpr static int BLOCK_SIZE_N_LOOPS = 4;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_n_loops ==1) {\
     constexpr static int BLOCK_SIZE_N_LOOPS = 1;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_n_loops ==2) {\
     constexpr static int BLOCK_SIZE_N_LOOPS = 2;            \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported BLOCK_SIZE_N_LOOPS: ", block_size_n_loops); \
  }                                                         \
}()

#define BLOCK_SIZE_K_LOOPS_SWITCH(block_size_k_loops, BLOCK_SIZE_K_LOOPS, ...) \
[&] {                                                       \
  if (block_size_k_loops == 1) {                            \
    constexpr static int BLOCK_SIZE_K_LOOPS = 1;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_k_loops == 2) {\
    constexpr static int BLOCK_SIZE_K_LOOPS = 2;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_k_loops == 4) {\
    constexpr static int BLOCK_SIZE_K_LOOPS = 4;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_k_loops == 7) {\
    constexpr static int BLOCK_SIZE_K_LOOPS = 7;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_k_loops == 8) {\
    constexpr static int BLOCK_SIZE_K_LOOPS = 8;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_k_loops == 14) {\
    constexpr static int BLOCK_SIZE_K_LOOPS = 14;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_k_loops == 28) {\
    constexpr static int BLOCK_SIZE_K_LOOPS = 28;            \
    return __VA_ARGS__();                                   \
  }else if(block_size_k_loops == 56) {\
    constexpr static int BLOCK_SIZE_K_LOOPS = 56;            \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported BLOCK_SIZE_K_LOOPS: ", block_size_k_loops); \
  }                                                         \
}()

#define GROUP_SIZE_SWITCH(group_size, GROUP_SIZE, ...)      \
[&] {                                                       \
  if (group_size == 64) {                                   \
    constexpr static int GROUP_SIZE = 64;                   \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    TORCH_CHECK(false, "unsupported GROUP_SIZE: ", group_size); \
  }                                                         \
}()






// Descriptive HIPC channelwise mode format: S F K MM VVV.
// S=stage, F=sorting, K=kernel family, MM=BLOCK_SIZE_M / 16,
// VVV=profile ID. The profile descriptor owns N_LOOP and any other static
// attributes. The n160/kgrain64 family is intentionally excluded from this
// descriptive mode namespace.
static inline bool is_moe_channel_mode(int64_t mode) {
  return mode >= 10000000LL && mode < 30000000LL;
}

// Kept as a named alias for callers that still describe the FP8 path.
static inline bool is_fp8_channel_mode(int64_t mode) {
  return is_moe_channel_mode(mode);
}

static inline int moe_channel_mode_stage(int64_t mode) {
  return static_cast<int>(mode / 10000000LL);
}

static inline int moe_channel_mode_sorting(int64_t mode) {
  return static_cast<int>((mode / 1000000LL) % 10);
}

static inline int moe_channel_mode_family(int64_t mode) {
  return static_cast<int>((mode / 100000LL) % 10);
}

static inline int moe_channel_mode_bm(int64_t mode) {
  return static_cast<int>((mode / 1000LL) % 100) * 16;
}

static inline int moe_channel_mode_profile(int64_t mode) {
    return static_cast<int>(mode % 1000LL);
}

static inline void validate_moe_channel_layout(int64_t mode,
                                                int expected_sorting) {
    TORCH_CHECK(is_moe_channel_mode(mode),
                "invalid HIPC W8A8 channelwise mode: ", mode);
    const int sorting = moe_channel_mode_sorting(mode);
    TORCH_CHECK(sorting == expected_sorting,
                "W8A8 channelwise mode ", mode,
                " requests sorting layout ", sorting,
                ", but this dispatch path supports layout ", expected_sorting,
                ". No kernel alias is allowed between CK sorting and moe_align.");
}

static inline void validate_moe_channel_mode(int64_t mode) {
    TORCH_CHECK(is_moe_channel_mode(mode),
                "invalid HIPC W8A8 channelwise mode: ", mode);
    const int sorting = moe_channel_mode_sorting(mode);
    TORCH_CHECK(sorting == 0 || sorting == 1,
                "invalid HIPC W8A8 channelwise sorting field in mode: ", mode);
}

static inline bool moe_w8a8_mode_uses_moe_align(int64_t mode) {
  return ((mode / 1000000) % 10) == 1;
}

static inline bool moe_w8a8_mode_is_decode(int64_t mode, int config_m) {
  TORCH_CHECK(mode >= 10000000 && mode < 30000000,
              "unsupported W8A8 mode encoding: ", mode);
  return ((mode / 100000) % 10) == 0;
}

static inline void moe_w8a8_throw_missing_kernel(
    const char* stage,
    const char* phase,
    int64_t mode,
    int config_m,
    int size_n,
    int size_k,
    int real_size_k) {
  TORCH_CHECK(false,
      "No matching moe_c w8a8 kernel: stage=", stage,
      ", phase=", phase,
      ", mode=", mode,
      ", config_m=", config_m,
      ", size_n=", size_n,
      ", size_k=", size_k,
      ", real_size_k=", real_size_k);
}

template <typename OutT, typename ElemT>
static void moe_marlin_w8a8_dispatch_gemm_stages(
    bool first_stage,
    const torch::Tensor& input,
    const torch::Tensor& b_qweight,
    torch::Tensor& output_alias,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    const float* topk_weights_ptr,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& expert_ids,
    int num_pad,
    const torch::Tensor& num_tokens_post_pad,
    int size_m,
    int size_n,
    int size_k,
    int stride_asm,
    int stride_ask,
    int stride_bse,
    int stride_bsn,
    int stride_bsk,
    int64_t top_k,
    uint32_t real_topk,
    bool is_marlin,
    int64_t mode,
    int config_m,
    int real_size_k,
    bool tensorwise_scale = false) {
  static_assert(
      std::is_same_v<ElemT, at::Float8_e4m3fn> || std::is_same_v<ElemT, int8_t>,
      "ElemT must be at::Float8_e4m3fn or int8_t");
  const bool is_decode = moe_w8a8_mode_is_decode(mode, config_m);
  const int routing = (mode / 1000000) % 10;
  TORCH_CHECK(mode / 10000000 == (first_stage ? 1 : 2),
              "W8A8 mode stage does not match GEMM stage: ", mode);
  if constexpr (std::is_same_v<ElemT, int8_t>) {
    TORCH_CHECK(routing == 0, "W8A8 INT8 requires CK sorting mode: ", mode);
  } else {
    TORCH_CHECK(routing == 0 || (routing == 1 && !is_decode),
                "unsupported W8A8 FP8 routing/kernel family: ", mode);
  }
  if (first_stage) {
    const int64_t EM = sorted_token_ids.size(0);
    GemmParams<char, OutT> params_in(
        (const char*)input.data_ptr<ElemT>(),
        (const char*)b_qweight.data_ptr<ElemT>(),
        (OutT*)output_alias.data_ptr(),
        (float*)a_scale.data_ptr(),
        (float*)b_scale.data_ptr(),
        topk_weights_ptr,
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_pad,
        num_tokens_post_pad.data_ptr<int32_t>(),
        size_m,
        size_n,
        size_k,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsn,
        stride_bsk,
        EM,
        top_k,
        real_topk,
        is_marlin,
        tensorwise_scale,
        real_size_k);

    if (is_decode) {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it = kernel_maps_gemm1_decode_fp8_ck_sorting<OutT>.find(mode);
        if (it != kernel_maps_gemm1_decode_fp8_ck_sorting<OutT>.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm1", "fp8_decode", mode, config_m, size_n, size_k, real_size_k);
        }
      } else {
        auto it = kernel_maps_gemm1_decode_ck_sorting<OutT>.find(mode);
        if (it != kernel_maps_gemm1_decode_ck_sorting<OutT>.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm1", "int8_decode", mode, config_m, size_n, size_k, real_size_k);
        }
      }
    } else {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto& prefill_kernels = moe_w8a8_mode_uses_moe_align(mode)
            ? kernel_maps_gemm1_prefill_fp8_moe_align<OutT>
            : kernel_maps_gemm1_prefill_fp8_ck_sorting<OutT>;
        auto it = prefill_kernels.find(mode);
        if (it != prefill_kernels.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm1", "fp8_prefill", mode, config_m, size_n, size_k, real_size_k);
        }
      } else {
        auto it = kernel_maps_gemm1_prefill_ck_sorting<OutT>.find(mode);
        if (it != kernel_maps_gemm1_prefill_ck_sorting<OutT>.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm1", "int8_prefill", mode, config_m, size_n, size_k, real_size_k);
        }
      }
    }
  } else {
    const int64_t EM = sorted_token_ids.size(0);
    GemmParams<char, OutT> params_in(
        (const char*)input.data_ptr<ElemT>(),
        (const char*)b_qweight.data_ptr<ElemT>(),
        (OutT*)output_alias.data_ptr(),
        (float*)a_scale.data_ptr(),
        (float*)b_scale.data_ptr(),
        topk_weights_ptr,
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_pad,
        num_tokens_post_pad.data_ptr<int32_t>(),
        size_m,
        size_n,
        size_k,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsn,
        stride_bsk,
        EM,
        top_k,
        real_topk,
        is_marlin,
        tensorwise_scale,
        real_size_k);

    if (is_decode) {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it = kernel_maps_gemm2_decode_fp8_ck_sorting<OutT>.find(mode);
        if (it != kernel_maps_gemm2_decode_fp8_ck_sorting<OutT>.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm2", "fp8_decode", mode, config_m, size_n, size_k, real_size_k);
        }
      } else {
        auto it = kernel_maps_gemm2_decode_ck_sorting<OutT>.find(mode);
        if (it != kernel_maps_gemm2_decode_ck_sorting<OutT>.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm2", "int8_decode", mode, config_m, size_n, size_k, real_size_k);
        }
      }
    } else {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto& prefill_kernels = moe_w8a8_mode_uses_moe_align(mode)
            ? kernel_maps_gemm2_prefill_fp8_moe_align<OutT>
            : kernel_maps_gemm2_prefill_fp8_ck_sorting<OutT>;
        auto it = prefill_kernels.find(mode);
        if (it != prefill_kernels.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm2", "fp8_prefill", mode, config_m, size_n, size_k, real_size_k);
        }
      } else {
        auto it = kernel_maps_gemm2_prefill_ck_sorting<OutT>.find(mode);
        if (it != kernel_maps_gemm2_prefill_ck_sorting<OutT>.end()) {
          it->second(params_in);
        } else {
          moe_w8a8_throw_missing_kernel("gemm2", "int8_prefill", mode, config_m, size_n, size_k, real_size_k);
        }
      }
    }
  }
}
// 模板抽象
// BLOCK_MNK: [16, 128, 128]
// WARP_MNK: [16, 32, 64]
// MMA_MNK: [16, 16, 32]
// BLOCK_MNK / WARP_MNK = [1, 4, 2]  代表warp在MN方向的排布 block_k/warp_k代表stage
// WARP_MNK / MMA_MNK = [1, 2, 2]    代表warp在MNK方向重复计算的次数 会分配额外的寄存器
