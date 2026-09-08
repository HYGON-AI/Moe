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
    std::cout<<"unsupported BIT"<<std::endl;                \
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
    std::cout<<"unsupported BLOCK_SIZE_M"<<std::endl;       \
  }                                                         \
}()

#define BLOCK_N_SWITCH(BLOCK_SIZE_N, BLOCK_SIZE_N_, ...)    \
[&] {                                                       \
  if (BLOCK_SIZE_N == 64) {                                \
    constexpr static int BLOCK_SIZE_N_ = 64;               \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    std::cout<<"unsupported BLOCK_SIZE_N"<<std::endl;       \
  }                                                         \
}()

#define BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, ...)    \
[&] {                                                       \
  if (BLOCK_SIZE_K == 128) {                                \
    constexpr static int BLOCK_SIZE_K_ = 128;               \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    std::cout<<"unsupported BLOCK_SIZE_K"<<std::endl;       \
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
    std::cout<<"unsupported TOPK"<<std::endl;\
  }                                         \
}()

#define GROUP_SIZE_N_SWITCH(group_size_n, GROUP_SIZE_N, ...)      \
[&] {                                                       \
  if (group_size_n == 128) {                                   \
    constexpr static int GROUP_SIZE_N = 128;                   \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    std::cout<<"unsupported GROUP_SIZE_N"<<std::endl;         \
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
    std::cout<<"unsupported GROUP_SIZE_K"<<std::endl;         \
  }                                                         \
}()

#define BLOCK_SIZE_M_LOOPS_SWITCH(block_size_m_loops, BLOCK_SIZE_M_LOOPS, ...) \
[&] {                                                       \
  if (block_size_m_loops == 1) {                            \
    constexpr static int BLOCK_SIZE_M_LOOPS = 1;            \
    return __VA_ARGS__();                                   \
  } else {                                                  \
    std::cout<<"unsupported BLOCK_SIZE_M_LOOPS"<<std::endl;         \
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
    std::cout<<"unsupported BLOCK_SIZE_N_LOOPS"<<std::endl;         \
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
    std::cout<<"unsupported BLOCK_SIZE_K_LOOPS"<<std::endl;         \
  }                                                         \
}()

#define GROUP_SIZE_SWITCH(group_size, GROUP_SIZE, ...)      \
[&] {                                                       \
  if (group_size == 64) {                                   \
    constexpr static int GROUP_SIZE = 64;                   \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    std::cout<<"unsupported GROUP_SIZE"<<std::endl;         \
  }                                                         \
}()






static inline int64_t normalize_fp8_prefill_mode(int64_t mode) {
  if (mode >= 90000 && mode < 100000) {
    return mode - 10000;
  }
  return mode;
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

    const bool use_prefill_mode = std::is_same_v<ElemT, at::Float8_e4m3fn> && mode >= 70000;
    if (config_m <= 512 && !use_prefill_mode) {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it = kernel_maps_gemm1_decode_fp8<OutT>.find(mode);
        if (it != kernel_maps_gemm1_decode_fp8<OutT>.end()) {
          it->second(params_in);
        } else {
          printf("bfloat version gemm1 No matching kernel configuration found, using default settings \n");
        }
      } else {
        auto it = kernel_maps_gemm1_decode<OutT>.find(mode);
        if (it != kernel_maps_gemm1_decode<OutT>.end()) {
          it->second(params_in);
        } else {
          if constexpr (std::is_same_v<OutT, half>) {
            printf("half version gemm1 No matching kernel configuration found, using default settings \n");
          } else {
            printf("bfloat version gemm1 No matching kernel configuration found, using default settings \n");
          }
        }
      }
    } else {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto& prefill_kernels = kernel_maps_gemm1_prefill_fp8<OutT>;
        auto it = prefill_kernels.find(mode);
        if (it == prefill_kernels.end()) {
          const int64_t prefill_mode = normalize_fp8_prefill_mode(mode);
          if (prefill_mode != mode) {
            it = prefill_kernels.find(prefill_mode);
          }
        }
        if (it != kernel_maps_gemm1_prefill_fp8<OutT>.end()) {
          it->second(params_in);
        } else {
          printf("bfloat version gemm1 No matching kernel configuration found, using default settings \n");
        }
      } else {
        auto it = kernel_maps_gemm1_prefill<OutT>.find(mode);
        if (it != kernel_maps_gemm1_prefill<OutT>.end()) {
          it->second(params_in);
        } else {
          if constexpr (std::is_same_v<OutT, half>) {
            printf("half version gemm1 No matching kernel configuration found \n");
          } else {
            printf("bfloat version gemm1 No matching kernel configuration found, using default settings \n");
          }
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

    const bool use_prefill_mode = std::is_same_v<ElemT, at::Float8_e4m3fn> && mode >= 70000;
    if (config_m <= 512 && !use_prefill_mode) {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it = kernel_maps_gemm2_decode_fp8<OutT>.find(mode);
        if (it != kernel_maps_gemm2_decode_fp8<OutT>.end()) {
          it->second(params_in);
        } else {
          printf("bfloat version gemm2 No matching kernel configuration found, using default settings \n");
        }
      } else {
        auto it = kernel_maps_gemm2_decode<OutT>.find(mode);
        if (it != kernel_maps_gemm2_decode<OutT>.end()) {
          it->second(params_in);
        } else {
          if constexpr (std::is_same_v<OutT, half>) {
            printf("half version gemm2 No matching kernel configuration found, using default settings \n");
          } else {
            printf("bfloat version gemm2 No matching kernel configuration found, using default settings \n");
          }
        }
      }
    } else {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto& prefill_kernels = kernel_maps_gemm2_prefill_fp8<OutT>;
        auto it = prefill_kernels.find(mode);
        if (it == prefill_kernels.end()) {
          const int64_t prefill_mode = normalize_fp8_prefill_mode(mode);
          if (prefill_mode != mode) {
            it = prefill_kernels.find(prefill_mode);
          }
        }
        if (it != kernel_maps_gemm2_prefill_fp8<OutT>.end()) {
          it->second(params_in);
        } else {
          printf("bfloat version gemm2 No matching kernel configuration found, using default settings \n");
        }
      } else {
        auto it = kernel_maps_gemm2_prefill<OutT>.find(mode);
        if (it != kernel_maps_gemm2_prefill<OutT>.end()) {
          it->second(params_in);
        } else {
          if constexpr (std::is_same_v<OutT, half>) {
            printf("half version gemm2  No matching kernel configuration found \n");
          } else {
            printf("bfloat version gemm2 No matching kernel configuration found, using default settings \n");
          }
        }
      }
    }
  }
}



template <typename OutT, typename ElemT>
static void moe_marlin_w8a8_dispatch_gemm_stages_n160(
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

    const bool use_prefill_mode = std::is_same_v<ElemT, at::Float8_e4m3fn> && mode >= 70000;
    if (config_m <= 512 && !use_prefill_mode) {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it = kernel_maps_gemm1_decode_n160_fp8<OutT>.find(mode);
        if (it != kernel_maps_gemm1_decode_n160_fp8<OutT>.end()) {
          it->second(params_in);
        } else {
          std::cout<<"moe_marlin_w8a8_dispatch_gemm_stages<half, at::Float8_e4m3fn> mode \n"<<mode<<std::endl;

        }
      }
    } else {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it_prefill = kernel_maps_gemm1_prefill_fp8<OutT>.find(mode);
        if (it_prefill != kernel_maps_gemm1_prefill_fp8<OutT>.end()) {
          it_prefill->second(params_in);
        } else {
          auto it = kernel_maps_gemm1_prefill_n160_fp8<OutT>.find(mode);
          if (it != kernel_maps_gemm1_prefill_n160_fp8<OutT>.end()) {
            it->second(params_in);
          } else {
            printf("bfloat version gemm1 No matching kernel configuration found, using default settings \n");
          }
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

    const bool use_prefill_mode = std::is_same_v<ElemT, at::Float8_e4m3fn> && mode >= 70000;
    if (config_m <= 512 && !use_prefill_mode) {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it = kernel_maps_gemm2_decode_n160_fp8<OutT>.find(mode);
        if (it != kernel_maps_gemm2_decode_n160_fp8<OutT>.end()) {
          it->second(params_in);
        } else {
          printf("bfloat version gemm2 No matching kernel configuration found, using default settings \n");
        }
      }
    } else {
      if constexpr (std::is_same_v<ElemT, at::Float8_e4m3fn>) {
        auto it_prefill = kernel_maps_gemm2_prefill_fp8<OutT>.find(mode);
        if (it_prefill != kernel_maps_gemm2_prefill_fp8<OutT>.end()) {
          it_prefill->second(params_in);
        } else {
          auto it = kernel_maps_gemm2_prefill_n160_fp8<OutT>.find(mode);
          if (it != kernel_maps_gemm2_prefill_n160_fp8<OutT>.end()) {
            it->second(params_in);
          } else {
            printf("bfloat version gemm2 No matching kernel configuration found, using default settings \n");
          }
        }
      }
    }
  }
}
