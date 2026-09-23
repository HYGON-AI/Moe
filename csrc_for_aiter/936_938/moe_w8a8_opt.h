// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT


#ifndef MOE_W8A8_OPT_HIP_H
#define MOE_W8A8_OPT_HIP_H

#include "moe_w8a8_utils.h"
#include "moe_w8a8_config.h"
#include "moe_w8a8_bk128_prefill.h"
#include "moe_w8a8_int8_bk128_prefill.h"
#include "moe_w8a8_int8_bk64_prefill.h"

template <
    typename scalar_t,
    typename Element,
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M,
    uint16_t BLOCK_SIZE_N,
    uint16_t BLOCK_SIZE_K,
    uint16_t WARP_M,
    uint16_t WARP_N,
    uint16_t WARP_K,
    uint16_t GROUP_N,
    uint16_t GROUP_K,
    int STAGES,
    bool mul_topk_weight>
__attribute__((hcu_co_issue_vgpr_size(256)))
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP(
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.x; // pid_n
  const int bidz = blockIdx.y; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;                                          // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];                                               // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;                         // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 64;                       // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  intx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    uint32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;

    a_scale_ptr_arr[idx] = input_scale + std::min(token_ids, size_m - 1) * stride_asm; // 计算M方向的偏移
  }

  float b_scale[(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
    {
      b_scale[idx] = expert_b_scale;
    }
  }
  else
  {
    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr);
#pragma unroll
    for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
    {

      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 0], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 0);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 1], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 4);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 2], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 8);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 3], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 12);
    }
  }

  {
    if (size_k == 7168)
    {
      constexpr static int SIZE_K = 7168;
      gemm_nt_marlin_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 6144)
    {
      constexpr static int SIZE_K = 6144;
      gemm_nt_marlin_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 4096)
    {
      constexpr static int SIZE_K = 4096;
      gemm_nt_marlin_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 3072)
    {
      constexpr static int SIZE_K = 3072;
      gemm_nt_marlin_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 2048)
    {
      constexpr static int SIZE_K = 2048;
      gemm_nt_marlin_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
  }
  if (warp_k_id == 0 || warp_k_num == 1)
  {
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      float *a_scale = a_scale_ptr_arr[min_tile_m];
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * a_scale[0] * b_scale[min_tile_n * 4 + reg_id];
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
          output_lds[index] = b32_to_b16<scalar_t>(value);
        }
      }
    }
  }

  __syncthreads();

  {
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;
      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) =
            *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight> // true
__attribute__((hcu_co_issue_vgpr_size(256)))
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{

  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

  const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    int token_ids = sorted_token_ids_element & 0x00FFFFFF;
    int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
    int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1);
    float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
    weight_dot_a_scale[idx] = topk_weights[token_index_safe] * input_scale_value;
  }

  float weight_dot_a_scale_value[WARP_M / mfma_m];
#pragma unroll
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
  {
    weight_dot_a_scale_value[min_tile_m] = weight_dot_a_scale[min_tile_m];
  }

  constexpr int n_loop_num = 1;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
      for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
      {
        b_scale[n_loop][idx] = expert_b_scale;
      }
    }
  }
  else
  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
        vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 0], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 0);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 1], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 4);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 2], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 8);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 3], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 12);
      }
    }
  }

  int token_index_store[BLOCK_SIZE_M * BLOCK_SIZE_N / (WARP_NUM * 512)];
  int tok_ids_store[BLOCK_SIZE_M * BLOCK_SIZE_N / (WARP_NUM * 512)];
  {
    int tid = threadIdx.x;
    int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

    int n_idx = threadIdx.x % N_thread; // tid % 4 = 0-3
    int m_idx = threadIdx.x / N_thread; // tid / 4 = 0-32
    int it_num = (WARP_NUM * 64) / N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += it_num)
    {
      int it = m_idx / it_num;
      const int32_t sorted_token_ids_element_store = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids_store = sorted_token_ids_element_store & 0x00FFFFFF;
      tok_ids_store[it] = (sorted_token_ids_element_store & 0xFF000000) >> 24;
      token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    }
  }

  intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2

  gemm_nt_marlin_decode_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, real_topk);

  __syncthreads();

  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {

    if (warp_k_id == 0 || warp_k_num == 1)
    {

#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
        {
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            float value = C_reg[n_loop][min_tile_m * (WARP_N / mfma_n) + min_tile_n][reg_id] * weight_dot_a_scale_value[min_tile_m] * b_scale[n_loop][min_tile_n * 4 + reg_id];
            int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 8 /*padding*/; // + (lane_id & 15 )/2 * 2/*padding*/
            output_lds[index] = b32_to_b16<scalar_t>(value);
          }
        }
      }
    }
    __syncthreads();

    {

      const int tid = threadIdx.x;
      constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

      int n_idx = threadIdx.x % N_thread; // tid % 4 = 0-3
      int m_idx = threadIdx.x / N_thread; // tid / 4 = 0-32
      int it_num = (WARP_NUM * 64) / N_thread;
      for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
      {
        int it = m_idx / it_num;
        if (tok_ids_store[it] < real_topk)
        {
          *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[bidz * size_m * top_k * size_n + token_index_store[it] * size_n + n_idx * 8 + BLOCK_SIZE_N * n_loop]) =
              *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 8 /*padding*/]);
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight>
__attribute__((hcu_co_issue_vgpr_size(256)))
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP(
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.x; // pid_n
  const int bidz = blockIdx.y; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;                                          // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];                                               // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;                         // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 64;                       // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  intx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    uint32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
    a_scale_ptr_arr[idx] = input_scale + std::min(token_ids, size_m - 1) * stride_asm; // 计算M方向的偏移
  }

  float b_scale[(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
    {
      b_scale[idx] = expert_b_scale;
    }
  }
  else
  {
    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr);
#pragma unroll
    for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
    {

      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 0], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 0);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 1], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 4);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 2], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 8);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 3], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 12);
    }
  }

  {
    if (size_k == 7168)
    {
      constexpr static int SIZE_K = 7168;
      gemm_nt_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 6144)
    {
      constexpr static int SIZE_K = 6144;
      gemm_nt_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 4096)
    {
      constexpr static int SIZE_K = 4096;
      gemm_nt_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 3072)
    {
      constexpr static int SIZE_K = 3072;
      gemm_nt_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 2048)
    {
      constexpr static int SIZE_K = 2048;
      gemm_nt_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
  }
  if (warp_k_id == 0 || warp_k_num == 1)
  {

    int index[WARP_M / mfma_m][WARP_N / mfma_n][4];
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      float *a_scale = a_scale_ptr_arr[min_tile_m];
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          index[min_tile_m][min_tile_n][reg_id] = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
        }
      }
    }

    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      float *a_scale = a_scale_ptr_arr[min_tile_m];
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * a_scale[0] * b_scale[min_tile_n * 4 + reg_id];
          output_lds[index[min_tile_m][min_tile_n][reg_id]] = b32_to_b16<scalar_t>(value);
        }
      }
    }
  }

  __syncthreads();

  {
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;
      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) =
            *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight> // true
__attribute__((hcu_co_issue_vgpr_size(256)))
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

  const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1);
      float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
      weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
    }
  }

  constexpr int n_loop_num = 4;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  constexpr int store_size = WARP_M / 4;
  int token_index_store[store_size];
  int tok_ids_store[store_size];
  int32_t sorted_token_ids_element_store[store_size];
  auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

  {

    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      for (int i = 0; i < 4; i++)
      {
        int m_idx = min_tile_m * 16 + i * 4 + col_id;
        int it = m_idx / 4;
        inline_buffer_load_dword(sorted_token_ids_element_store[it], m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
      for (int idx = 0; idx < WARP_N / mfma_n; idx++)
      {
        b_scale[n_loop][idx] = expert_b_scale;
      }
    }
  }
  else
  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
        for (int i = 0; i < 32 / mfma_n; i++)
        {

          vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

          inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i], row_id * 2, b_scale_ptr_prepared, min_tile_n * 32 + i);
        }
      }
    }
  }

  intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if (size_k == 128)
  {
    constexpr int SIZE_K = 128;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 256)
  {
    constexpr int SIZE_K = 256;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 320)
  {
    constexpr int SIZE_K = 320;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 384)
  {
    constexpr int SIZE_K = 384;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 512)
  {
    constexpr int SIZE_K = 512;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 640)
  {
    constexpr int SIZE_K = 640;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 768)
  {
    constexpr int SIZE_K = 768;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 1024)
  {
    constexpr int SIZE_K = 1024;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_2<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }

  __builtin_amdgcn_sched_barrier(0);

  scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {

#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {
#pragma unroll
      for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
      {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {

          const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t> /* f32_to_bf16  */ (tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
      {
        int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
          if (tok_ids_store[it] < real_topk)
          {

            *(vec_element_2<scalar_t> *)(&g_output[token_index_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
          }
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M,
    uint16_t BLOCK_SIZE_N,
    uint16_t BLOCK_SIZE_K,
    uint16_t WARP_M,
    uint16_t WARP_N,
    uint16_t WARP_K,
    uint16_t GROUP_N,
    uint16_t GROUP_K,
    int STAGES,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP_N160_FP8(
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;                                          // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];                                               // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;                         // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 64;                       // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  vec4_fp32 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    uint32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
    a_scale_ptr_arr[idx] = input_scale + std::min(token_ids, size_m - 1) * stride_asm; // 计算M方向的偏移
  }

  float b_scale[(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
    {
      b_scale[idx] = expert_b_scale;
    }
  }
  else
  {
    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr);
#pragma unroll
    for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
    {

      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 0], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 0);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 1], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 4);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 2], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 8);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 3], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 12);
    }
  }

  {
    if (size_k == 7168)
    {
      constexpr static int SIZE_K = 7168;
      gemm_nt_marlin_decode_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 6144)
    {
      constexpr static int SIZE_K = 6144;
      gemm_nt_marlin_decode_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 4096)
    {
      constexpr static int SIZE_K = 4096;
      gemm_nt_marlin_decode_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 3072)
    {
      constexpr static int SIZE_K = 3072;
      gemm_nt_marlin_decode_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 2048)
    {
      constexpr static int SIZE_K = 2048;
      gemm_nt_marlin_decode_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
  }
  if (warp_k_id == 0 || warp_k_num == 1)
  {
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      float *a_scale = a_scale_ptr_arr[min_tile_m];
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * a_scale[0] * b_scale[min_tile_n * 4 + reg_id];
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
          output_lds[index] = b32_to_b16<scalar_t>(value);
        }
      }
    }
  }

  __syncthreads();

  {
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;
      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) =
            *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN_N160_FP8(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{

  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

  const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    int token_ids = sorted_token_ids_element & 0x00FFFFFF;
    int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
    int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1);
    float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
    weight_dot_a_scale[idx] = topk_weights[token_index_safe] * input_scale_value;
  }

  float weight_dot_a_scale_value[WARP_M / mfma_m];
#pragma unroll
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
  {
    weight_dot_a_scale_value[min_tile_m] = weight_dot_a_scale[min_tile_m];
  }

  constexpr int n_loop_num = 1;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
      for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
      {
        b_scale[n_loop][idx] = expert_b_scale;
      }
    }
  }
  else
  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
        vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 0], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 0);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 1], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 4);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 2], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 8);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 3], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 12);
      }
    }
  }

  int token_index_store[BLOCK_SIZE_M * BLOCK_SIZE_N / (WARP_NUM * 512)];
  int tok_ids_store[BLOCK_SIZE_M * BLOCK_SIZE_N / (WARP_NUM * 512)];
  {
    int tid = threadIdx.x;
    int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

    int n_idx = threadIdx.x % N_thread; // tid % 4 = 0-3
    int m_idx = threadIdx.x / N_thread; // tid / 4 = 0-32
    int it_num = (WARP_NUM * 64) / N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += it_num)
    {
      int it = m_idx / it_num;
      const int32_t sorted_token_ids_element_store = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids_store = sorted_token_ids_element_store & 0x00FFFFFF;
      tok_ids_store[it] = (sorted_token_ids_element_store & 0xFF000000) >> 24;
      token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2

  gemm_nt_marlin_decode_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, real_topk);

  __syncthreads();

  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {

    if (warp_k_id == 0 || warp_k_num == 1)
    {

#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
        {
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            float value = C_reg[n_loop][min_tile_m * (WARP_N / mfma_n) + min_tile_n][reg_id] * weight_dot_a_scale_value[min_tile_m] * b_scale[n_loop][min_tile_n * 4 + reg_id];
            int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 8 /*padding*/; // + (lane_id & 15 )/2 * 2/*padding*/
            output_lds[index] = b32_to_b16<scalar_t>(value);
          }
        }
      }
    }
    __syncthreads();

    {

      const int tid = threadIdx.x;
      constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

      int n_idx = threadIdx.x % N_thread; // tid % 4 = 0-3
      int m_idx = threadIdx.x / N_thread; // tid / 4 = 0-32
      int it_num = (WARP_NUM * 64) / N_thread;
      for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
      {
        int it = m_idx / it_num;
        if (tok_ids_store[it] < real_topk)
        {
          *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[bidz * size_m * top_k * size_n + token_index_store[it] * size_n + n_idx * 8 + BLOCK_SIZE_N * n_loop]) =
              *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 8 /*padding*/]);
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    int N_LOOP_NUM,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_N160_FP8(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{

  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

  const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index_safe = std::min(uint32_t(token_ids /* top_k */), size_m - 1);
      float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
      weight_dot_a_scale[idx][i] = /* topk_weights[token_index_safe] * */ input_scale_value;
    }
  }

  constexpr int n_loop_num = N_LOOP_NUM;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                        // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                                   // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse /* * stride_bsn */ * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t *g_output;
  g_output = output + output_offset;

  constexpr int store_size = WARP_M / 4;
  int token_index_store[store_size];
  int tok_ids_store[store_size];
  int32_t sorted_token_ids_element_store[store_size];
  auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

  {

    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      for (int i = 0; i < 4; i++)
      {
        int m_idx = min_tile_m * 16 + i * 4 + col_id;
        int it = m_idx / 4;
        inline_buffer_load_dword(sorted_token_ids_element_store[it], m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
      for (int idx = 0; idx < WARP_N / mfma_n; idx++)
      {
        b_scale[n_loop][idx] = expert_b_scale;
      }
    }
  }
  else
  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
        for (int i = 0; i < 32 / mfma_n; i++)
        {

          vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

          inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i], row_id * 2, b_scale_ptr_prepared, min_tile_n * 32 + i);
        }
      }
    }
  }

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  __builtin_amdgcn_sched_barrier(0);

  if (size_k == 4096)
  {
    constexpr int SIZE_K = 4096;
    gemm_nt_marlin_prefill_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 6144)
  {
    constexpr int SIZE_K = 6144;
    gemm_nt_marlin_prefill_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 7168)
  {
    constexpr int SIZE_K = 7168;
    gemm_nt_marlin_prefill_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 3072)
  {
    constexpr int SIZE_K = 3072;
    gemm_nt_marlin_prefill_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, weight_dot_a_scale, b_scale, tmp, real_topk);
  }

  __builtin_amdgcn_sched_barrier(0);

  scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {

#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {
#pragma unroll
      for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
      {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {

          const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t> /* f32_to_bf16  */ (tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
      {
        int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
          if (tok_ids_store[it] < real_topk)
          {

            *(vec_element_2<scalar_t> *)(&g_output[token_index_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
          }
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_N160_FP8(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{

  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

  const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  constexpr int n_loop_num = 2;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t *g_output;
  g_output = output + output_offset;

  constexpr int store_size = WARP_M / 4;
  int token_index_store[store_size];
  int tok_ids_store[store_size];
  int32_t sorted_token_ids_element_store[store_size];
  auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

  {

    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      for (int i = 0; i < 4; i++)
      {
        int m_idx = min_tile_m * 16 + i * 4 + col_id;
        int it = m_idx / 4;
        inline_buffer_load_dword(sorted_token_ids_element_store[it], m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
      for (int idx = 0; idx < WARP_N / mfma_n; idx++)
      {
        b_scale[n_loop][idx] = expert_b_scale;
      }
    }
  }
  else
  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
        for (int i = 0; i < 32 / mfma_n; i++)
        {

          vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

          inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i], row_id * 2, b_scale_ptr_prepared, min_tile_n * 32 + i);
        }
      }
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2

  if (size_k == 128)
  {
    constexpr int SIZE_K = 128;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 256)
  {
    constexpr int SIZE_K = 256;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 320)
  {
    constexpr int SIZE_K = 320;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 384)
  {
    constexpr int SIZE_K = 384;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 512)
  {
    constexpr int SIZE_K = 512;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 640)
  {
    constexpr int SIZE_K = 640;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 768)
  {
    constexpr int SIZE_K = 768;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 1024)
  {
    constexpr int SIZE_K = 1024;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 1280)
  {
    constexpr int SIZE_K = 1280;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_2_n160_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
  }

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1);
      float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
      weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
    }
  }

  scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {

#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {
#pragma unroll
      for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
      {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {

          const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            float tmp = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + i];
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t> /* f32_to_bf16  */ (tmp);
          }
        }
      }
    }
  }

#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
      {
        int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
          if (tok_ids_store[it] < real_topk)
          {

            *(vec_element_2<scalar_t> *)(&g_output[token_index_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
          }
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M,
    uint16_t BLOCK_SIZE_N,
    uint16_t BLOCK_SIZE_K,
    uint16_t WARP_M,
    uint16_t WARP_N,
    uint16_t WARP_K,
    uint16_t GROUP_N,
    uint16_t GROUP_K,
    int STAGES,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP_FP8(
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.x; // pid_n
  const int bidz = blockIdx.y; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;                                          // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];                                               // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;                         // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 64;                       // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  vec4_fp32 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    uint32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
    a_scale_ptr_arr[idx] = input_scale + std::min(token_ids, size_m - 1) * stride_asm; // 计算M方向的偏移
  }

  float b_scale[(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
    {
      b_scale[idx] = expert_b_scale;
    }
  }
  else
  {
    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr);
#pragma unroll
    for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
    {

      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 0], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 0);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 1], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 4);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 2], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 8);
      inline_buffer_load_dword(b_scale[min_tile_n * 4 + 3], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 12);
    }
  }

  {
    if (size_k == 7168)
    {
      constexpr static int SIZE_K = 7168;
      gemm_nt_marlin_decode_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 6144)
    {
      constexpr static int SIZE_K = 6144;
      gemm_nt_marlin_decode_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 4096)
    {
      constexpr static int SIZE_K = 4096;
      gemm_nt_marlin_decode_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 3072)
    {
      constexpr static int SIZE_K = 3072;
      gemm_nt_marlin_decode_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 2048)
    {
      constexpr static int SIZE_K = 2048;
      gemm_nt_marlin_decode_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
  }
  if (warp_k_id == 0 || warp_k_num == 1)
  {
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      float *a_scale = a_scale_ptr_arr[min_tile_m];
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * a_scale[0] * b_scale[min_tile_n * 4 + reg_id];
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
          output_lds[index] = b32_to_b16<scalar_t>(value);
        }
      }
    }
  }

  __syncthreads();

  {
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;
      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) =
            *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN_FP8(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{

  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

  const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移

  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];                             // 声明lds信息
  Element *input_lds = (Element *)&(smem);                      // decode这里没用
  Element *qweight_lds = input_lds;                             // decode这里没用
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds

  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    int token_ids = sorted_token_ids_element & 0x00FFFFFF;
    int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
    int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1);
    float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
    weight_dot_a_scale[idx] = topk_weights[token_index_safe] * input_scale_value;
  }

  float weight_dot_a_scale_value[WARP_M / mfma_m];
#pragma unroll
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
  {
    weight_dot_a_scale_value[min_tile_m] = weight_dot_a_scale[min_tile_m];
  }

  constexpr int n_loop_num = 1;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
      for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
      {
        b_scale[n_loop][idx] = expert_b_scale;
      }
    }
  }
  else
  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
        vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 0], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 0);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 1], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 4);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 2], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 8);
        inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 4 + 3], col_id, b_scale_ptr_prepared, min_tile_n * mfma_n + 12);
      }
    }
  }

  int token_index_store[BLOCK_SIZE_M * BLOCK_SIZE_N / (WARP_NUM * 512)];
  int tok_ids_store[BLOCK_SIZE_M * BLOCK_SIZE_N / (WARP_NUM * 512)];
  {
    int tid = threadIdx.x;
    int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

    int n_idx = threadIdx.x % N_thread; // tid % 4 = 0-3
    int m_idx = threadIdx.x / N_thread; // tid / 4 = 0-32
    int it_num = (WARP_NUM * 64) / N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += it_num)
    {
      int it = m_idx / it_num;
      const int32_t sorted_token_ids_element_store = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      int token_ids_store = sorted_token_ids_element_store & 0x00FFFFFF;
      tok_ids_store[it] = (sorted_token_ids_element_store & 0xFF000000) >> 24;
      token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2

  gemm_nt_marlin_decode_2_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, real_topk);

  __syncthreads();

  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {

    if (warp_k_id == 0 || warp_k_num == 1)
    {

#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
        {
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            float value = C_reg[n_loop][min_tile_m * (WARP_N / mfma_n) + min_tile_n][reg_id] * weight_dot_a_scale_value[min_tile_m] * b_scale[n_loop][min_tile_n * 4 + reg_id];
            int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 8 /*padding*/; // + (lane_id & 15 )/2 * 2/*padding*/
            output_lds[index] = b32_to_b16<scalar_t>(value);
          }
        }
      }
    }
    __syncthreads();

    {

      const int tid = threadIdx.x;
      constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

      int n_idx = threadIdx.x % N_thread; // tid % 4 = 0-3
      int m_idx = threadIdx.x / N_thread; // tid / 4 = 0-32
      int it_num = (WARP_NUM * 64) / N_thread;
      for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
      {
        int it = m_idx / it_num;
        if (tok_ids_store[it] < real_topk)
        {
          *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[bidz * size_m * top_k * size_n + token_index_store[it] * size_n + n_idx * 8 + BLOCK_SIZE_N * n_loop]) =
              *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 8 /*padding*/]);
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    int N_LOOP_NUM,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_FP8(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  const uint32_t valid_route_count = size_m * real_topk;
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= valid_route_count ||
      bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];
  constexpr int n_loop_num = N_LOOP_NUM;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t *g_output = output + output_offset;

  if (expert_id == -1)
  {
    constexpr int N_thread = BLOCK_SIZE_N / 8;
    vec_element_8<scalar_t> zero_element_8;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;
    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      if (sorted_token_ids_element < valid_route_count)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[sorted_token_ids_element * size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  auto g_input = input;
  auto g_input_scale = input_scale;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];
  Element *input_lds = (Element *)&(smem);
  Element *qweight_lds = input_lds;
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem));
  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][1];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][1];

  if (size_k != 2048 && size_k != 3072 && size_k != 4096 && size_k != 6144 && size_k != 7168)
  {
    return;
  }

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;
  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;

  constexpr int store_size = WARP_M / mfma_m;

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);
  moe_w8a8_setprio_high();
  if (size_k == 4096)
  {
    constexpr int SIZE_K = 4096;
    gemm_nt_marlin_prefill_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 6144)
  {
    constexpr int SIZE_K = 6144;
    gemm_nt_marlin_prefill_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 7168)
  {
    constexpr int SIZE_K = 7168;
    gemm_nt_marlin_prefill_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 3072)
  {
    constexpr int SIZE_K = 3072;
    gemm_nt_marlin_prefill_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, nullptr, nullptr, real_topk);
  }
  moe_w8a8_setprio_normal();
  __builtin_amdgcn_sched_barrier(0);

  int32_t sorted_token_ids_element_store[store_size];
  auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

#pragma unroll
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
  {
    int m_idx = min_tile_m * mfma_m + row_id;
    inline_buffer_load_dword(sorted_token_ids_element_store[min_tile_m], m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
  }

  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;
  const float expert_b_scale = tensorwise_scale ? weight_scale[expert_id] : 0.0f;

  float weight_dot_a_scale[WARP_M / mfma_m];
  float b_scale[n_loop_num][(WARP_N / mfma_n) * 4];

#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);
#pragma unroll
    for (int min_tile_n = 0; min_tile_n < (WARP_N / mfma_n); min_tile_n++)
    {
#pragma unroll
      for (int reg_id = 0; reg_id < 4; reg_id++)
      {
        if (tensorwise_scale)
        {
          b_scale[n_loop][min_tile_n * 4 + reg_id] = expert_b_scale;
        }
        else
        {
          inline_buffer_load_dword(
              b_scale[n_loop][min_tile_n * 4 + reg_id],
              col_id * 4 + reg_id,
              b_scale_ptr_prepared,
              min_tile_n * mfma_n);
        }
      }
    }
  }
  vmcnt_only_wait(0);

  uint64_t store_row_offset[store_size];
  bool store_valid[store_size];
#pragma unroll
  for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
  {
    const int token_index = sorted_token_ids_element_store[min_tile_m];
    int token_index_safe = std::min(
        uint32_t(moe_w8a8_route_token_id(token_index, real_topk)),
        size_m - 1);
    weight_dot_a_scale[min_tile_m] = *(input_scale + token_index_safe * stride_asm);
    store_valid[min_tile_m] = token_index < valid_route_count;
    store_row_offset[min_tile_m] = static_cast<uint64_t>(token_index) * size_n;
  }

  union
  {
    scalar_t scalar_array[4];
    uint64_t uint64_value;
  } store_value;

#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {
#pragma unroll
      for (int min_tile_n = 0; min_tile_n < (WARP_N / mfma_n); min_tile_n++)
      {
        const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n;
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          float tmp_val = C_reg[n_loop][tile_idx][reg_id] *
                          weight_dot_a_scale[min_tile_m] *
                          b_scale[n_loop][min_tile_n * 4 + reg_id];
          store_value.scalar_array[reg_id] = b32_to_b16<scalar_t>(tmp_val);
        }
        int n_offset = min_tile_n * mfma_n + col_id * 4 +
                       warp_n_id * WARP_N + BLOCK_SIZE_N * n_loop;
        if (store_valid[min_tile_m])
        {

          *(uint64_t *)(&g_output[store_row_offset[min_tile_m] + n_offset]) = store_value.uint64_value;
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    int N_LOOP_NUM,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_FP8(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
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
    bool tensorwise_scale)
{
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  const uint32_t valid_route_count = size_m;
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= valid_route_count ||
      bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;

  auto g_input = input;
  auto g_input_scale = input_scale;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];
  Element *input_lds = (Element *)&(smem);
  Element *qweight_lds = input_lds;
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem));
  float *b_scale_lds = (float *)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][1];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][1];

  constexpr int n_loop_num = N_LOOP_NUM;

  if (size_k != 128 && size_k != 256 && size_k != 320 && size_k != 384 && size_k != 512 &&
      size_k != 640 && size_k != 768 && size_k != 1024 && size_k != 1280 && size_k != 2048)
  {
    return;
  }

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  { // EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    vec_element_8<scalar_t> zero_element_8;

#pragma unroll
    for (int i = 0; i < 8; ++i)
      zero_element_8.data[i] = 0;

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      if (sorted_token_ids_element < valid_route_count)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[sorted_token_ids_element * size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  constexpr int store_size = WARP_M / mfma_m;

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  moe_w8a8_setprio_high();
  if (size_k == 128)
  {
    constexpr int SIZE_K = 128;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 256)
  {
    constexpr int SIZE_K = 256;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 320)
  {
    constexpr int SIZE_K = 320;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 384)
  {
    constexpr int SIZE_K = 384;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 512)
  {
    constexpr int SIZE_K = 512;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 640)
  {
    constexpr int SIZE_K = 640;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 768)
  {
    constexpr int SIZE_K = 768;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 1024)
  {
    constexpr int SIZE_K = 1024;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 1280)
  {
    constexpr int SIZE_K = 1280;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_2_n160_fp8_bk64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, nullptr, nullptr, nullptr, real_topk);
  }
  moe_w8a8_setprio_normal();

  int32_t sorted_token_ids_element_store[store_size];
  auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

#pragma unroll
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
  {
    int m_idx = min_tile_m * mfma_m + row_id;
    inline_buffer_load_dword(sorted_token_ids_element_store[min_tile_m], m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
  }

  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n) * 4];

  if (tensorwise_scale)
  {
    const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
      for (int idx = 0; idx < (WARP_N / mfma_n) * 4; idx++)
      {
        b_scale[n_loop][idx] = expert_b_scale;
      }
    }
  }
  else
  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
      vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);
#pragma unroll
      for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
      {
#pragma unroll
        for (int elem = 0; elem < 8; elem++)
        {
          inline_buffer_load_dword(
              b_scale[n_loop][n_tile * 8 + elem],
              col_id * 8 + elem,
              b_scale_ptr_prepared,
              n_tile * 32);
        }
      }
    }
  }

  float weight_dot_a_scale[WARP_M / mfma_m];
  uint64_t store_row_offset[store_size];
  bool store_valid[store_size];
  vmcnt_only_wait(0);
#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    int token_index = sorted_token_ids_element_store[idx];
    int token_index_safe = std::min(uint32_t(token_index), size_m - 1);
    float input_scale_value = *(input_scale + token_index_safe * stride_asm);
    weight_dot_a_scale[idx] = topk_weights[token_index_safe] * input_scale_value;
    store_valid[idx] = token_index < valid_route_count;
    store_row_offset[idx] = static_cast<uint64_t>(token_index) * size_n;
  }

#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
    {
#pragma unroll
      for (int n_tile = 0; n_tile < (WARP_N / 32); n_tile++)
      {
        union
        {
          scalar_t scalar_array[8];
          vec4_uint uint4_value;
        } store_value;
#pragma unroll
        for (int it = 0; it < 2; it++)
        {
          const int min_tile_n = n_tile * 2 + it;
          const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n;
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            float tmp_val = C_reg[n_loop][tile_idx][reg_id] *
                            weight_dot_a_scale[min_tile_m] *
                            b_scale[n_loop][n_tile * 8 + it + reg_id * 2];
            store_value.scalar_array[it + reg_id * 2] = b32_to_b16<scalar_t>(tmp_val);
          }
        }
        int n_offset = n_tile * 32 + col_id * 8 +
                       warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
        if (store_valid[min_tile_m])
        {
          *reinterpret_cast<vec4_uint *>(
              &g_output[store_row_offset[min_tile_m] + n_offset]) = store_value.uint4_value;
        }
      }
    }
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill(const GemmParams<T, T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4; // max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  { // marlin版本

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.y = 1;                                                                                      // k方向

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                    GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill(const GemmParams<T, T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  constexpr int n_loop_num = 4;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
  gridDim.x = 1;                                                // k方向

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;

  const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                      GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_decode(const GemmParams<T, T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4; // max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  { // marlin版本

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.y = 1;                                                                                      // k方向

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                   GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_decode(const GemmParams<T, T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  constexpr int n_loop_num = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
  gridDim.x = 1;                                                // k方向

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 2 + BLOCK_SIZE_M / 2 * 16;

  const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                     GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill_fp8_nloop(const GemmParams<T, T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = N_LOOP_NUM;
  static_assert(N_LOOP_NUM > 0, "N_LOOP_NUM must be positive");

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向

  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;

  gridDim.y = DIVIDE(params.size_n, n_loop_num * BLOCK_SIZE_N); // n方向
  gridDim.x = 1;                                                // k方向

  MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                      GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
      params.ptr_A,
      params.ptr_B0,
      params.ptr_C,
      params.ptr_A_scale,
      params.ptr_B_scale,
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids,
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.stride_asm,
      params.stride_ask,
      params.stride_bse,
      params.stride_bsn,
      params.stride_bsk,
      params.sorted_token_lens,
      params.top_k,
      params.real_topk,
      params.tensorwise_scale);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill_fp8(const GemmParams<T, T_hidden> &params)
{
  if (params.size_n == 640)
  {
    launch_moe_w8a8_first_stage_prefill_fp8_nloop<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, 1, T, T_hidden>(params);
  }
  else
  {
    launch_moe_w8a8_first_stage_prefill_fp8_nloop<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, 2, T, T_hidden>(params);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill_fp8_nloop(const GemmParams<T, T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = N_LOOP_NUM;
  static_assert(N_LOOP_NUM > 0, "N_LOOP_NUM must be positive");
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
  gridDim.x = 1;                                                // k方向

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;

  const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                          GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill_fp8(const GemmParams<T, T_hidden> &params)
{
  launch_moe_w8a8_second_stage_prefill_fp8_nloop<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, 4, T, T_hidden>(params);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_decode_fp8(const GemmParams<T, T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4; // max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  { // marlin版本

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.y = 1;                                                                                      // k方向

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                       GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_decode_fp8(const GemmParams<T, T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  constexpr int n_loop_num = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
  gridDim.x = 1;                                                // k方向

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 2 + BLOCK_SIZE_M / 2 * 16;

  const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                         GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_prefill_n160_fp8(const GemmParams<T, T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = 20 * 1024;
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.size_n == 640)
  {
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向

    constexpr int n_loop_num = 1;
    if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
      return;

    gridDim.y = DIVIDE(params.size_n, n_loop_num * BLOCK_SIZE_N); // n方向
    gridDim.x = 1;                                                // k方向

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_N160_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                             GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
  else
  { // marlin版本

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向

    constexpr int n_loop_num = 2;
    if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
      return;

    gridDim.y = DIVIDE(params.size_n, n_loop_num * BLOCK_SIZE_N); // n方向
    gridDim.x = 1;                                                // k方向

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_N160_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                             GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_prefill_n160_fp8(const GemmParams<T, T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  constexpr int n_loop_num = 2;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);                                       // n方向
  gridDim.x = 1;                                                                                      // k方向

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;

  const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_N160_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                               GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_first_stage_decode_n160_fp8(const GemmParams<T, T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4; // max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  { // marlin版本

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.x = 1;                                                                                      // k方向

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP_N160_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                            GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_second_stage_decode_n160_fp8(const GemmParams<T, T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  constexpr int n_loop_num = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
  gridDim.x = 1;                                                // k方向

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 2 + BLOCK_SIZE_M / 2 * 16;

  const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {

    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN_N160_FP8<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                              GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
  }
}

#endif
