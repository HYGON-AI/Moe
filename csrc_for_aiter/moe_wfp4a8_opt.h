// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT


#ifndef MOE_WFP4A8_OPT_HIP_H
#define MOE_WFP4A8_OPT_HIP_H

#include "moe_wfp4a8_utils.h"
#include "moe_wfp4a8_config.h"

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
    bool GroupwisePost,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP(
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const uint8_t *__restrict__ weight_scale_u8,
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
    uint32_t real_topk)
{
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.x;
  const int bidz = blockIdx.y;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;
  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N / 32 * 32 * 32;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;
  const uint64_t weight_scale_offset = stride_bse * expert_id + bidy * BLOCK_SIZE_N;

  auto g_input = input;
  auto g_input_scale = input_scale;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index = token_ids * real_topk + topk_ids;

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

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  float *scale_lds = reinterpret_cast<float *>(
      reinterpret_cast<uint8_t *>(input_lds) + BLOCK_SIZE_M * WARP_K * 2);
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem));

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];

  auto g_qweight = qweight + qweight_offset;
  float *g_weight_scale = nullptr;
  const uint8_t *g_weight_scale_u8 = nullptr;
  if constexpr (GroupwisePost)
  {
    g_weight_scale_u8 = weight_scale_u8 + stride_bse * expert_id + bidy * BLOCK_SIZE_N * stride_bsn;
  }
  else
  {
    g_weight_scale = weight_scale + weight_scale_offset;
  }
  vec4_fp32 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *b_scale_ptr = nullptr;
  if constexpr (!GroupwisePost)
  {
    b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;
  }

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    const int token_id = sorted_token_ids_element / int(top_k);

    a_scale_ptr_arr[idx] = input_scale + std::min(token_id, int(size_m - 1)) * stride_asm;
  }

  float b_scale[(WARP_N / mfma_n) * 4];

  if constexpr (!GroupwisePost)
  {
    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr);
#pragma unroll
    for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
    {
#pragma unroll
      for (int it = 0; it < 2; it++)
      {

        b_scale[(min_tile_n * 2 + it) * 4 + 0] = b_scale_ptr[(min_tile_n * 32 + it) + 0 + col_id * 2];
        b_scale[(min_tile_n * 2 + it) * 4 + 1] = b_scale_ptr[(min_tile_n * 32 + it) + 8 + col_id * 2];
        b_scale[(min_tile_n * 2 + it) * 4 + 2] = b_scale_ptr[(min_tile_n * 32 + it) + 16 + col_id * 2];
        b_scale[(min_tile_n * 2 + it) * 4 + 3] = b_scale_ptr[(min_tile_n * 32 + it) + 24 + col_id * 2];
      }
    }
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

  {

    gemm_nt_marlin_decode_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, GroupwisePost>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, size_n, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
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

        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  if constexpr (GroupwisePost)
  {
    if (warp_k_id == 0 || warp_k_num == 1)
    {
#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
        {
#pragma unroll
          for (int it = 0; it < 2; it++)
          {
            const int tile_idx = min_tile_m * WARP_N / mfma_n + min_tile_n * 2 + it;
            const int index = warp_n_id * WARP_N + min_tile_n * 32 + row_id * 2 + it;
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {
              const int m_idx = min_tile_m * mfma_m + reg_id * 4 + col_id;
              const int32_t sorted_token_ids_element =
                  sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
              if (sorted_token_ids_element < size_m * top_k)
              {
                float value = C_reg[0][tile_idx][reg_id];
                g_output[sorted_token_ids_element * size_n + index] = b32_to_b16<scalar_t>(value);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    if (warp_k_id == 0 || warp_k_num == 1)
    {
#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
        float *a_scale = a_scale_ptr_arr[min_tile_m];
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
        {
#pragma unroll
          for (int it = 0; it < 2; it++)
          {
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {

              float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n * 2 + it][reg_id] * a_scale[0];
              value *= b_scale[(min_tile_n * 2 + it) * 4 + reg_id];

              int index = min_tile_m * mfma_m * BLOCK_SIZE_N + (min_tile_n) * 32 + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 8 + lane_id / 16 * 2 + it + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2;
              output_lds[index] = b32_to_b16<scalar_t>(value);
            }
          }
        }
      }
    }

    __syncthreads();

    {
      constexpr int N_thread = BLOCK_SIZE_N / 8;
      int m_idx = threadIdx.x / N_thread;
      int n_idx = threadIdx.x % N_thread;
      for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
      {
        const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
        if (sorted_token_ids_element < size_m * top_k)
        {
          *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[sorted_token_ids_element * size_n + n_idx * 8]) =
              *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2]);
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
    bool mul_topk_weight,
    bool AGroupScale = true>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_DECODE_UP(
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    const uint8_t *__restrict__ weight_scale_u8,
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
    uint32_t real_topk)
{
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.x;
  const int bidz = blockIdx.y;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;
  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N / 32 * 32 * 32;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;

  auto g_input = input;
  auto g_input_scale = input_scale;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index = token_ids * real_topk + topk_ids;

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

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem));

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];

  auto g_qweight = qweight + qweight_offset;
  const uint8_t *g_weight_scale_u8 =
      weight_scale_u8 + stride_bse * expert_id + bidy * BLOCK_SIZE_N * stride_bsn;
  vec4_fp32 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    const int token_id = sorted_token_ids_element / int(top_k);
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

  {

    gemm_nt_marlin_decode_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, true, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, size_n, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
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

        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  if (warp_k_id == 0 || warp_k_num == 1)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
#pragma unroll
        for (int it = 0; it < 2; it++)
        {
          const int tile_idx = min_tile_m * WARP_N / mfma_n + min_tile_n * 2 + it;
          const int index = warp_n_id * WARP_N + min_tile_n * 32 + row_id * 2 + it;
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            const int m_idx = min_tile_m * mfma_m + reg_id * 4 + col_id;
            const int32_t sorted_token_ids_element =
                sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            if (sorted_token_ids_element < size_m * top_k)
            {
              float value = C_reg[0][tile_idx][reg_id];
              if constexpr (!AGroupScale)
              {
                const int token_id = sorted_token_ids_element / int(top_k);
                value *= input_scale[std::min(token_id, int(size_m - 1)) * stride_asm];
              }
              g_output[sorted_token_ids_element * size_n + index] = b32_to_b16<scalar_t>(value);
            }
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
    int N_LOOP_NUM,
    int FIXED_SIZE_K,
    bool mul_topk_weight,
    bool AGroupScale = true>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_PREFILL_DOWN(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    const uint8_t *__restrict__ weight_scale_u8,
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
    uint32_t real_topk)
{
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k ||
      bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const int32_t expert_id = expert_ids[bidx];
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;

  auto g_input = input;
  auto g_input_scale = input_scale;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
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
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  float *scale_lds = reinterpret_cast<float *>(
      reinterpret_cast<uint8_t *>(input_lds) + BLOCK_SIZE_M * WARP_K * 2);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];
  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element =
          sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4,
                                    int(sorted_token_lens - 1))];
      int token_index_safe = std::min((uint32_t)sorted_token_ids_element, size_m * top_k - 1);
      weight_dot_a_scale[idx][i] = topk_weights[token_index_safe];
      if constexpr (!AGroupScale)
      {
        weight_dot_a_scale[idx][i] *= input_scale[token_index_safe * stride_asm];
      }
    }
  }

  constexpr int n_loop_num = N_LOOP_NUM;
  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num;
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
      const int32_t sorted_token_ids_element =
          sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;
      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) =
            zero_element_8;
      }
    }
    return;
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

  constexpr int store_size = WARP_M / 4;
  int32_t sorted_token_ids_element_store[store_size];
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
  {
    for (int i = 0; i < 4; i++)
    {
      int m_idx = min_tile_m * 16 + i * 4 + col_id;
      int it = m_idx / 4;
      sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale_u8 =
      weight_scale_u8 + stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num * stride_bsn;
  float b_scale[n_loop_num][(WARP_N / mfma_n)] = {0};
  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if constexpr (FIXED_SIZE_K != 0 && FIXED_SIZE_K % 64 == 0)
  {
    constexpr int SIZE_K = FIXED_SIZE_K;
    gemm_nt_marlin_prefill_2_wfp4a8_fixed_k64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num, true, true, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, size_n, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk, scale_lds);
  }

  __builtin_amdgcn_sched_barrier(0);

  scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] =
                b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
  {
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
        int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
          if (sorted_token_ids_element_store[it] < size_m * top_k)
          {
            *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) =
                *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
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
    bool GroupwisePost,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const uint8_t *__restrict__ weight_scale_u8,
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
    uint32_t real_topk)
{

  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;

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
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  float *scale_lds = reinterpret_cast<float *>(
      reinterpret_cast<uint8_t *>(input_lds) + BLOCK_SIZE_M * WARP_K * 2);
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(input_lds);

  float *b_scale_lds = reinterpret_cast<float *>(input_lds);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_index_safe = std::min((uint32_t)sorted_token_ids_element, size_m * top_k - 1);
      float input_scale_value = *(input_scale + token_index_safe * stride_asm);
      weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
    }
  }

  constexpr int n_loop_num = 4;

  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

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
        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  float *g_weight_scale = nullptr;
  const uint8_t *g_weight_scale_u8 = nullptr;
  if constexpr (GroupwisePost)
  {
    g_weight_scale_u8 = weight_scale_u8 + stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num * stride_bsn;
  }
  else
  {
    g_weight_scale = weight_scale + weight_scale_offset;
  }
  float *b_scale_ptr = nullptr;
  if constexpr (!GroupwisePost)
  {
    b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;
  }

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

  if constexpr (!GroupwisePost)
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

          b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
        }
      }
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 1536)
  {
    constexpr int SIZE_K = 1536;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 1024)
  {
    constexpr int SIZE_K = 1024;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 768)
  {
    constexpr int SIZE_K = 768;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 512)
  {
    constexpr int SIZE_K = 512;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 384)
  {
    constexpr int SIZE_K = 384;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 256)
  {
    constexpr int SIZE_K = 256;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 192)
  {
    constexpr int SIZE_K = 192;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 128)
  {
    constexpr int SIZE_K = 128;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, GroupwisePost, true, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
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
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
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
          if (sorted_token_ids_element_store[it] < size_m * top_k)
          {

            *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
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
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP(
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
    uint32_t real_topk)
{

  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;

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

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_index_safe = std::min((uint32_t)sorted_token_ids_element, (size_m - 1) * top_k);
      float input_scale_value = *(input_scale + (token_index_safe / top_k) * stride_asm);
      weight_dot_a_scale[idx][i] = input_scale_value;
    }
  }

  constexpr int n_loop_num = 4;

  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;

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
        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
        for (int i = 0; i < 32 / mfma_n; i++)
        {

          b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
        }
      }
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if (size_k == 7168)
  {
    constexpr int SIZE_K = 7168;
    gemm_nt_marlin_prefill_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 6144)
  {
    constexpr int SIZE_K = 6144;
    gemm_nt_marlin_prefill_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 4096)
  {
    constexpr int SIZE_K = 4096;
    gemm_nt_marlin_prefill_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 3584)
  {
    constexpr int SIZE_K = 3584;
    gemm_nt_marlin_prefill_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 3072)
  {
    constexpr int SIZE_K = 3072;
    gemm_nt_marlin_prefill_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
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
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
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
          if (sorted_token_ids_element_store[it] < size_m * top_k)
          {

            *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
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
    bool mul_topk_weight,
    bool AGroupScale = true>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_DECODE_DOWN(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    const uint8_t *__restrict__ weight_scale_u8,
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
    uint32_t real_topk)
{
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;

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
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  float *scale_lds = reinterpret_cast<float *>(
      reinterpret_cast<uint8_t *>(input_lds) + BLOCK_SIZE_M * WARP_K * 2);
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(input_lds);

  float *b_scale_lds = reinterpret_cast<float *>(input_lds);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_index_safe = std::min((uint32_t)sorted_token_ids_element, size_m * top_k - 1);
      weight_dot_a_scale[idx][i] = topk_weights[token_index_safe];
      if constexpr (!AGroupScale)
      {
        weight_dot_a_scale[idx][i] *= input_scale[token_index_safe * stride_asm];
      }
    }
  }

  constexpr int n_loop_num = 4;

  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

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
        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  const uint8_t *g_weight_scale_u8 =
      weight_scale_u8 + stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num * stride_bsn;

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 1536)
  {
    constexpr int SIZE_K = 1536;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 1024)
  {
    constexpr int SIZE_K = 1024;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 768)
  {
    constexpr int SIZE_K = 768;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 512)
  {
    constexpr int SIZE_K = 512;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 384)
  {
    constexpr int SIZE_K = 384;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 256)
  {
    constexpr int SIZE_K = 256;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 192)
  {
    constexpr int SIZE_K = 192;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 128)
  {
    constexpr int SIZE_K = 128;
    gemm_nt_marlin_decode_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, true, AGroupScale, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, g_weight_scale_u8, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
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
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
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
          if (sorted_token_ids_element_store[it] < size_m * top_k)
          {

            *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
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
    int N_LOOP_NUM,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256(
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
    uint32_t real_topk)
{

  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;

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
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  float *scale_lds = reinterpret_cast<float *>(
      reinterpret_cast<uint8_t *>(input_lds) + BLOCK_SIZE_M * WARP_K * 2);
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(input_lds);

  float *b_scale_lds = reinterpret_cast<float *>(input_lds);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_index_safe = std::min((uint32_t)sorted_token_ids_element, (size_m - 1) * top_k);
      float input_scale_value = *(input_scale + (token_index_safe / top_k) * stride_asm);
      weight_dot_a_scale[idx][i] = input_scale_value;
    }
  }

  constexpr int n_loop_num = N_LOOP_NUM;

  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

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
        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

  {
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {

#pragma unroll
      for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
      {
        for (int i = 0; i < 32 / mfma_n; i++)
        {

          b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
        }
      }
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if (size_k == 7168)
  {
    constexpr int SIZE_K = 7168;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 6144)
  {
    constexpr int SIZE_K = 6144;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 4096)
  {
    constexpr int SIZE_K = 4096;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 3584)
  {
    constexpr int SIZE_K = 3584;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 3072)
  {
    constexpr int SIZE_K = 3072;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
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
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
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
          if (sorted_token_ids_element_store[it] < size_m * top_k)
          {

            *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
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
    int N_LOOP_NUM,
    bool mul_topk_weight,
    bool SplitBLoadK32 = false,
    bool ShortScaleMode = false,
    bool KOuterScale = false,
    bool FastScaleMode = true,
    bool DirectStore = false,
    bool PairKScale = false,
    bool U8ScaleReg = false,
    bool FastBRow = false,
    bool AGroupScale = true>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    const uint8_t *__restrict__ weight_scale_u8,
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
    uint32_t real_topk)
{

  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;

  auto g_input = input;
  auto g_input_scale = input_scale;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int n_loop_num = N_LOOP_NUM;

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
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  float *scale_lds = reinterpret_cast<float *>(
      reinterpret_cast<uint8_t *>(input_lds) + BLOCK_SIZE_M * WARP_K * 2);
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(input_lds);

  float *b_scale_lds = reinterpret_cast<float *>(input_lds);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_index_safe = std::min((uint32_t)sorted_token_ids_element, (size_m - 1) * top_k);
      weight_dot_a_scale[idx][i] = 1.0f;
      if constexpr (!AGroupScale)
      {
        const int token_id = sorted_token_ids_element / int(top_k);
        weight_dot_a_scale[idx][i] = input_scale[std::min(token_id, int(size_m - 1)) * stride_asm];
      }
    }
  }

  const uint32_t qweight_k_tile_bytes = 32;
  const uint64_t qweight_offset = expert_offset + bidy * qweight_k_tile_bytes * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  const uint32_t params_size_n_for_groupwise = size_n;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

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
        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale_u8 = weight_scale_u8 + stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num * stride_bsn;

  float b_scale[n_loop_num][(WARP_N / mfma_n)] = {0};

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if (size_k == 7168)
  {
    constexpr int SIZE_K = 7168;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t, true, false, true, false, false, false, true, false, false, false, false, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, params_size_n_for_groupwise, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk, g_weight_scale_u8, size_n);
  }
  else if (size_k == 6144)
  {
    constexpr int SIZE_K = 6144;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t, true, false, true, false, false, false, true, false, false, false, false, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, params_size_n_for_groupwise, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk, g_weight_scale_u8, size_n);
  }
  else if (size_k == 4096)
  {
    constexpr int SIZE_K = 4096;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t, true, false, true, false, false, false, true, false, false, false, false, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, params_size_n_for_groupwise, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk, g_weight_scale_u8, size_n);
  }
  else if (size_k == 3584)
  {
    constexpr int SIZE_K = 3584;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t, true, false, true, ShortScaleMode, DirectStore, false, FastScaleMode, true, true, KOuterScale, PairKScale, U8ScaleReg, FastBRow, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, params_size_n_for_groupwise, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk, g_weight_scale_u8, size_n, scale_lds);
  }
  else if (size_k == 3072)
  {
    constexpr int SIZE_K = 3072;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t, true, false, true, false, false, false, true, false, false, false, false, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, params_size_n_for_groupwise, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk, g_weight_scale_u8, size_n);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_wfp4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t, true, false, true, false, false, false, true, false, false, false, false, AGroupScale>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, nullptr, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, params_size_n_for_groupwise, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk, g_weight_scale_u8, size_n);
  }
  __builtin_amdgcn_sched_barrier(0);

  if constexpr (DirectStore)
  {
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
            if (sorted_token_ids_element_store[it] < size_m * top_k)
            {
              scalar_t store_value[2];
              const int tile_idx0 = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2;
              const int tile_idx1 = tile_idx0 + 1;
              store_value[0] = b32_to_b16<scalar_t>(
                  C_reg[n_loop][tile_idx0][reg_id] * weight_dot_a_scale[min_tile_m][reg_id]);
              store_value[1] = b32_to_b16<scalar_t>(
                  C_reg[n_loop][tile_idx1][reg_id] * weight_dot_a_scale[min_tile_m][reg_id]);
              *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) =
                  *(vec_element_2<scalar_t> *)(store_value);
            }
          }
        }
      }
    }
  }
  else
  {
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
              value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] =
                  b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
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
            if (sorted_token_ids_element_store[it] < size_m * top_k)
            {

              *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
            }
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
    int N_LOOP_NUM,
    int FIXED_SIZE_K,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN(
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
    uint32_t real_topk)
{

  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;

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
  uint32_t *fp4x2_lut_lds = reinterpret_cast<uint32_t *>(smem);
  Element *input_lds = reinterpret_cast<Element *>(
      reinterpret_cast<uint8_t *>(smem) + WFP4A8_FP4X2_LUT_BYTES);
  Element *qweight_lds = input_lds;
  scalar_t *output_lds = reinterpret_cast<scalar_t *>(input_lds);

  float *b_scale_lds = reinterpret_cast<float *>(input_lds);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    for (int i = 0; i < 4; i++)
    {
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_index_safe = std::min((uint32_t)sorted_token_ids_element, size_m * top_k - 1);
      float input_scale_value = *(input_scale + token_index_safe * stride_asm);
      weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
    }
  }

  constexpr int n_loop_num = N_LOOP_NUM;

  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t *g_output;
  g_output = output + output_offset;

  if (expert_id == -1)
  {
    const int tid = threadIdx.x;
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
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;

      if (topk_ids < real_topk)
      {
        *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  wfp4a8_init_fp4x2_lut(fp4x2_lut_lds);

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
        sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n)];

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

          b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
        }
      }
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
  __builtin_amdgcn_sched_barrier(0);

  float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  if constexpr (FIXED_SIZE_K != 0 && FIXED_SIZE_K % 64 == 0)
  {
    constexpr int SIZE_K = FIXED_SIZE_K;
    gemm_nt_marlin_prefill_2_wfp4a8_fixed_k64<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, fp4x2_lut_lds, g_input_scale, g_weight_scale, nullptr, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 3072)
  {
    constexpr int SIZE_K = 3072;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 2048)
  {
    constexpr int SIZE_K = 2048;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 1536)
  {
    constexpr int SIZE_K = 1536;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 1024)
  {
    constexpr int SIZE_K = 1024;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 768)
  {
    constexpr int SIZE_K = 768;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 512)
  {
    constexpr int SIZE_K = 512;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 384)
  {
    constexpr int SIZE_K = 384;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 256)
  {
    constexpr int SIZE_K = 256;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 192)
  {
    constexpr int SIZE_K = 192;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
  }
  else if (size_k == 128)
  {
    constexpr int SIZE_K = 128;
    gemm_nt_marlin_prefill_2_wfp4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
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
            value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
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
          if (sorted_token_ids_element_store[it] < size_m * top_k)
          {

            *(vec_element_2<scalar_t> *)(&g_output[sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
          }
        }
      }
    }
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill(const GemmParams_wfp4a8<T_hidden> &params)
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
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;

  const int shared_mem_size = lds_size;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {
    MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1N256(const GemmParams_wfp4a8<T_hidden> &params)
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
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;

  const int shared_mem_size = lds_size;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {
    MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                                GROUP_N, GROUP_K, STAGES, 2, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
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
        params.real_topk);
  }
}

template <int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1Nloop(const GemmParams_wfp4a8<T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = N_LOOP_NUM;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;
  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == true)
  {
    MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int WARP_M, bool ShortScaleMode, bool KOuterScale, bool FastScaleMode, bool DirectStore, bool PairKScale, bool U8ScaleReg, bool FastBRow, typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_groupwise_impl(const GemmParams_wfp4a8<T_hidden> &params)
{
  constexpr int BLOCK_SIZE_N = 128;
  constexpr int BLOCK_SIZE_K = 64;
  constexpr int WARP_N = 32;
  constexpr int WARP_K = 64;
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = 3;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;
  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == true && params.ptr_B_scale_u8 != nullptr)
  {
    MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                            GROUP_N, GROUP_K, 2, n_loop_num, mul_topk_weight, true, ShortScaleMode, KOuterScale, FastScaleMode, DirectStore, PairKScale, U8ScaleReg, FastBRow, false><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale_u8,
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int WARP_M, bool ShortScaleMode, bool KOuterScale, bool FastScaleMode, bool DirectStore, bool PairKScale, bool U8ScaleReg, bool FastBRow, typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_groupwise_qgroup_impl(const GemmParams_wfp4a8<T_hidden> &params)
{
  constexpr int BLOCK_SIZE_N = 128;
  constexpr int BLOCK_SIZE_K = 64;
  constexpr int WARP_N = 32;
  constexpr int WARP_K = 64;
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = 3;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;
  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == true && params.ptr_B_scale_u8 != nullptr)
  {
    MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                            GROUP_N, GROUP_K, 2, n_loop_num, mul_topk_weight, true, ShortScaleMode, KOuterScale, FastScaleMode, DirectStore, PairKScale, U8ScaleReg, FastBRow, true><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale_u8,
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
        params.real_topk);
  }
}

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_9537(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_prefill_groupwise_impl<64, 64, false, true, true, true, false, true, false>(params);
}

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1_groupwise_qgroup_9537(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_prefill_groupwise_qgroup_impl<64, 64, false, true, true, true, false, true, false>(params);
}

template <bool AGroupScale, typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_9524_impl(const GemmParams_wfp4a8<T_hidden> &params)
{
  constexpr int BLOCK_SIZE_M = 16;
  constexpr int BLOCK_SIZE_N = 128;
  constexpr int BLOCK_SIZE_K = 64;
  constexpr int WARP_M = 16;
  constexpr int WARP_N = 32;
  constexpr int WARP_K = 64;
  constexpr int STAGES = 2;
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  constexpr bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = 3;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int a_lds_bytes = BLOCK_SIZE_M * WARP_K * 2;
  constexpr int reduce_bytes =
      WARP_M * (WARP_N + 1) * n_loop_num * warp_n_num * (warp_k_num - 1) * sizeof(float);
  const int shared_mem_size =
      WFP4A8_FP4X2_LUT_BYTES + (reduce_bytes > a_lds_bytes ? reduce_bytes : a_lds_bytes);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == true && params.ptr_B_scale_u8 != nullptr)
  {
    MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                            GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight, false, false, false, true, false, false, false, false, AGroupScale><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale_u8,
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
        params.real_topk);
  }
}

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_9524(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_decode_groupwise_9524_impl<false>(params);
}

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9524(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_decode_groupwise_9524_impl<true>(params);
}

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_9121(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_decode_groupwise<16, 32, 64, 16, 32, 64, 4>(params);
}

template <typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup_9121(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup<16, 32, 64, 16, 32, 64, 4>(params);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_prefill_GEMM1N384(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_prefill_GEMM1Nloop<3, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES>(params);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill(const GemmParams_wfp4a8<T_hidden> &params)
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
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;

  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
    return;
  }
  else
  {
    MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                        GROUP_N, GROUP_K, STAGES, n_loop_num, 0, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
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
        params.real_topk);
  }
}

template <int FIXED_SIZE_K, int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_fixed_k(const GemmParams_wfp4a8<T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = N_LOOP_NUM;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;
  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == true)
  {
    MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                        GROUP_N, GROUP_K, STAGES, n_loop_num, FIXED_SIZE_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_K192(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_second_stage_prefill_fixed_k<192, 4, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES>(params);
}

template <bool AGroupScale, typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_groupwise_9523_impl(const GemmParams_wfp4a8<T_hidden> &params)
{
  constexpr int BLOCK_SIZE_M = 64;
  constexpr int BLOCK_SIZE_N = 128;
  constexpr int BLOCK_SIZE_K = 64;
  constexpr int WARP_M = 64;
  constexpr int WARP_N = 32;
  constexpr int WARP_K = 64;
  constexpr int STAGES = 2;
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  constexpr bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = 4;
  constexpr int FIXED_SIZE_K = 192;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;
  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == true && params.ptr_B_scale_u8 != nullptr)
  {
    MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_PREFILL_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                    GROUP_N, GROUP_K, STAGES, n_loop_num, FIXED_SIZE_K, mul_topk_weight, AGroupScale><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale_u8,
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
        params.real_topk);
  }
}

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_groupwise_9523(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_second_stage_prefill_groupwise_9523_impl<false>(params);
}

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_prefill_groupwise_qgroup_9523(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_second_stage_prefill_groupwise_9523_impl<true>(params);
}

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise_9525(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_second_stage_decode_groupwise<16, 128, 64, 16, 32, 64, 2>(params);
}

template <typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise_qgroup_9525(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_second_stage_decode_groupwise_qgroup<16, 128, 64, 16, 32, 64, 2>(params);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode(const GemmParams_wfp4a8<T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;
  const int shared_mem_size = lds_size + WFP4A8_FP4X2_LUT_BYTES;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
    return;
  }
  else
  {

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);
    gridDim.y = 1;

    MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                     GROUP_N, GROUP_K, STAGES, false, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        nullptr,
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, bool AGroupScale, typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_impl(const GemmParams_wfp4a8<T_hidden> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  constexpr bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;
  const int shared_mem_size = lds_size + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false || params.ptr_B_scale_u8 == nullptr)
  {
    return;
  }
  else
  {
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);
    gridDim.y = 1;

    MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_DECODE_UP<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                 GROUP_N, GROUP_K, STAGES, mul_topk_weight, AGroupScale><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale_u8,
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_decode_groupwise_impl<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, false>(params);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_first_stage_decode_groupwise_qgroup(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_first_stage_decode_groupwise_impl<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, true>(params);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode(const GemmParams_wfp4a8<T_hidden> &params)
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
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;

  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
    return;
  }
  else
  {

    MOE_WFP4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                       GROUP_N, GROUP_K, STAGES, false, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        nullptr,
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, bool AGroupScale, typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise_impl(const GemmParams_wfp4a8<T_hidden> &params)
{
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  constexpr bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  constexpr int n_loop_num = 4;
  gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
  if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
    return;
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;

  const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2 + WFP4A8_FP4X2_LUT_BYTES;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false || params.ptr_B_scale_u8 == nullptr)
  {
    return;
  }
  else
  {

    MOE_WFP4A8_GROUPWISE_MARLIN_HIP_NT_DECODE_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                   GROUP_N, GROUP_K, STAGES, mul_topk_weight, AGroupScale><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale_u8,
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
        params.real_topk);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_second_stage_decode_groupwise_impl<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, false>(params);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T_hidden>
void launch_moe_wfp4a8_second_stage_decode_groupwise_qgroup(const GemmParams_wfp4a8<T_hidden> &params)
{
  launch_moe_wfp4a8_second_stage_decode_groupwise_impl<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, true>(params);
}

#endif
