// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT



#include "moe_wfp4a16_utils.h"
#include "moe_wfp4a16_config.h"

template <
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
    bool INTERLEAVE,
    bool DEQUANT_16BIT,
    int32_t PACK_SCALES>
__global__ void __launch_bounds__(512, 1) GEMM_NT_1STAGE(
    const Element *input,
    const uint32_t *__restrict__ qweight,
    Element *__restrict__ output,
    uint32_t *__restrict__ weight_zeros,
    Element *__restrict__ weight_scale,
    const uint8_t *__restrict__ weight_scale_u8,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t delta)
{
  const int bidx = blockIdx.z; // pid_m
  const int bidy = blockIdx.x; // pid_n
  const int bidz = blockIdx.y; // pid_k

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  using Dtype = ScalarType<Element>;
  using scalar_t2 = typename ScalarType<Element>::scalar_t2;

  const int32_t expert_id = expert_ids[bidx];

  const uint64_t expert_offset_elements = (uint64_t)size_n * size_k * expert_id;
  const uint64_t block_n_offset_elements = (uint64_t)bidy * BLOCK_SIZE_N * size_k;
  const uint64_t total_elements_offset = expert_offset_elements + block_n_offset_elements;

  const uint64_t qweight_offset = total_elements_offset / 8;
  const uint64_t weight_scale_offset = total_elements_offset / GROUP_K;
  const uint64_t weight_zeros_offset = total_elements_offset / (GROUP_K * 8);

  auto g_input = input;
  Element *g_output = output + bidy * BLOCK_SIZE_N;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;

  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_m_id = warp_id / (warp_n_num * warp_k_num);
  int warp_nk_id = warp_id % (warp_n_num * warp_k_num);
  int warp_k_id = warp_nk_id % warp_k_num;
  int warp_n_id = warp_nk_id / warp_k_num;

  extern __shared__ uint16_t output_lds_tmp[];
  Element *output_lds = reinterpret_cast<Element *>(output_lds_tmp);
  const int *sorted_token_ids_offset = sorted_token_ids;

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale ? weight_scale + weight_scale_offset : nullptr;
  auto g_weight_scale_u8 = weight_scale_u8 ? weight_scale_u8 + weight_scale_offset : nullptr;
  auto g_weight_zeros = weight_zeros + weight_zeros_offset;

  floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0};
  float weight_dot_a_scale[WARP_M / mfma_m];

  if constexpr (mul_topk_weight)
  {
    int row_id = lane_id % 16;
#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++)
    {
      int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_index_safe = std::min(uint32_t(token_index), size_m * top_k - 1);
      weight_dot_a_scale[idx] = topk_weights[token_index_safe];
    }
  }
  gemm_nt_w4a16_stage1_b128_k64_marlinS_scale_ldx<WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, GROUP_K, Element, scalar_t2, INTERLEAVE, DEQUANT_16BIT, PACK_SCALES>(
      g_input, g_qweight, g_weight_zeros, g_weight_scale, size_m, size_n, C_reg, warp_id, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, warp_m_id);

  if constexpr (WARP_M == 16)
  {
    if (warp_k_id == 0 || warp_k_num == 1)
    {
#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
        int32_t m_idx_global = warp_m_id * WARP_M + min_tile_m * mfma_m + (lane_id & 15);
        const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx_global, int(sorted_token_lens - 1))];

        if (token_index < size_m * top_k)
        {
#pragma unroll
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
          {
            if constexpr (INTERLEAVE)
            {
              int32_t col = warp_n_id * WARP_N + min_tile_n * mfma_n + (lane_id / 16) * 4;
#pragma unroll
              for (int reg_id = 0; reg_id < 4; reg_id++)
              {
                g_output[token_index * size_n + col + reg_id] =
                    Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] *
                                                           weight_dot_a_scale[min_tile_m]
                                                     : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
              }
            }
            else
            {
#pragma unroll
              for (int32_t r_o = 0; r_o < 2; r_o++)
              {
#pragma unroll
                for (int32_t r_i = 0; r_i < 2; r_i++)
                {
                  int32_t reg_id = r_o * 2 + r_i;
                  int32_t col = warp_n_id * WARP_N + min_tile_n * mfma_n +
                                r_o * 8 + (lane_id / 16) * 2 + r_i;
                  g_output[token_index * size_n + col] =
                      Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] *
                                                             weight_dot_a_scale[min_tile_m]
                                                       : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
                }
              }
            }
          }
        }
      }
    }
  }
  else if constexpr (INTERLEAVE)
  {
    constexpr int32_t WARP_M_TILES = WARP_M / mfma_m;
    constexpr int32_t HALF_TILES_M = WARP_M_TILES / 2;
    constexpr int32_t WARP_M_HALF = HALF_TILES_M * mfma_m; // 对应 WARP_M / 2
    constexpr int32_t HALF_M = BLOCK_SIZE_M / 2;

#pragma unroll
    for (int phase = 0; phase < 2; ++phase)
    {

      int32_t start_tile_m = phase * HALF_TILES_M;
      int32_t end_tile_m = (phase + 1) * HALF_TILES_M;

      if (warp_k_id == 0 || warp_k_num == 1)
      {
        for (int min_tile_m = start_tile_m; min_tile_m < end_tile_m; min_tile_m++)
        {
          int32_t min_tile_m_local = min_tile_m - start_tile_m; // 局部 Tile 偏移，确保 r_lds 落在 [0, HALF_M) 内
          int32_t r_lds = warp_m_id * WARP_M_HALF + min_tile_m_local * mfma_m + (lane_id % 16);
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
          {
            constexpr int32_t SWZ_COLS = BLOCK_SIZE_N / 4;
            constexpr int32_t SWZ_ELEMENT = 2;
            int32_t c = (warp_n_id * WARP_N + min_tile_n * mfma_n) / 4 + (lane_id / 16);
            int32_t c_swz = (r_lds * SWZ_ELEMENT + c) % SWZ_COLS;
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {
              int32_t index = r_lds * BLOCK_SIZE_N + c_swz * 4 + reg_id;
              output_lds[index] = Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * weight_dot_a_scale[min_tile_m] : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
            }
          }
        }
      }

      __syncthreads();

      constexpr int32_t N_thread = BLOCK_SIZE_N / 8;
      int32_t m_idx_local = threadIdx.x / N_thread;
      int32_t n_idx = threadIdx.x % N_thread;

      for (; m_idx_local < HALF_M; m_idx_local += (WARP_NUM * 64) / N_thread)
      {
        int32_t warp_idx = m_idx_local / WARP_M_HALF;
        int32_t offset = m_idx_local % WARP_M_HALF;
        int32_t m_idx_global = warp_idx * WARP_M + (phase * WARP_M_HALF) + offset;

        const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx_global, int(sorted_token_lens - 1))];
        int32_t n_swz = (m_idx_local + n_idx) % N_thread;

        if (token_index < size_m * top_k)
        {
          *reinterpret_cast<vec_element_8<Element> *>(&g_output[token_index * size_n + n_idx * 8]) =
              *reinterpret_cast<vec_element_8<Element> *>(&output_lds[m_idx_local * BLOCK_SIZE_N + n_swz * 8]);
        }
      }

      __syncthreads();
    }
  }
  else
  {
    constexpr int32_t WARP_M_TILES = WARP_M / mfma_m;
    constexpr int32_t HALF_TILES_M = WARP_M_TILES / 2;
    constexpr int32_t WARP_M_HALF = HALF_TILES_M * mfma_m; // 对应 WARP_M / 2
    constexpr int32_t HALF_M = BLOCK_SIZE_M / 2;
    constexpr int32_t SWZ_ATOM = 2 * 4; // 8

#pragma unroll
    for (int phase = 0; phase < 2; ++phase)
    {

      int32_t start_tile_m = phase * HALF_TILES_M;
      int32_t end_tile_m = (phase + 1) * HALF_TILES_M;

      if (warp_k_id == 0 || warp_k_num == 1)
      {
        for (int min_tile_m = start_tile_m; min_tile_m < end_tile_m; min_tile_m++)
        {
          int32_t min_tile_m_local = min_tile_m - start_tile_m; // 局部 Tile 偏移，确保紧密压入 LDS 的前一半空间
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
          {
#pragma unroll
            for (int32_t r_o = 0; r_o < 2; r_o++)
            {
              for (int32_t r_i = 0; r_i < 2; r_i++)
              {
                int32_t row_lds = warp_m_id * WARP_M_HALF + min_tile_m_local * mfma_m + (lane_id & 15);
                int32_t col = warp_n_id * WARP_N + min_tile_n * mfma_n + r_o * 8 + (lane_id / 16) * 2 + r_i;
                int32_t col_swz = (row_lds * SWZ_ATOM + col) % BLOCK_SIZE_N;
                int32_t index = row_lds * BLOCK_SIZE_N + col_swz;
                int32_t reg_id = r_o * 2 + r_i;

                output_lds[index] = Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * weight_dot_a_scale[min_tile_m] : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
              }
            }
          }
        }
      }

      __syncthreads();

      {
        constexpr int N_thread = BLOCK_SIZE_N / 8;
        int32_t m_idx_local = threadIdx.x / N_thread; // 在 [0, HALF_M) 空间内迭代
        int32_t n_idx = threadIdx.x % N_thread;

        for (; m_idx_local < HALF_M; m_idx_local += (WARP_NUM * 64) / N_thread)
        {

          int32_t warp_idx = m_idx_local / WARP_M_HALF;
          int32_t offset = m_idx_local % WARP_M_HALF;
          int32_t m_idx_global = warp_idx * WARP_M + (phase * WARP_M_HALF) + offset;

          const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx_global, int(sorted_token_lens - 1))];

          if (token_index < size_m * top_k)
          {
            int32_t lds_offset = m_idx_local * BLOCK_SIZE_N + (m_idx_local * SWZ_ATOM + n_idx * 8) % BLOCK_SIZE_N;
            vec8_Element<Element> tmp = *reinterpret_cast<vec8_Element<Element> *>(&output_lds[lds_offset]);

            vec8_Element<Element> tmp1;
            tmp1[0] = tmp[0];
            tmp1[4] = tmp[1];
            tmp1[1] = tmp[2];
            tmp1[5] = tmp[3];
            tmp1[2] = tmp[4];
            tmp1[6] = tmp[5];
            tmp1[3] = tmp[6];
            tmp1[7] = tmp[7];

            *reinterpret_cast<vec8_Element<Element> *>(&g_output[token_index * size_n + n_idx * 8]) = tmp1;
          }
        }
      }

      if (phase == 0)
      {
        __syncthreads();
      }
    }

  } // end else
}

template <
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
    bool INTERLEAVE,
    bool DEQUANT_16BIT,
    int32_t PACK_SCALES,
    bool FP4_K_SPLIT = false>
__global__ void __launch_bounds__(512, 1) GEMM_NT_1STAGE_WFP4A16(
    const Element *input,
    const uint32_t *__restrict__ qweight,
    Element *__restrict__ output,
    uint32_t *__restrict__ weight_zeros,
    Element *__restrict__ weight_scale,
    const uint8_t *__restrict__ weight_scale_u8,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t delta)
{
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.x;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k ||
      bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  using Dtype = ScalarType<Element>;
  using scalar_t2 = typename ScalarType<Element>::scalar_t2;

  const int32_t expert_id = expert_ids[bidx];
  const uint64_t expert_offset_elements = (uint64_t)size_n * size_k * expert_id;
  const uint64_t block_n_offset_elements = (uint64_t)bidy * BLOCK_SIZE_N * size_k;
  const uint64_t total_elements_offset = expert_offset_elements + block_n_offset_elements;

  const uint64_t qweight_offset = total_elements_offset / 8;
  const uint64_t weight_scale_offset = total_elements_offset / GROUP_K;
  const uint64_t weight_zeros_offset = total_elements_offset / (GROUP_K * 8);

  auto g_input = input;
  Element *g_output = output + bidy * BLOCK_SIZE_N;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;

  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_m_id = warp_id / (warp_n_num * warp_k_num);
  int warp_nk_id = warp_id % (warp_n_num * warp_k_num);
  int warp_k_id = warp_nk_id % warp_k_num;
  int warp_n_id = warp_nk_id / warp_k_num;

  extern __shared__ uint16_t output_lds_tmp[];
  Element *output_lds = reinterpret_cast<Element *>(output_lds_tmp);
  const int *sorted_token_ids_offset = sorted_token_ids;

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale ? weight_scale + weight_scale_offset : nullptr;
  auto g_weight_scale_u8 = weight_scale_u8 ? weight_scale_u8 + weight_scale_offset : nullptr;
  auto g_weight_zeros = weight_zeros + weight_zeros_offset;

  floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0};
  float weight_dot_a_scale[WARP_M / mfma_m];

  if constexpr (mul_topk_weight)
  {
    int row_id = lane_id % 16;
#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++)
    {
      int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_index_safe = std::min(uint32_t(token_index), size_m * top_k - 1);
      weight_dot_a_scale[idx] = topk_weights[token_index_safe];
    }
  }

  gemm_nt_wfp4a16_stage1_b128_k64_marlinS_scale_ldx<
      WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N,
      WARP_K, GROUP_K, Element, scalar_t2, INTERLEAVE, DEQUANT_16BIT,
      PACK_SCALES, true, true, FP4_K_SPLIT>(
      g_input, g_qweight, g_weight_zeros, g_weight_scale, g_weight_scale_u8,
      size_m, size_n, C_reg, warp_id, size_k, top_k, sorted_token_ids_offset,
      sorted_token_lens, expert_id, bidx, warp_m_id);

  if constexpr (FP4_K_SPLIT && (BLOCK_SIZE_K / WARP_K > 1))
  {
    constexpr int32_t REDUCE_PADDED_N = WARP_N + 1;
    constexpr int32_t REDUCE_TILE_ELEMS = WARP_M * REDUCE_PADDED_N;
    float *reduce_lds = reinterpret_cast<float *>(output_lds_tmp);

    if (warp_k_id != 0)
    {
      const int32_t reduce_group = warp_m_id * warp_n_num + warp_n_id;
      const int32_t reduce_base =
          (reduce_group * (warp_k_num - 1) + (warp_k_id - 1)) * REDUCE_TILE_ELEMS;
#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
        const int32_t row = min_tile_m * mfma_m + (lane_id & 15);
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
        {
          const int32_t col_base = min_tile_n * mfma_n + (lane_id / 16) * 4;
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
          {
            reduce_lds[reduce_base + row * REDUCE_PADDED_N + col_base + reg_id] =
                C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id];
          }
        }
      }
    }

    __syncthreads();

    if (warp_k_id == 0)
    {
      const int32_t reduce_group = warp_m_id * warp_n_num + warp_n_id;
#pragma unroll
      for (int k_part = 0; k_part < warp_k_num - 1; k_part++)
      {
        const int32_t reduce_base = (reduce_group * (warp_k_num - 1) + k_part) * REDUCE_TILE_ELEMS;
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
        {
          const int32_t row = min_tile_m * mfma_m + (lane_id & 15);
#pragma unroll
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
          {
            const int32_t col_base = min_tile_n * mfma_n + (lane_id / 16) * 4;
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {
              C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] +=
                  reduce_lds[reduce_base + row * REDUCE_PADDED_N + col_base + reg_id];
            }
          }
        }
      }
    }

    __syncthreads();
  }

  if constexpr (WARP_M == 16)
  {
    if (warp_k_id == 0 || warp_k_num == 1)
    {
#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
        int32_t m_idx_global = warp_m_id * WARP_M + min_tile_m * mfma_m + (lane_id & 15);
        const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx_global, int(sorted_token_lens - 1))];

        if (token_index < size_m * top_k)
        {
#pragma unroll
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
          {
            if constexpr (INTERLEAVE)
            {
              int32_t col = warp_n_id * WARP_N + min_tile_n * mfma_n + (lane_id / 16) * 4;
#pragma unroll
              for (int reg_id = 0; reg_id < 4; reg_id++)
              {
                g_output[token_index * size_n + col + reg_id] =
                    Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] *
                                                           weight_dot_a_scale[min_tile_m]
                                                     : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
              }
            }
            else
            {
#pragma unroll
              for (int32_t r_o = 0; r_o < 2; r_o++)
              {
#pragma unroll
                for (int32_t r_i = 0; r_i < 2; r_i++)
                {
                  int32_t reg_id = r_o * 2 + r_i;
                  int32_t col = warp_n_id * WARP_N + min_tile_n * mfma_n +
                                r_o * 8 + (lane_id / 16) * 2 + r_i;
                  g_output[token_index * size_n + col] =
                      Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] *
                                                             weight_dot_a_scale[min_tile_m]
                                                       : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
                }
              }
            }
          }
        }
      }
    }
  }
  else if constexpr (INTERLEAVE)
  {
    constexpr int32_t WARP_M_TILES = WARP_M / mfma_m;
    constexpr int32_t HALF_TILES_M = WARP_M_TILES / 2;
    constexpr int32_t WARP_M_HALF = HALF_TILES_M * mfma_m; // 对应 WARP_M / 2
    constexpr int32_t HALF_M = BLOCK_SIZE_M / 2;

#pragma unroll
    for (int phase = 0; phase < 2; ++phase)
    {

      int32_t start_tile_m = phase * HALF_TILES_M;
      int32_t end_tile_m = (phase + 1) * HALF_TILES_M;

      if (warp_k_id == 0 || warp_k_num == 1)
      {
        for (int min_tile_m = start_tile_m; min_tile_m < end_tile_m; min_tile_m++)
        {
          int32_t min_tile_m_local = min_tile_m - start_tile_m; // 局部 Tile 偏移，确保 r_lds 落在 [0, HALF_M) 内
          int32_t r_lds = warp_m_id * WARP_M_HALF + min_tile_m_local * mfma_m + (lane_id % 16);
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
          {
            constexpr int32_t SWZ_COLS = BLOCK_SIZE_N / 4;
            constexpr int32_t SWZ_ELEMENT = 2;
            int32_t c = (warp_n_id * WARP_N + min_tile_n * mfma_n) / 4 + (lane_id / 16);
            int32_t c_swz = (r_lds * SWZ_ELEMENT + c) % SWZ_COLS;
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {
              int32_t index = r_lds * BLOCK_SIZE_N + c_swz * 4 + reg_id;
              output_lds[index] = Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * weight_dot_a_scale[min_tile_m] : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
            }
          }
        }
      }

      __syncthreads();

      constexpr int32_t N_thread = BLOCK_SIZE_N / 8;
      int32_t m_idx_local = threadIdx.x / N_thread;
      int32_t n_idx = threadIdx.x % N_thread;

      for (; m_idx_local < HALF_M; m_idx_local += (WARP_NUM * 64) / N_thread)
      {
        int32_t warp_idx = m_idx_local / WARP_M_HALF;
        int32_t offset = m_idx_local % WARP_M_HALF;
        int32_t m_idx_global = warp_idx * WARP_M + (phase * WARP_M_HALF) + offset;

        const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx_global, int(sorted_token_lens - 1))];
        int32_t n_swz = (m_idx_local + n_idx) % N_thread;

        if (token_index < size_m * top_k)
        {
          *reinterpret_cast<vec_element_8<Element> *>(&g_output[token_index * size_n + n_idx * 8]) =
              *reinterpret_cast<vec_element_8<Element> *>(&output_lds[m_idx_local * BLOCK_SIZE_N + n_swz * 8]);
        }
      }

      __syncthreads();
    }
  }
  else
  {
    constexpr int32_t WARP_M_TILES = WARP_M / mfma_m;
    constexpr int32_t HALF_TILES_M = WARP_M_TILES / 2;
    constexpr int32_t WARP_M_HALF = HALF_TILES_M * mfma_m; // 对应 WARP_M / 2
    constexpr int32_t HALF_M = BLOCK_SIZE_M / 2;
    constexpr int32_t SWZ_ATOM = 2 * 4; // 8

#pragma unroll
    for (int phase = 0; phase < 2; ++phase)
    {

      int32_t start_tile_m = phase * HALF_TILES_M;
      int32_t end_tile_m = (phase + 1) * HALF_TILES_M;

      if (warp_k_id == 0 || warp_k_num == 1)
      {
        for (int min_tile_m = start_tile_m; min_tile_m < end_tile_m; min_tile_m++)
        {
          int32_t min_tile_m_local = min_tile_m - start_tile_m; // 局部 Tile 偏移，确保紧密压入 LDS 的前一半空间
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
          {
#pragma unroll
            for (int32_t r_o = 0; r_o < 2; r_o++)
            {
              for (int32_t r_i = 0; r_i < 2; r_i++)
              {
                int32_t row_lds = warp_m_id * WARP_M_HALF + min_tile_m_local * mfma_m + (lane_id & 15);
                int32_t col = warp_n_id * WARP_N + min_tile_n * mfma_n + r_o * 8 + (lane_id / 16) * 2 + r_i;
                int32_t col_swz = (row_lds * SWZ_ATOM + col) % BLOCK_SIZE_N;
                int32_t index = row_lds * BLOCK_SIZE_N + col_swz;
                int32_t reg_id = r_o * 2 + r_i;

                output_lds[index] = Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * weight_dot_a_scale[min_tile_m] : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
              }
            }
          }
        }
      }

      __syncthreads();

      {
        constexpr int N_thread = BLOCK_SIZE_N / 8;
        int32_t m_idx_local = threadIdx.x / N_thread; // 在 [0, HALF_M) 空间内迭代
        int32_t n_idx = threadIdx.x % N_thread;

        for (; m_idx_local < HALF_M; m_idx_local += (WARP_NUM * 64) / N_thread)
        {

          int32_t warp_idx = m_idx_local / WARP_M_HALF;
          int32_t offset = m_idx_local % WARP_M_HALF;
          int32_t m_idx_global = warp_idx * WARP_M + (phase * WARP_M_HALF) + offset;

          const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx_global, int(sorted_token_lens - 1))];

          if (token_index < size_m * top_k)
          {
            int32_t lds_offset = m_idx_local * BLOCK_SIZE_N + (m_idx_local * SWZ_ATOM + n_idx * 8) % BLOCK_SIZE_N;
            vec8_Element<Element> tmp = *reinterpret_cast<vec8_Element<Element> *>(&output_lds[lds_offset]);

            vec8_Element<Element> tmp1;
            tmp1[0] = tmp[0];
            tmp1[4] = tmp[1];
            tmp1[1] = tmp[2];
            tmp1[5] = tmp[3];
            tmp1[2] = tmp[4];
            tmp1[6] = tmp[5];
            tmp1[3] = tmp[6];
            tmp1[7] = tmp[7];

            *reinterpret_cast<vec8_Element<Element> *>(&g_output[token_index * size_n + n_idx * 8]) = tmp1;
          }
        }
      }

      if (phase == 0)
      {
        __syncthreads();
      }
    }

  } // end else
}

template <
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
    bool INTERLEAVE,
    bool DEQUANT_16BIT,
    int32_t PACK_SCALES,
    bool FP4_E2M1 = false>
__global__ void __launch_bounds__(512, 1) GEMM_NT_2STAGE(
    const Element *input,
    const uint32_t *__restrict__ qweight,
    Element *__restrict__ output,
    uint32_t *__restrict__ weight_zeros,
    Element *__restrict__ weight_scale,
    const uint8_t *__restrict__ weight_scale_u8,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t delta)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.x; // pid_n
  const int bidz = blockIdx.y; // pid_k

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  using Dtype = ScalarType<Element>;
  using scalar_t2 = typename ScalarType<Element>::scalar_t2;
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];                       // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + (uint64_t)bidy * BLOCK_SIZE_N * size_k;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置
  constexpr uint32_t scale_range = 32;
  const uint64_t weight_scale_offset = expert_id * size_n * (size_k / scale_range) + bidy * BLOCK_SIZE_N * (size_k / scale_range);
  const uint64_t weight_zeros_offset = (((uint64_t)size_n) * size_k * expert_id / 32 + (uint64_t)bidy * BLOCK_SIZE_N * size_k / 32) / 8;

  auto g_input = input;
  Element *g_output;
  g_output = output + output_offset;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                            // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  constexpr int warp_m_num = BLOCK_SIZE_M / WARP_M;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_m_id = warp_id / (warp_n_num * warp_k_num);
  int warp_nk_id = warp_id % (warp_n_num * warp_k_num);
  int warp_k_id = warp_nk_id % warp_k_num;
  int warp_n_id = warp_nk_id / warp_k_num;
  extern __shared__ uint16_t output_lds_tmp[]; // 声明lds信息
  Element *output_lds = reinterpret_cast<Element *>(output_lds_tmp);
  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec_opt_w4a16_A<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt_w4a16<uint32_t, WARP_K / 32> B_int_reg[WARP_N / mfma_n][STAGES];
  Element B_scale_reg[WARP_N / mfma_n][STAGES];
  uint32_t B_zeros_reg[WARP_N / mfma_n][STAGES];
  reg_bf16_fp16<scalar_t2> B_reg[WARP_N / mfma_n][STAGES][WARP_K / 32];

  auto g_qweight = qweight + qweight_offset / 8;
  auto g_weight_scale = weight_scale ? weight_scale + weight_scale_offset : nullptr; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
  auto g_weight_scale_u8 = weight_scale_u8 ? weight_scale_u8 + weight_scale_offset : nullptr;
  auto g_weight_zeros = weight_zeros + weight_zeros_offset;

  floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float weight_dot_a_scale[WARP_M / mfma_m] = {0.f};
  if constexpr (mul_topk_weight)
  {
#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++)
    {
      int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_index_safe = std::min(uint32_t(token_index), size_m * top_k - 1);
      weight_dot_a_scale[idx] = topk_weights[token_index_safe];
    }
  }

  gemm_nt_w4a16_stage_lambda_scale_ldx<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t2, INTERLEAVE, DEQUANT_16BIT, PACK_SCALES, FP4_E2M1>(g_input, g_qweight, g_weight_zeros, g_weight_scale, g_weight_scale_u8, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_zeros_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, warp_m_id);

  if (warp_k_id == 0 || warp_k_num == 1)
  {
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          int index = (warp_m_id * WARP_M + min_tile_m * mfma_m) * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + (warp_m_id * WARP_M + min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
          index += INTERLEAVE ? (reg_id + 4 * (lane_id / 16)) : (reg_id * 4 + lane_id / 16);
          output_lds[index] = Dtype::float2num(mul_topk_weight ? C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * weight_dot_a_scale[min_tile_m] : C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
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
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
      if (token_index < size_m * top_k)
      {
        *reinterpret_cast<vec_element_8<Element> *>(&g_output[token_index * size_n + n_idx * 8]) =
            *reinterpret_cast<vec_element_8<Element> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
      }
    }
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, bool FP4_E2M1, bool FP4_K_SPLIT>
void launch_moe_w4a16_first_stage_decode(const GemmParams_w4a16<T> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 32;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {                                                                                                     // marlin版本
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.y = 1;                                                                                      // k方向
    auto arch = W4A16_UTILS::ARCH::getArch();

    constexpr int fp4_ksplit_ab_lds =
        WARP_NUM * (BLOCK_SIZE_M * WARP_K + WARP_N * WARP_K) * sizeof(T);
    constexpr int fp4_ksplit_reduce_lds =
        (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N) *
        (BLOCK_SIZE_K / WARP_K - 1) * WARP_M * (WARP_N + 1) * sizeof(float);
    const int lds_size = (STAGES == 1) ? (FP4_K_SPLIT ? (fp4_ksplit_ab_lds + fp4_ksplit_reduce_lds) : (16 * 1024)) : std::max(BLOCK_SIZE_M * BLOCK_SIZE_N * sizeof(T) + 1024 /*for store pad*/, BLOCK_SIZE_M * BLOCK_SIZE_K * sizeof(T));
    const int shared_mem_size = lds_size + (FP4_E2M1 ? 64 : 0);

    if constexpr (STAGES == 1)
    {
      W4A16_ARCH_SWITCH(arch, [&]
                        {
          if constexpr (FP4_E2M1) {
            WFP4A16_STAGE1_SCALE_SWITCH(params.size_k,[&]{
              GEMM_NT_1STAGE_WFP4A16<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                GROUP_N, GROUP_K, STAGES, mul_topk_weight,INTERLEAVE,DEQUANT_16BIT,PACK_SCALES,FP4_K_SPLIT><<<gridDim, blockDim, shared_mem_size, stream>>>(
                params.ptr_A, 
                params.ptr_B0, 
                params.ptr_C, 
	                params.ptr_B_zeros, 
	                params.ptr_B_scale,
	                params.ptr_B_scale_u8,
                params.topk_weights,
                params.sorted_token_ids,
                params.expert_ids, 
                params.num_tokens_post_pad_ptr,
                params.size_m,
                params.size_n,
                params.size_k,
                params.sorted_token_lens,
                params.top_k,
                params.delta);
            })
          } else {
            W4A16_STAGE1_SCALE_SWITCH(params.size_k,[&]{
              GEMM_NT_1STAGE<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                GROUP_N, GROUP_K, STAGES, mul_topk_weight,INTERLEAVE,DEQUANT_16BIT,PACK_SCALES><<<gridDim, blockDim, shared_mem_size, stream>>>(
                params.ptr_A, 
                params.ptr_B0, 
                params.ptr_C, 
	                params.ptr_B_zeros, 
	                params.ptr_B_scale,
	                params.ptr_B_scale_u8,
                params.topk_weights,
                params.sorted_token_ids,
                params.expert_ids, 
                params.num_tokens_post_pad_ptr,
                params.size_m,
                params.size_n,
                params.size_k,
                params.sorted_token_lens,
                params.top_k,
                params.delta);
            })
          } })
    }
    else
    {

      W4A16_ARCH_SWITCH(arch, [&]
                        { W4A16_SCALE_SWITCH(params.size_k, [&]
                                             { GEMM_NT_2STAGE<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, GROUP_N,
                                                              GROUP_K, STAGES, mul_topk_weight, INTERLEAVE, DEQUANT_16BIT, PACK_SCALES, FP4_E2M1><<<gridDim, blockDim, shared_mem_size, stream>>>(
                                                   params.ptr_A,
                                                   params.ptr_B0,
                                                   params.ptr_C,
                                                   params.ptr_B_zeros,
                                                   params.ptr_B_scale,
                                                   params.ptr_B_scale_u8,
                                                   params.topk_weights,
                                                   params.sorted_token_ids,
                                                   params.expert_ids,
                                                   params.num_tokens_post_pad_ptr,
                                                   params.size_m,
                                                   params.size_n,
                                                   params.size_k,
                                                   params.sorted_token_lens,
                                                   params.top_k,
                                                   params.delta); }) })
    }
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, bool FP4_E2M1>
void launch_moe_w4a16_second_stage_decode(const GemmParams_w4a16<T> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 32;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;
  auto arch = W4A16_UTILS::ARCH::getArch();

  const int lds_size = (STAGES == 1) ? (16 * 1024) : std::max(BLOCK_SIZE_M * BLOCK_SIZE_N * sizeof(T) + 1024 /*for store pad*/, BLOCK_SIZE_M * BLOCK_SIZE_K * sizeof(T));
  const int shared_mem_size = lds_size + (FP4_E2M1 ? 64 : 0);

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  {                                                                                                     // marlin版本
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.y = 1;                                                                                      // k方向
    if constexpr (STAGES == 1)
    {

      W4A16_ARCH_SWITCH(arch, [&]
                        {
          if constexpr (FP4_E2M1) {
            WFP4A16_STAGE1_SCALE_SWITCH(params.size_k,[&]{
              GEMM_NT_1STAGE_WFP4A16<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                GROUP_N, GROUP_K, STAGES, mul_topk_weight,INTERLEAVE,DEQUANT_16BIT,PACK_SCALES><<<gridDim, blockDim, shared_mem_size, stream>>>(
                params.ptr_A, 
                params.ptr_B0, 
                params.ptr_C, 
	                params.ptr_B_zeros, 
	                params.ptr_B_scale,
	                params.ptr_B_scale_u8,
                params.topk_weights,
                params.sorted_token_ids,
                params.expert_ids, 
                params.num_tokens_post_pad_ptr,
                params.size_m,
                params.size_n,
                params.size_k,
                params.sorted_token_lens,
                params.top_k,
                params.delta);
            })
          } else {
            W4A16_STAGE1_SCALE_SWITCH(params.size_k,[&]{
              GEMM_NT_1STAGE<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                GROUP_N, GROUP_K, STAGES, mul_topk_weight,INTERLEAVE,DEQUANT_16BIT,PACK_SCALES><<<gridDim, blockDim, shared_mem_size, stream>>>(
                params.ptr_A, 
                params.ptr_B0, 
                params.ptr_C, 
	                params.ptr_B_zeros, 
	                params.ptr_B_scale,
	                params.ptr_B_scale_u8,
                params.topk_weights,
                params.sorted_token_ids,
                params.expert_ids, 
                params.num_tokens_post_pad_ptr,
                params.size_m,
                params.size_n,
                params.size_k,
                params.sorted_token_lens,
                params.top_k,
                params.delta);
            })
          } })
    }
    else
    {

      W4A16_ARCH_SWITCH(arch, [&]
                        { W4A16_SCALE_SWITCH(params.size_k, [&]
                                             { GEMM_NT_2STAGE<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, GROUP_N,
                                                              GROUP_K, STAGES, mul_topk_weight, INTERLEAVE, DEQUANT_16BIT, PACK_SCALES, FP4_E2M1><<<gridDim, blockDim, shared_mem_size, stream>>>(
                                                   params.ptr_A,
                                                   params.ptr_B0,
                                                   params.ptr_C,
                                                   params.ptr_B_zeros,
                                                   params.ptr_B_scale,
                                                   params.ptr_B_scale_u8,
                                                   params.topk_weights,
                                                   params.sorted_token_ids,
                                                   params.expert_ids,
                                                   params.num_tokens_post_pad_ptr,
                                                   params.size_m,
                                                   params.size_n,
                                                   params.size_k,
                                                   params.sorted_token_lens,
                                                   params.top_k,
                                                   params.delta); }) })
    }
  }
}
