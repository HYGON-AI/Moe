// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "moe_wna16_utils.h"

#include "intrinsic.h"

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

template <typename scalar_t,
          int bit,
          uint16_t top_k,
          uint16_t BLOCK_SIZE_M,
          uint16_t BLOCK_SIZE_N,
          uint16_t BLOCK_SIZE_K,
          bool has_zp,
          bool mul_topk_weight,
          uint16_t group_size,
          int GROUPS>
__global__ void moe_wna16_gemm_kernel_base(
    const scalar_t *__restrict__ input,
    float *__restrict__ output,
    const uint32_t *__restrict__ qweight,
    const scalar_t *__restrict__ scales,
    const uint32_t *__restrict__ qzeros,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k)
{
  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;

  if (blockIdx.x * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 分块超过了tokens的数量，会直接返回

  const int32_t offset_n = blockIdx.y * BLOCK_SIZE_N + threadIdx.x; // 每个线程对应N的一个维度
  const int32_t offset_k = blockIdx.z * BLOCK_SIZE_K;

  const int32_t expert_id = expert_ids[blockIdx.x];

  int32_t num_valid_tokens = 0;
  extern __shared__ uint16_t block_input_tmp[];
  scalar_t *block_input = reinterpret_cast<scalar_t *>(block_input_tmp);
  scalar_t2 *block_input_half2 = reinterpret_cast<scalar_t2 *>(block_input);

#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; m++)
  {
    const int32_t offset_m = blockIdx.x * BLOCK_SIZE_M + m;
    const int32_t token_index = sorted_token_ids[offset_m];
    if (token_index / top_k < size_m)
    {
      num_valid_tokens = m + 1;
      if (blockIdx.z == 0 && offset_n < size_n)
        output[token_index * size_n + offset_n] = 0; // 初始化output为0

      if (expert_id != -1)
      {
        int k_per_thread = DIVIDE(BLOCK_SIZE_K, BLOCK_SIZE_N); // 每个线程需要读K方向的数据量
        for (int i = 0; i < k_per_thread; i++)
        {
          int k = BLOCK_SIZE_N * i + threadIdx.x;
          if (k >= BLOCK_SIZE_K)
            break;
          if (offset_k + k >= size_k)
            break;

          int origin_k;
          if constexpr (bit == 4)
          {
            int8_t order = (threadIdx.x % 2) * 4 + ((threadIdx.x % 8) / 2);
            origin_k = BLOCK_SIZE_N * i + threadIdx.x / 8 * 8 + order;
          }
          else
          {
            int8_t order = (threadIdx.x % 2) * 2 + ((threadIdx.x % 4) / 2);
            origin_k = BLOCK_SIZE_N * i + threadIdx.x / 4 * 4 + order;
          }

          origin_k += token_index / top_k * size_k + blockIdx.z * BLOCK_SIZE_K;
          block_input[m * BLOCK_SIZE_K + k] = input[origin_k];
        }
      }
    }
  }

  if (expert_id == -1)
    return;
  __syncthreads();
  if (threadIdx.x >= BLOCK_SIZE_N || offset_n >= size_n)
    return;

  float res[64]; // assume BLOCK_SIZE_M <= 64
  scalar_t2 res2;
  scalar_t2 scale_f2;
  scalar_t2 qzero_f2;

  constexpr int8_t pack_factor = 32 / bit; // 4字节能存多少元素
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;
  const uint32_t *expert_qweight = qweight + expert_offset / pack_factor;
  const scalar_t *expert_scales = scales + expert_offset / group_size;
  const uint32_t *expert_qzeros =
      qzeros + expert_offset / group_size / pack_factor;

  union_vec<uint32_t, 4> expert_qweight_tmp;

  union_vec<scalar_t, GROUPS> expert_scales_groups;
  int scales_offset_tmp =
      (offset_n * size_k + offset_k) / group_size / GROUPS;
  auto g_expert_scales = tcp_cache_swizzle_func<1, scalar_t>(expert_scales);
  if constexpr (GROUPS == 1)
  {
    expert_scales_groups.scalar_array[0] = expert_scales[scales_offset_tmp];
  }
  else if constexpr (GROUPS == 2)
  {
    int soffset = 0;
    int voffset = scales_offset_tmp;
    inline_buffer_load_dword<float, 2>(expert_scales_groups.float_array[0], voffset, g_expert_scales, soffset);
  }
  else if constexpr (GROUPS == 4)
  {
    expert_scales_groups.float2_array[0] = reinterpret_cast<const vec<float, 2> *>(expert_scales)[scales_offset_tmp];
  }
  else if constexpr (GROUPS == 8)
  {
    expert_scales_groups.float4_array[0] = reinterpret_cast<const vec<float, 4> *>(expert_scales)[scales_offset_tmp];
  }

  uint8_t expert_qzeros_groups[GROUPS];
  if (!has_zp)
  {
    if constexpr (bit == 4)
    {
      qzero_f2 = Dtype::num2num2(Dtype::int2num(8));
    }
    else
    {
      qzero_f2 = Dtype::num2num2(Dtype::int2num(128));
    }
  }
  else
  {
    int qzeros_offset_tmp =
        (offset_n / (8 / bit)) * (size_k / group_size / GROUPS) +
        offset_k / group_size / GROUPS;
    if constexpr (GROUPS == 1)
    {
      uint8_t *expert_qzeros_groups_tmp =
          reinterpret_cast<uint8_t *>(expert_qzeros_groups);
      *expert_qzeros_groups_tmp =
          reinterpret_cast<const uint8_t *>(expert_qzeros)[qzeros_offset_tmp];
    }
    else if constexpr (GROUPS == 2)
    {
      uint16_t *expert_qzeros_groups_tmp =
          reinterpret_cast<uint16_t *>(expert_qzeros_groups);
      *expert_qzeros_groups_tmp =
          reinterpret_cast<const uint16_t *>(expert_qzeros)[qzeros_offset_tmp];
    }
    else if constexpr (GROUPS == 4)
    {
      uint32_t *expert_qzeros_groups_tmp =
          reinterpret_cast<uint32_t *>(expert_qzeros_groups);
      *expert_qzeros_groups_tmp =
          reinterpret_cast<const uint32_t *>(expert_qzeros)[qzeros_offset_tmp];
    }
    else if constexpr (GROUPS == 8)
    {
      uint64_t *expert_qzeros_groups_tmp =
          reinterpret_cast<uint64_t *>(expert_qzeros_groups);
      *expert_qzeros_groups_tmp =
          reinterpret_cast<const uint64_t *>(expert_qzeros)[qzeros_offset_tmp];
    }
  }

#pragma unroll
  for (int tmp_k1 = 0; tmp_k1 < BLOCK_SIZE_K / group_size; tmp_k1++)
  {
    {
      scalar_t scale_f =
          expert_scales_groups.scalar_array[tmp_k1];
      scale_f2 = Dtype::num2num2(scale_f);

      if (has_zp)
      {
        uint8_t qzero =
            expert_qzeros_groups[tmp_k1];
        if constexpr (bit == 4)
        {
          qzero = (qzero >> ((threadIdx.x % 2) * 4)) & 0xF;
        }
        qzero_f2 = Dtype::num2num2(Dtype::int2num(qzero));
      }
    }
#pragma unroll
    for (int tmp_k2 = 0; tmp_k2 < (group_size / pack_factor); tmp_k2++)
    {
      int tmp_k = tmp_k1 * (group_size / pack_factor) + tmp_k2;
      int k = offset_k + tmp_k * pack_factor;
      if (k < size_k)
      {
        const int32_t weight_offset = offset_n * size_k + k;
        if (tmp_k % 4 == 0)
        {
          expert_qweight_tmp.float4_array[0] = reinterpret_cast<const vec<float, 4> *>(
              expert_qweight)[weight_offset / pack_factor / 4];
        }
        scalar_t2 weight_half2[16 / bit];
        dequant<scalar_t2, bit>(expert_qweight_tmp.scalar_array[tmp_k % 4], weight_half2);
        for (int m = 0; m < BLOCK_SIZE_M /*num_valid_tokens*/; m++)
        { // 有效的token数量
          if (m < num_valid_tokens)
          {
            res2.x = 0.0f;
            res2.y = 0.0f;

#pragma unroll
            for (int i = 0; i < 16 / bit; i++)
            { // 计算一个float内quant类型数量的fma
              int32_t offset_input = m * BLOCK_SIZE_K / 2 + tmp_k * (16 / bit) + i;
              res2 = __hfma2(__hmul2(__hsub2(weight_half2[i], qzero_f2), scale_f2),
                             block_input_half2[offset_input], res2);
            }

            if (tmp_k == 0)
            {
              res[m] = Dtype::num2float(res2.x) + Dtype::num2float(res2.y);
            }
            else
            {
              res[m] += Dtype::num2float(res2.x) + Dtype::num2float(res2.y);
            }
          }
        }
      }
    }
  }
  for (int m = 0; m < BLOCK_SIZE_M /*num_valid_tokens*/; ++m)
  { // 有效的token数量
    if (m < num_valid_tokens)
    {
      const int32_t token_index =
          sorted_token_ids[blockIdx.x * BLOCK_SIZE_M + m];
      if (mul_topk_weight)
      {
        res[m] *= topk_weights[token_index]; // 乘以对应expert的权重
      }
      atomicAdd(&output[token_index * size_n + offset_n],
                res[m]);
    }
  }
}

template <typename scalar_t,
          int bit,
          int top_k,
          int BLOCK_SIZE_M,
          int BLOCK_SIZE_N,
          int BLOCK_SIZE_K,
          bool has_zp,
          bool mul_topk_weight,
          int group_size>
void run_moe_wna16_gemm_base(const scalar_t *input,
                             float *output_fp32,
                             const uint32_t *b_qweight,
                             const scalar_t *b_scales,
                             const uint32_t *b_qzeros,
                             const float *topk_weights,
                             const int32_t *sorted_token_ids,
                             const int32_t *expert_ids,
                             const int32_t *num_tokens_post_pad,
                             int num_token_blocks,
                             int size_m,
                             int size_n,
                             int size_k)
{
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_SIZE_N;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = num_token_blocks;
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N);
  gridDim.z = DIVIDE(size_k, BLOCK_SIZE_K);

  auto kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 1>;
  if (bit == 4)
  {
    if (BLOCK_SIZE_K / group_size == 2)
    {
      kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 2>;
    }
    else if (BLOCK_SIZE_K / group_size == 4)
    {
      kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 4>;
    }
    else if (BLOCK_SIZE_K / group_size == 8)
    {
      kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 8>;
    }
  }
  else
  {
    if (BLOCK_SIZE_K / group_size == 1)
    {
      kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 1>;
    }
    else if (BLOCK_SIZE_K / group_size == 2)
    {
      kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 2>;
    }
    else if (BLOCK_SIZE_K / group_size == 4)
    {
      kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 4>;
    }
    else if (BLOCK_SIZE_K / group_size == 8)
    {
      kernel = moe_wna16_gemm_kernel_base<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, 8>;
    }
  }

  const int shared_mem_size = 8 * 1024;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  kernel<<<gridDim, blockDim, shared_mem_size, stream>>>(
      input,
      output_fp32,
      b_qweight,
      b_scales,
      b_qzeros,
      topk_weights,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      size_m, size_n, size_k);
}