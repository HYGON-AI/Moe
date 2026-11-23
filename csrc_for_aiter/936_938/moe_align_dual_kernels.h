// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "dispatch_utils.h"

namespace moe_c
{
  // Last-block part of the multi-block sgl align: padded exclusive scan of counts (block_size
  // padding per expert) into sh[0..num_experts] and seg_start, num_tokens_post_pad and expert_ids.
  // 256 threads, num_experts <= 1024.
  __device__ __forceinline__ void sgl_moe_align_dual_padded_scan(const int32_t *__restrict__ counts, int32_t *sh,
                                                            int32_t *wsum, int32_t *__restrict__ seg_start,
                                                            int32_t *__restrict__ expert_ids,
                                                            int32_t *__restrict__ total_tokens_post_pad,
                                                            int32_t num_experts, int32_t block_size,
                                                            int32_t max_num_tokens_padded)
  {
    constexpr int NT = 256;
    constexpr int PER = 4; // experts per thread: num_experts <= NT * PER
    const int tid = threadIdx.x;
    int v[PER], local = 0;
#pragma unroll
    for (int j = 0; j < PER; j++)
    {
      const int e = tid * PER + j;
      v[j] = e < num_experts ? at::ceil_div(__atomic_load_n(&counts[e], __ATOMIC_RELAXED), block_size) * block_size : 0;
      local += v[j];
    }
    const int lane = tid & 63, wid = tid >> 6;
    int x = local;
#pragma unroll
    for (int o = 1; o < 64; o <<= 1)
    {
      const int y = __shfl_up(x, o, 64);
      if (lane >= o)
        x += y;
    }
    if (lane == 63)
      wsum[wid] = x;
    __syncthreads();
    int run = x - local;
    for (int w = 0; w < wid; w++)
      run += wsum[w];
#pragma unroll
    for (int j = 0; j < PER; j++)
    {
      const int e = tid * PER + j;
      if (e < num_experts)
        sh[e] = run;
      run += v[j];
    }
    if (tid == NT - 1)
      sh[num_experts] = run;
    __syncthreads();
    for (int i = tid; i <= num_experts; i += NT)
      seg_start[i] = sh[i];
    const int total = sh[num_experts];
    if (tid == 0)
      *total_tokens_post_pad = total;
    // expert_ids: each thread writes the blocks of its experts, then the tail blocks are zeroed
    const int num_blocks = total / block_size;
    const int max_m_blocks = at::ceil_div(max_num_tokens_padded, block_size);
    for (int e = tid; e < num_experts; e += NT)
      for (int i = sh[e] / block_size; i < sh[e + 1] / block_size; i++)
        expert_ids[i] = e;
    for (int i = num_blocks + tid; i < max_m_blocks; i += NT)
      expert_ids[i] = 0;
    __syncthreads(); // sh / wsum are reused by a second scan
  }

  // Accumulate per-expert counts and fill both output buffers with padding. The
  // last block computes the padded expert offsets for each block size.
  // buf = counts[E] | rank[E] | seg[E + 1] | seg2[E + 1] | done[1]
  template <typename scalar_t>
  __global__ void __launch_bounds__(256) sgl_moe_align_dual_count_scan_kernel(
      const scalar_t *__restrict__ topk_ids, int32_t *__restrict__ buf, int32_t *__restrict__ sorted_token_ids,
      int32_t *__restrict__ expert_ids, int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
      int32_t block_size, size_t numel, int32_t max_num_tokens_padded, int32_t *__restrict__ sorted_token_ids2,
      int32_t *__restrict__ expert_ids2, int32_t *__restrict__ total_tokens_post_pad2, int32_t block_size2,
      int32_t max_num_tokens_padded2)
  {
    constexpr int NT = 256;
    extern __shared__ int32_t sh[]; // histogram, then padded prefix [num_experts + 1]
    __shared__ int32_t wsum[NT / 64];
    __shared__ bool last;
    int32_t *counts = buf;
    int32_t *seg = buf + 2 * num_experts;
    int32_t *seg2 = buf + 3 * num_experts + 1;
    int32_t *done = buf + 4 * num_experts + 2;
    const int tid = threadIdx.x;

    for (int i = tid; i < num_experts; i += NT)
      sh[i] = 0;
    __syncthreads();
    const size_t stride = (size_t)gridDim.x * NT;
    for (size_t i = (size_t)blockIdx.x * NT + tid; i < numel; i += stride)
    {
      const int32_t e = static_cast<int32_t>(topk_ids[i]);
      if (e >= 0 && e < num_experts)
        atomicAdd(&sh[e], 1);
    }
    for (size_t i = (size_t)blockIdx.x * NT + tid; i < (size_t)max_num_tokens_padded; i += stride)
      sorted_token_ids[i] = static_cast<int32_t>(numel);
    for (size_t i = (size_t)blockIdx.x * NT + tid; i < (size_t)max_num_tokens_padded2; i += stride)
      sorted_token_ids2[i] = static_cast<int32_t>(numel);
    __syncthreads();
    for (int i = tid; i < num_experts; i += NT)
      if (sh[i])
        atomicAdd(&counts[i], sh[i]);
    __threadfence();
    __syncthreads();
    if (tid == 0)
      last = atomicAdd(done, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (!last)
      return;
    __threadfence();
    sgl_moe_align_dual_padded_scan(counts, sh, wsum, seg, expert_ids, total_tokens_post_pad, num_experts, block_size,
                              max_num_tokens_padded);
    sgl_moe_align_dual_padded_scan(counts, sh, wsum, seg2, expert_ids2, total_tokens_post_pad2, num_experts,
                              block_size2, max_num_tokens_padded2);
  }

  // Each CTA takes a consecutive slice of slots and reserves a rank range for
  // each expert. The same rank is used in both routings.
  template <typename scalar_t, int PER>
  __global__ void __launch_bounds__(256) sgl_moe_align_dual_scatter_kernel(const scalar_t *__restrict__ topk_ids,
                                                                      int32_t *__restrict__ sorted_token_ids,
                                                                      int32_t *__restrict__ buf, int32_t num_experts,
                                                                      size_t numel,
                                                                      int32_t *__restrict__ sorted_token_ids2)
  {
    constexpr int NT = 256;
    extern __shared__ int32_t sh[]; // per-expert count, then the block's base in the segment
    int32_t *rank = buf + num_experts;
    const int32_t *seg = buf + 2 * num_experts;
    const int32_t *seg2 = buf + 3 * num_experts + 1;
    for (int i = threadIdx.x; i < num_experts; i += NT)
      sh[i] = 0;
    __syncthreads();
    const size_t base = (size_t)blockIdx.x * NT * PER;
    int32_t ex[PER], rk[PER];
#pragma unroll
    for (int j = 0; j < PER; j++)
    {
      const size_t i = base + (size_t)j * NT + threadIdx.x;
      ex[j] = i < numel ? static_cast<int32_t>(topk_ids[i]) : -1;
      rk[j] = (ex[j] >= 0 && ex[j] < num_experts) ? atomicAdd(&sh[ex[j]], 1) : 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_experts; i += NT)
      sh[i] = sh[i] ? atomicAdd(&rank[i], sh[i]) : 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < PER; j++)
    {
      const size_t i = base + (size_t)j * NT + threadIdx.x;
      if (ex[j] >= 0 && ex[j] < num_experts)
      {
        const int32_t r = sh[ex[j]] + rk[j];
        sorted_token_ids[seg[ex[j]] + r] = static_cast<int32_t>(i);
        sorted_token_ids2[seg2[ex[j]] + r] = static_cast<int32_t>(i);
      }
    }
  }
} // namespace moe_c

// scatter launch for PER slots per thread (1, 2, 4, 8 or 16)
template <typename scalar_t>
static void moe_c_sgl_align_dual_scatter(int per, int blocks, int threads, size_t lds, cudaStream_t stream,
                                    const scalar_t *ids, int32_t *sorted, int32_t *buf, int32_t ne, size_t numel,
                                    int32_t *sorted2)
{
  if (per == 16)
    moe_c::sgl_moe_align_dual_scatter_kernel<scalar_t, 16><<<blocks, threads, lds, stream>>>(ids, sorted, buf, ne, numel, sorted2);
  else if (per == 8)
    moe_c::sgl_moe_align_dual_scatter_kernel<scalar_t, 8><<<blocks, threads, lds, stream>>>(ids, sorted, buf, ne, numel, sorted2);
  else if (per == 4)
    moe_c::sgl_moe_align_dual_scatter_kernel<scalar_t, 4><<<blocks, threads, lds, stream>>>(ids, sorted, buf, ne, numel, sorted2);
  else if (per == 2)
    moe_c::sgl_moe_align_dual_scatter_kernel<scalar_t, 2><<<blocks, threads, lds, stream>>>(ids, sorted, buf, ne, numel, sorted2);
  else
    moe_c::sgl_moe_align_dual_scatter_kernel<scalar_t, 1><<<blocks, threads, lds, stream>>>(ids, sorted, buf, ne, numel, sorted2);
}

// Build two paddings of the same expert routing for GEMM1 and GEMM2.
void moe_c_sgl_moe_align_block_size_dual(torch::Tensor topk_ids, int64_t num_experts,
                                         int64_t block_size, torch::Tensor sorted_token_ids,
                                         torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                                         int64_t block_size2, torch::Tensor sorted_token_ids2,
                                         torch::Tensor experts_ids2, torch::Tensor num_tokens_post_pad2)
{
  TORCH_CHECK(num_experts == 256 || num_experts == 384 || num_experts == 896,
              "sgl_moe_align_block_size_dual kernel only supports 256, 384 or 896 experts.");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto options = torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  torch::Tensor buf = torch::zeros({4 * num_experts + 3}, options);
  const int64_t numel = topk_ids.numel();
  const int64_t max_pad = sorted_token_ids.numel();
  const int64_t max_pad2 = sorted_token_ids2.numel();
  constexpr int threads = 256;
  const int64_t work = std::max({numel, max_pad, max_pad2});
  const int count_blocks = std::clamp<int64_t>(at::ceil_div<int64_t>(work, threads * 4), 1, 160);
  const int32_t ne = static_cast<int32_t>(num_experts);

  MOE_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "sgl_moe_align_block_size_dual_kernel", [&] {
    auto ids = topk_ids.data_ptr<scalar_t>();
    auto sorted = sorted_token_ids.data_ptr<int32_t>();
    auto sorted2 = sorted_token_ids2.data_ptr<int32_t>();
    auto buffer = buf.data_ptr<int32_t>();
    moe_c::sgl_moe_align_dual_count_scan_kernel<scalar_t><<<count_blocks, threads, (ne + 1) * sizeof(int32_t), stream>>>(
        ids, buffer, sorted, experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
        ne, static_cast<int32_t>(block_size), numel, static_cast<int32_t>(max_pad),
        sorted2, experts_ids2.data_ptr<int32_t>(), num_tokens_post_pad2.data_ptr<int32_t>(),
        static_cast<int32_t>(block_size2), static_cast<int32_t>(max_pad2));
    if (numel == 0)
      return;

    // Use the largest slots-per-thread value that keeps at least 160 CTAs when possible.
    constexpr int min_slots = threads * 160;
    const int per = numel >= min_slots * 16 ? 16 : numel >= min_slots * 8 ? 8 :
                    numel >= min_slots * 4 ? 4 : numel >= min_slots * 2 ? 2 : 1;
    // Each scatter CTA owns a fixed slice of slots and must be launched.
    const int blocks = at::ceil_div<int64_t>(numel, threads * per);
    moe_c_sgl_align_dual_scatter<scalar_t>(per, blocks, threads, ne * sizeof(int32_t),
                                          stream, ids, sorted, buffer, ne, numel, sorted2);
  });
}
