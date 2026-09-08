// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <hipcub/hipcub.hpp>

#include "compat.h"
#include "dispatch_utils.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

namespace moe_c
{

  namespace
  {
    __device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                             int32_t col)
    {
      // don't worry about overflow because num_experts is relatively small
      return row * total_col + col;
    }

    inline uint32_t next_pow2(uint32_t x) noexcept
    {
      --x;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      x |= x >> 16;
      return x + 1;
    }
  } // namespace

  template <typename scalar_t, typename token_cnts_t>
  __global__ void moe_align_block_size_kernel(scalar_t *__restrict__ topk_ids,
                                              int32_t *sorted_token_ids,
                                              int32_t *expert_ids,
                                              int32_t *total_tokens_post_pad,
                                              int32_t num_experts,
                                              int32_t block_size, size_t numel)
  {
    const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
    const size_t start_idx = threadIdx.x * tokens_per_thread;

    extern __shared__ int32_t shared_mem[];
    int32_t *cumsum = shared_mem; // 1d tensor with shape (num_experts + 1)
    token_cnts_t *tokens_cnts =
        (token_cnts_t *)(shared_mem + num_experts +
                         1); // 2d tensor with shape (blockDim.x + 1, num_experts)

    for (int i = 0; i < num_experts; ++i)
    {
      tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
    }

    /**
     * In the first step we compute token_cnts[thread_index + 1][expert_index],
     * which counts how many tokens in the token shard of thread_index are
     * assigned to expert expert_index.
     */
    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
    {
      int res = topk_ids[i];
      if (res == -1)
      {
      }
      else
      {
        ++tokens_cnts[index(num_experts, threadIdx.x + 1, res)];
      }
    }

    __syncthreads();

    // For each expert we accumulate the token counts from the different threads.
    if (threadIdx.x < num_experts)
    {
      tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
      for (int i = 1; i <= blockDim.x; ++i)
      {
        tokens_cnts[index(num_experts, i, threadIdx.x)] +=
            tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
      }
    }

    __syncthreads();

    // We accumulate the token counts of all experts in thread 0.
    if (threadIdx.x == 0)
    {
      cumsum[0] = 0;
      for (int i = 1; i <= num_experts; ++i)
      {
        cumsum[i] = cumsum[i - 1] +
                    CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                            block_size) *
                        block_size;
      }
      *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }

    __syncthreads();

    const int32_t total_padded = cumsum[num_experts];
    const int32_t valid_m_blocks = CEILDIV(total_padded, block_size);
    const int32_t max_m_blocks = CEILDIV(
        static_cast<int32_t>(numel) + num_experts * (block_size - 1),
        block_size);
    for (int32_t i = valid_m_blocks + threadIdx.x; i < max_m_blocks;
         i += blockDim.x)
    {
      expert_ids[i] = 0;
    }

    /**
     * For each expert, each thread processes the tokens of the corresponding
     * blocks and stores the corresponding expert_id for each block.
     */
    if (threadIdx.x < num_experts)
    {
      const int32_t expert_begin = cumsum[threadIdx.x];
      const int32_t expert_end = cumsum[threadIdx.x + 1];
      for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
           i += block_size)
      {
        expert_ids[i / block_size] = threadIdx.x;
      }

      const int32_t raw_count = static_cast<int32_t>(
          tokens_cnts[index(num_experts, blockDim.x, threadIdx.x)]);
      for (int32_t i = expert_begin + raw_count; i < expert_end; ++i)
      {
        sorted_token_ids[i] = static_cast<int32_t>(numel);
      }
    }

    /**
     * Each thread processes a token shard, calculating the index of each token
     * after sorting by expert number. Given the example topk_ids =
     * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
     * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
     * padding value.
     */
    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
    {
      int res2 = topk_ids[i];
      if (res2 == -1)
      {
      }
      else
      {
        int32_t expert_id = res2;
        /** The cumsum[expert_id] stores the starting index of the tokens that the
         * expert with expert_id needs to process, and
         * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
         * processed by the expert with expert_id within the current thread's token
         * shard.
         */
        int32_t rank_post_pad =
            tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
            cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
        ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
      }
    }
  }

  // Global-memory fallback migrated to the current SGLang two-kernel
  // algorithm. The token order within the same expert is atomic-order based.
  template <typename scalar_t>
  __global__ void moe_align_block_size_global_mem_kernel(
      scalar_t *__restrict__ topk_ids, int32_t *sorted_token_ids,
      int32_t *expert_ids, int32_t *total_tokens_post_pad, int32_t num_experts,
      int32_t block_size, size_t numel, int32_t *cumsum,
      bool pad_sorted_token_ids, int32_t scan_size,
      int32_t max_num_tokens_padded)
  {
    constexpr int kVecSize = 4;
    using Vec = int4;

    if (blockIdx.x == 1)
    {
      if (pad_sorted_token_ids)
      {
        Vec fill_vec;
        fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w =
            static_cast<int32_t>(numel);
        int32_t total_vecs = (max_num_tokens_padded + kVecSize - 1) / kVecSize;
        Vec *out_ptr = reinterpret_cast<Vec *>(sorted_token_ids);
        for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x)
        {
          out_ptr[i] = fill_vec;
        }
      }
      return;
    }

    extern __shared__ int32_t smem[];
    int32_t *shared_counts = smem;
    int32_t *prefix = shared_counts + num_experts;
    int32_t *scan_buf = prefix + num_experts + 1;
    __shared__ int32_t s_total_tokens_post_pad;

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    for (int32_t i = tid; i < num_experts; i += stride)
    {
      shared_counts[i] = 0;
    }
    __syncthreads();

    for (size_t i = tid; i < numel; i += stride)
    {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
      if (expert_id >= 0 && expert_id < num_experts)
      {
        atomicAdd(&shared_counts[expert_id], 1);
      }
    }
    __syncthreads();

    if (tid < num_experts)
    {
      int32_t count = shared_counts[tid];
      scan_buf[tid] = CEILDIV(count, block_size) * block_size;
    }
    if (tid >= num_experts && tid < scan_size)
    {
      scan_buf[tid] = 0;
    }
    __syncthreads();

    int offset = 1;
#pragma unroll
    for (int d = scan_size >> 1; d > 0; d >>= 1)
    {
      if (tid < d)
      {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        scan_buf[bi] += scan_buf[ai];
      }
      offset <<= 1;
      __syncthreads();
    }

    if (tid == 0)
    {
      prefix[num_experts] = scan_buf[scan_size - 1];
      scan_buf[scan_size - 1] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int d = 1; d < scan_size; d <<= 1)
    {
      offset >>= 1;
      if (tid < d)
      {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        if (bi < scan_size)
        {
          int temp = scan_buf[ai];
          scan_buf[ai] = scan_buf[bi];
          scan_buf[bi] += temp;
        }
      }
      __syncthreads();
    }

    for (int32_t i = tid; i < num_experts; i += stride)
    {
      prefix[i] = scan_buf[i];
    }
    if (tid == 0)
    {
      s_total_tokens_post_pad = prefix[num_experts];
      *total_tokens_post_pad = s_total_tokens_post_pad;
    }
    __syncthreads();

    for (int32_t i = tid; i <= num_experts; i += stride)
    {
      cumsum[i] = prefix[i];
    }

    const int32_t num_blocks = s_total_tokens_post_pad / block_size;
    const int32_t max_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
    for (int32_t i = num_blocks + tid; i < max_m_blocks; i += stride)
    {
      expert_ids[i] = 0;
    }

    for (int32_t i = tid; i < num_blocks; i += stride)
    {
      int32_t block_start = i * block_size;
      int32_t left = 0;
      int32_t right = num_experts;
      while (left < right)
      {
        int32_t mid = (left + right) >> 1;
        if (prefix[mid] <= block_start)
        {
          left = mid + 1;
        }
        else
        {
          right = mid;
        }
      }
      expert_ids[i] = left - 1;
    }
  }

  template <typename scalar_t, int EXPERTS_PER_THREAD>
  __global__ void moe_align_block_size_global_mem_kernel_v2(
      scalar_t *__restrict__ topk_ids, int32_t *sorted_token_ids,
      int32_t *expert_ids, int32_t *total_tokens_post_pad, int32_t num_experts,
      int32_t padded_num_experts, int32_t block_size, size_t numel,
      int32_t *cumsum, bool pad_sorted_token_ids,
      int32_t max_num_tokens_padded)
  {
    constexpr int kVecSize = 4;
    using Vec = int4;

    if (blockIdx.x == 1)
    {
      if (pad_sorted_token_ids)
      {
        Vec fill_vec;
        fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w =
            static_cast<int32_t>(numel);
        int32_t total_vecs = (max_num_tokens_padded + kVecSize - 1) / kVecSize;
        Vec *out_ptr = reinterpret_cast<Vec *>(sorted_token_ids);
        for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x)
        {
          out_ptr[i] = fill_vec;
        }
      }
      return;
    }

    extern __shared__ int32_t smem[];
    int32_t *shared_counts = smem;
    int32_t *warp_sums = smem + padded_num_experts;

    const size_t tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);

    for (int32_t i = tid; i < padded_num_experts; i += blockDim.x)
    {
      shared_counts[i] = 0;
    }
    __syncthreads();

    for (size_t i = tid; i < numel; i += blockDim.x)
    {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
      if (expert_id >= 0 && expert_id < num_experts)
      {
        atomicAdd(&shared_counts[expert_id], 1);
      }
    }
    __syncthreads();

    const int my_start = tid * EXPERTS_PER_THREAD;
    int32_t local_padded[EXPERTS_PER_THREAD];
    int32_t thread_sum = 0;
#pragma unroll
    for (int i = 0; i < EXPERTS_PER_THREAD; ++i)
    {
      int eid = my_start + i;
      if (eid < num_experts)
      {
        local_padded[i] = CEILDIV(shared_counts[eid], block_size) * block_size;
      }
      else
      {
        local_padded[i] = 0;
      }
      thread_sum += local_padded[i];
    }

    int32_t warp_prefix = thread_sum;
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
    {
      int32_t n = __shfl_up(warp_prefix, offset, WARP_SIZE);
      if (lane_id >= offset)
      {
        warp_prefix += n;
      }
    }
    int32_t warp_total = warp_prefix;
    warp_prefix -= thread_sum;
    if (lane_id == WARP_SIZE - 1)
    {
      warp_sums[warp_id] = warp_total;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (tid < WARP_SIZE)
    {
      int32_t val = (tid < num_warps) ? warp_sums[tid] : 0;
      int32_t incl = val;
#pragma unroll
      for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
      {
        int32_t n = __shfl_up(incl, offset, WARP_SIZE);
        if (lane_id >= offset)
        {
          incl += n;
        }
      }
      warp_sums[tid] = incl - val;
    }
    __syncthreads();

    int32_t thread_prefix = warp_sums[warp_id] + warp_prefix;
    int32_t running = 0;
#pragma unroll
    for (int i = 0; i < EXPERTS_PER_THREAD; ++i)
    {
      int eid = my_start + i;
      if (eid < num_experts)
      {
        cumsum[eid] = thread_prefix + running;
      }
      running += local_padded[i];
    }

    if (tid == blockDim.x - 1)
    {
      cumsum[num_experts] = thread_prefix + thread_sum;
      *total_tokens_post_pad = thread_prefix + thread_sum;
    }
    __syncthreads();

    const int32_t total_padded = cumsum[num_experts];
    const int32_t valid_m_blocks = CEILDIV(total_padded, block_size);
    const int32_t max_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
    for (int32_t i = valid_m_blocks + tid; i < max_m_blocks; i += blockDim.x)
    {
      expert_ids[i] = 0;
    }

#pragma unroll
    for (int i = 0; i < EXPERTS_PER_THREAD; ++i)
    {
      int eid = my_start + i;
      if (eid < num_experts)
      {
        for (int j = cumsum[eid]; j < cumsum[eid + 1]; j += block_size)
        {
          expert_ids[j / block_size] = eid;
        }
      }
    }
  }

  // Adapted from current SGLang moe_align_kernel.cu, but keeps AITER native
  // expert id semantics: valid experts are [0, num_experts), and -1 is skipped.
  template <typename scalar_t>
  __global__ void sgl_moe_align_block_size_kernel(
      scalar_t *__restrict__ topk_ids, int32_t *sorted_token_ids,
      int32_t *expert_ids, int32_t *total_tokens_post_pad, int32_t num_experts,
      int32_t block_size, size_t numel, int32_t *cumsum,
      bool pad_sorted_token_ids, int32_t scan_size,
      int32_t max_num_tokens_padded)
  {
    constexpr int kVecSize = 4;
    using Vec = int4;

    if (blockIdx.x == 1)
    {
      if (pad_sorted_token_ids)
      {
        Vec fill_vec;
        fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w =
            static_cast<int32_t>(numel);
        int32_t total_vecs = (max_num_tokens_padded + kVecSize - 1) / kVecSize;
        Vec *out_ptr = reinterpret_cast<Vec *>(sorted_token_ids);
        for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x)
        {
          out_ptr[i] = fill_vec;
        }
      }
      return;
    }

    extern __shared__ int32_t smem[];
    int32_t *shared_counts = smem;
    int32_t *prefix = shared_counts + num_experts;
    int32_t *scan_buf = prefix + num_experts + 1;
    __shared__ int32_t s_total_tokens_post_pad;

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    for (int32_t i = tid; i < num_experts; i += stride)
    {
      shared_counts[i] = 0;
    }
    __syncthreads();

    for (size_t i = tid; i < numel; i += stride)
    {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
      if (expert_id >= 0 && expert_id < num_experts)
      {
        atomicAdd(&shared_counts[expert_id], 1);
      }
    }
    __syncthreads();

    if (tid < num_experts)
    {
      int32_t count = shared_counts[tid];
      scan_buf[tid] = CEILDIV(count, block_size) * block_size;
    }
    if (tid >= num_experts && tid < scan_size)
    {
      scan_buf[tid] = 0;
    }
    __syncthreads();

    int offset = 1;
#pragma unroll
    for (int d = scan_size >> 1; d > 0; d >>= 1)
    {
      if (tid < d)
      {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        scan_buf[bi] += scan_buf[ai];
      }
      offset <<= 1;
      __syncthreads();
    }

    if (tid == 0)
    {
      prefix[num_experts] = scan_buf[scan_size - 1];
      scan_buf[scan_size - 1] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int d = 1; d < scan_size; d <<= 1)
    {
      offset >>= 1;
      if (tid < d)
      {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        if (bi < scan_size)
        {
          int temp = scan_buf[ai];
          scan_buf[ai] = scan_buf[bi];
          scan_buf[bi] += temp;
        }
      }
      __syncthreads();
    }

    for (int32_t i = tid; i < num_experts; i += stride)
    {
      prefix[i] = scan_buf[i];
    }
    if (tid == 0)
    {
      s_total_tokens_post_pad = prefix[num_experts];
      *total_tokens_post_pad = s_total_tokens_post_pad;
    }
    __syncthreads();

    for (int32_t i = tid; i <= num_experts; i += stride)
    {
      cumsum[i] = prefix[i];
    }

    const int32_t num_blocks = s_total_tokens_post_pad / block_size;
    const int32_t max_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
    for (int32_t i = num_blocks + tid; i < max_m_blocks; i += stride)
    {
      expert_ids[i] = 0;
    }

    for (int32_t i = tid; i < num_blocks; i += stride)
    {
      int32_t block_start = i * block_size;
      int32_t left = 0;
      int32_t right = num_experts;
      while (left < right)
      {
        int32_t mid = (left + right) >> 1;
        if (prefix[mid] <= block_start)
        {
          left = mid + 1;
        }
        else
        {
          right = mid;
        }
      }
      expert_ids[i] = left - 1;
    }
  }

  template <typename scalar_t>
  __global__ void sgl_moe_token_sort_kernel(scalar_t *__restrict__ topk_ids,
                                            int32_t *sorted_token_ids,
                                            int32_t *cumsum_buffer,
                                            size_t numel)
  {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < numel; i += stride)
    {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
      if (expert_id >= 0)
      {
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
      }
    }
  }

  template <typename scalar_t, int TOPK>
  __global__ void moe_sum_kernel(
      scalar_t *__restrict__ out,         // [..., d]
      const scalar_t *__restrict__ input, // [..., topk, d]
      int32_t *__restrict__ topk_ids,
      const int d)
  {
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x)
    {
      scalar_t x = 0.0;
#pragma unroll
      for (int k = 0; k < TOPK; ++k)
      {
        if (topk_ids[token_idx * TOPK + k] != -1)
        {
          x += MOE_LDG(&input[token_idx * TOPK * d + k * d + idx]);
        }
      }
      out[token_idx * d + idx] = x;
    }
  }

  template <typename T>
  __device__ __forceinline__ T silu_kernel(const T &x)
  {
    // x * sigmoid(x)
    return (T)(((float)x) / (1.0f + expf((float)-x)));
  }

  template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &),
            bool act_first>
  __device__ __forceinline__ scalar_t compute(const scalar_t &x,
                                              const scalar_t &y)
  {
    return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
  }
  // Activation and gating kernel template.

  template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &),
            bool act_first>
  __global__ void act_and_mul_kernel(
      scalar_t *__restrict__ out,         // [..., d]
      const scalar_t *__restrict__ input, // [..., 2, d]
      const int d)
  {
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x)
    {
      const scalar_t x = MOE_LDG(&input[token_idx * 2 * d + idx]);
      const scalar_t y = MOE_LDG(&input[token_idx * 2 * d + d + idx]);
      out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
  }

  // -----------------------------------------------------------------------
  // CUB-optimized alignment kernel (ported from lightop).
  // Key improvements over the legacy kernel:
  //   1. atomicAdd – all threads count tokens in parallel
  //   2. hipcub::BlockScan – O(log N) prefix sum instead of serial O(N)
  //   3. All CUB_BLOCK_THREADS threads participate (vs num_experts only)
  // -----------------------------------------------------------------------
  template <typename scalar_t, int NUM_EXPERTS, int TOPK, int CUB_BLOCK_THREADS>
  __global__ void moe_align_block_size_kernel_opt_fused_cub(
      const scalar_t *__restrict__ topk_ids,
      int32_t *__restrict__ sorted_token_ids,
      int32_t *__restrict__ expert_ids,
      int32_t *__restrict__ total_tokens_post_pad,
      int32_t num_experts,
      int32_t block_size,
      size_t numel)
  {

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    __shared__ int32_t shared_counts[NUM_EXPERTS + 1];
    __shared__ int32_t raw_counts[NUM_EXPERTS];
    typedef hipcub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // ---- init counts ----
    for (int i = tid; i <= num_experts; i += stride)
      shared_counts[i] = 0;
    __syncthreads();

    // ---- parallel count via atomicAdd ----
    for (size_t i = tid; i < numel; i += stride)
    {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
      if (expert_id != -1)
        atomicAdd(&shared_counts[expert_id], 1);
    }
    __syncthreads();

    // ---- pad to block_size ----
    if (tid < num_experts)
    {
      int32_t cnt = shared_counts[tid];
      raw_counts[tid] = cnt;
      shared_counts[tid] = ((cnt + block_size - 1) / block_size) * block_size;
    }
    __syncthreads();

    // ---- exclusive prefix sum via hipcub::BlockScan ----
    int32_t my_padded = (tid < num_experts) ? shared_counts[tid] : 0;
    int32_t exclusive_prefix = 0;
    BlockScan(temp_storage).ExclusiveScan(my_padded, exclusive_prefix, 0, hipcub::Sum());

    if (tid == num_experts - 1)
    {
      *total_tokens_post_pad = exclusive_prefix + my_padded;
      shared_counts[num_experts] = exclusive_prefix + my_padded;
    }

    // store prefix sum back into shared_counts
    if (tid < num_experts)
      shared_counts[tid] = exclusive_prefix;
    __syncthreads();

    const int32_t total_padded = shared_counts[num_experts];
    const int32_t valid_m_blocks = CEILDIV(total_padded, block_size);
    const int32_t max_m_blocks = CEILDIV(
        static_cast<int32_t>(numel) + num_experts * (block_size - 1),
        block_size);
    for (int32_t i = valid_m_blocks + tid; i < max_m_blocks; i += stride)
    {
      expert_ids[i] = 0;
    }
    __syncthreads();

    // ---- assign expert_ids for each block ----
    if (tid < num_experts)
    {
      const int32_t expert_begin = shared_counts[tid];
      const int32_t expert_end = shared_counts[tid + 1];
      for (int32_t i = expert_begin; i < expert_end; i += block_size)
      {
        expert_ids[i / block_size] = tid;
      }

      for (int32_t i = expert_begin + raw_counts[tid]; i < expert_end; ++i)
      {
        sorted_token_ids[i] = static_cast<int32_t>(numel);
      }
    }
    __syncthreads();

    // ---- sort tokens ----
    for (size_t i = tid; i < numel; i += stride)
    {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
      if (expert_id != -1)
      {
        int32_t rank = atomicAdd(&shared_counts[expert_id], 1);
        sorted_token_ids[rank] = static_cast<int32_t>(i);
      }
    }
  }

} // namespace moe_c

// Helper macro: launch the CUB-optimized kernel for a given expert count.
// CUB_BLOCK_THREADS is set to NUM_EXPERTS to match hipcub::BlockScan requirement.
#define LAUNCH_FAST_ALIGN_KERNEL(NUM_EXPERTS)                     \
  do                                                              \
  {                                                               \
    constexpr int N = (NUM_EXPERTS);                              \
    constexpr int CUB_THREADS = N;                                \
    const size_t numel = topk_ids.numel();                        \
    int threads = 128;                                            \
    if (numel <= 256)                                             \
      threads = 128;                                              \
    else if (numel <= 512)                                        \
      threads = 256;                                              \
    else if (numel <= 1024)                                       \
      threads = 512;                                              \
    else                                                          \
      threads = 1024;                                             \
    if (threads < N)                                              \
      threads = N;                                                \
    if (CUB_THREADS != threads)                                   \
    {                                                             \
      /* Fall back to legacy kernel when thread count mismatch */ \
      break;                                                      \
    }                                                             \
    size_t shared_mem_size = (N + 1) * sizeof(int32_t);           \
    MOE_DISPATCH_INTEGRAL_TYPES(                                  \
        topk_ids.scalar_type(),                                   \
        "moe_align_block_size_kernel_opt_fused_cub", [&] {                      \
          auto kernel =                                                         \
              moe_c::moe_align_block_size_kernel_opt_fused_cub<                 \
                  scalar_t, N, 8, CUB_THREADS>;                                 \
          kernel<<<1, threads, shared_mem_size, stream>>>(                      \
              topk_ids.data_ptr<scalar_t>(),                                    \
              sorted_token_ids.data_ptr<int32_t>(),                             \
              experts_ids.data_ptr<int32_t>(),                                  \
              num_tokens_post_pad.data_ptr<int32_t>(),                          \
              static_cast<int32_t>(num_experts),                                \
              static_cast<int32_t>(block_size), numel); });    \
    return;                                                       \
  } while (0)

void moe_c_moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                                int64_t block_size, torch::Tensor sorted_token_ids,
                                torch::Tensor experts_ids,
                                torch::Tensor num_tokens_post_pad)
{
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // ---- fast path: CUB-optimized kernel for common expert counts ----
  switch (num_experts)
  {
  case 128:
    LAUNCH_FAST_ALIGN_KERNEL(128);
    break;
  case 64:
    LAUNCH_FAST_ALIGN_KERNEL(64);
    break;
  case 256:
    LAUNCH_FAST_ALIGN_KERNEL(256);
    break;
  default: /* fall through to legacy kernel */
    break;
  }

  int device_max_shared_mem;
  auto dev = topk_ids.get_device();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

  const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);
  const int32_t shared_mem_i32 =
      ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
  const int32_t shared_mem_i16 =
      ((num_thread + 1) * num_experts) * sizeof(uint16_t) +
      (num_experts + 1) * sizeof(int32_t);

  bool use_global_memory = false;
  bool use_i16 = false; // Use uint16_t for shared memory token counts
  if (shared_mem_i32 < device_max_shared_mem)
  {
    // Do nothing in this case. We're all set to use int32_t token counts
  }
  else if (shared_mem_i16 < device_max_shared_mem &&
           topk_ids.numel() <= 65535)
  {
    // when nelements of topk_ids is smaller than 65535 (max value of uint16),
    // element value of token_cnts would also smaller than 65535,
    // so we can use uint16 as dtype of token_cnts
    use_i16 = true;
  }
  else
  {
    use_global_memory = true;
  }

  if (use_global_memory)
  {
    MOE_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_global_mem_kernel", [&]
        {
          auto options_int = torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(topk_ids.device());
          torch::Tensor cumsum_buffer =
              torch::zeros({num_experts + 1}, options_int);
          const int threads = 1024;
          const int scan_size = moe_c::next_pow2(static_cast<uint32_t>(num_experts));
          const int64_t max_num_tokens_padded = sorted_token_ids.numel();

          if (num_experts <= 1024) {
            const size_t shared_mem_size =
                (num_experts + (num_experts + 1) + scan_size) * sizeof(int32_t);
            auto kernel =
                moe_c::moe_align_block_size_global_mem_kernel<scalar_t>;
            kernel<<<2, threads, shared_mem_size, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                experts_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
                topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(), true,
                scan_size, static_cast<int32_t>(max_num_tokens_padded));
          } else {
            const int padded_num_experts =
                ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
            const size_t shared_mem_size =
                (padded_num_experts + WARP_SIZE) * sizeof(int32_t);
            auto launch_v2 = [&](auto ept_tag) {
              constexpr int EPT = decltype(ept_tag)::value;
              auto kernel =
                  moe_c::moe_align_block_size_global_mem_kernel_v2<scalar_t, EPT>;
              kernel<<<2, threads, shared_mem_size, stream>>>(
                  topk_ids.data_ptr<scalar_t>(),
                  sorted_token_ids.data_ptr<int32_t>(),
                  experts_ids.data_ptr<int32_t>(),
                  num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                  padded_num_experts, block_size, topk_ids.numel(),
                  cumsum_buffer.data_ptr<int32_t>(), true,
                  static_cast<int32_t>(max_num_tokens_padded));
            };
            if (padded_num_experts <= 2048) {
              launch_v2(std::integral_constant<int, 2>{});
            } else if (padded_num_experts <= 4096) {
              launch_v2(std::integral_constant<int, 4>{});
            } else {
              launch_v2(std::integral_constant<int, 8>{});
            }
          }

          const int block_threads = 256;
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);
          auto sort_kernel = moe_c::sgl_moe_token_sort_kernel<scalar_t>;
          sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel()); });
  }
  else if (use_i16)
  {
    MOE_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&]
        {
          // set dynamic shared mem
          auto kernel =
              moe_c::moe_align_block_size_kernel<scalar_t, uint16_t>;
          AT_CUDA_CHECK(MOE_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i16));
          kernel<<<1, num_thread, shared_mem_i16, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel()); });
  }
  else
  {
    MOE_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&]
        {
          auto kernel =
              moe_c::moe_align_block_size_kernel<scalar_t, int32_t>;
          AT_CUDA_CHECK(MOE_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i32));
          kernel<<<1, num_thread, shared_mem_i32, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel()); });
  }
}

void moe_c_sgl_moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                                    int64_t block_size,
                                    torch::Tensor sorted_token_ids,
                                    torch::Tensor experts_ids,
                                    torch::Tensor num_tokens_post_pad)
{
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(num_experts == 256 || num_experts == 896,
              "sgl_moe_align_block_size kernel only supports 256 or 896 experts.");

  MOE_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "sgl_moe_align_block_size_kernel", [&]
      {
        // calc needed amount of shared mem for `cumsum` tensors
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
        torch::Tensor cumsum_buffer =
            torch::zeros({num_experts + 1}, options_int);
        const int threads = 1024;
        const int scan_size = moe_c::next_pow2(static_cast<uint32_t>(num_experts));
        const int64_t max_num_tokens_padded = sorted_token_ids.numel();
        const size_t shared_mem_size =
            (num_experts + (num_experts + 1) + scan_size) *
            sizeof(int32_t);

        auto align_kernel =
            moe_c::sgl_moe_align_block_size_kernel<scalar_t>;
        align_kernel<<<2, threads, shared_mem_size, stream>>>(
            topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
            experts_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts), block_size,
            topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(), true,
            scan_size, static_cast<int32_t>(max_num_tokens_padded));

        const int block_threads = 256;
        const int num_blocks =
            (topk_ids.numel() + block_threads - 1) / block_threads;
        const int max_blocks = 65535;
        const int actual_blocks = std::min(num_blocks, max_blocks);
        auto sort_kernel = moe_c::sgl_moe_token_sort_kernel<scalar_t>;
        sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel()); });
}

void moe_c_moe_sum(torch::Tensor &input,   // [num_tokens, topk, hidden_size]
                   torch::Tensor &output,  // [num_tokens, hidden_size]
                   torch::Tensor topk_ids) // [num_tokens, top_k]
{
  const int hidden_size = input.size(-1);
  const int num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor topk_ids_int32 = topk_ids.to(torch::kInt32);
  switch (topk)
  {
  case 2:
    MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&]
                                { moe_c::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
                                      output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), topk_ids_int32.data_ptr<int32_t>(),
                                      hidden_size); });
    break;

  case 3:
    MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&]
                                { moe_c::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
                                      output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), topk_ids_int32.data_ptr<int32_t>(),
                                      hidden_size); });
    break;

  case 4:
    MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&]
                                { moe_c::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
                                      output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), topk_ids_int32.data_ptr<int32_t>(),
                                      hidden_size); });
    break;

  case 8:
    MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&]
                                { moe_c::moe_sum_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
                                      output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), topk_ids_int32.data_ptr<int32_t>(),
                                      hidden_size); });
    break;

  default:
    at::sum_out(output, input, 1);
    break;
  }
}

#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                                                                \
  int d = input.size(-1) / 2;                                                                                           \
  int64_t num_tokens = input.numel() / input.size(-1);                                                                  \
  dim3 grid(num_tokens);                                                                                                \
  dim3 block(std::min(d, 1024));                                                                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                                                     \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                         \
  MOE_DISPATCH_FLOATING_TYPES(                                                                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] { moe_c::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST> \
                                                           <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                                                                        input.data_ptr<scalar_t>(), d); });
