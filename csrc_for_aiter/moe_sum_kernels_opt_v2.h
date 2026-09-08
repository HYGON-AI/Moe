#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "compat.h"
#include "dispatch_utils.h"
#include "intrinsic_2.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// ============================================================
// Vector type for aligned, vectorized memory access
// ============================================================
template <typename T, int N>
struct alignas(sizeof(T) * N) Vector
{
  T val[N];
};

namespace moe_c
{

  // ============================================================
  // moe_sum kernel v2 — Vector aligned load + 2D grid + float32 reg accum
  // Uses stride-based indexing to match Triton's access pattern for better perf.
  //
  // Computes: out[m, n] = sum_k(input[m, k, n]) * scale
  //
  // Grid: 2D
  //   x: tile parallelism across N dimension
  //   y: token index
  //
  // Key design decisions (memory-bound kernel):
  //   1. Load-one-k-at-a-time — low register pressure → high occupancy
  //   2. BLOCK_SIZE=128 — matches Triton config
  //   3. Vector<T,N> two-DWORD loads (8-byte) — 1 thread handles 4 bf16 / 2 fp32
  //   4. float32 register accum — precision + ILP
  //   5. Template TOPK + #pragma unroll — zero branch overhead
  //   6. Explicit strides (m, k, n) like Triton for optimal memory access
  // ============================================================
  template <typename scalar_t, int TOPK>
  __global__ void moe_sum_kernel_opt_v2(
      scalar_t *__restrict__ out,
      const scalar_t *__restrict__ input,
      const int64_t n_vec, // N / VEC_SIZE (units of Vector<scalar_t, VEC>)
      const float scale,
      const int num_tokens,
      const int64_t stride_input_m, // in elements
      const int64_t stride_input_k,
      const int64_t stride_input_n)
  {

    using native_t = typename std::conditional_t<
        std::is_same_v<scalar_t, c10::Half>, __half,
        typename std::conditional_t<
            std::is_same_v<scalar_t, c10::BFloat16>, __hip_bfloat16,
            scalar_t>>;

    constexpr int VEC = 8 / sizeof(native_t); // two DWORDs (8-byte) per thread: bf16/fp16 -> 4 elems, fp32 -> 2 elems
    using Vec = Vector<native_t, VEC>;

    const int tid = threadIdx.x;

    // One block per row (token m). Grid-stride loop allows one block to handle
    // multiple rows when M > gridDim.x (large batch support).
    for (int64_t m = blockIdx.x; m < num_tokens; m += gridDim.x)
    {
      const int64_t stride_m_vec = stride_input_m / VEC; // assume divisible for vectorized
      const int64_t stride_k_vec = stride_input_k / VEC;
      // stride_n_vec == 1 since we do contiguous Vec loads of VEC elements (stride_n=1)
      const Vec *input_row =
          reinterpret_cast<const Vec *>(input) + m * stride_m_vec;
      // Output is [M, N], so its m-stride in vec units is n_vec (== N / VEC)
      const int64_t stride_out_m_vec = n_vec;
      Vec *out_base = reinterpret_cast<Vec *>(out) + m * stride_out_m_vec;

      // Each thread strides over the N vectors (DWORD loads for bf16 → 2 elems/thread)
      for (int vec_idx = tid; vec_idx < n_vec; vec_idx += blockDim.x)
      {
        // load-and-accumulate one k at a time:
        //   VEC=4 for bf16 → 1 thread handles 4 bf16 (two DWORD loads),
        //   keeps register count low, lets high warp occupancy hide DRAM latency.
        float acc[VEC] = {0};

#pragma unroll
        for (int k = 0; k < TOPK; ++k)
        {
          // same access pattern as Triton: input_row + k*stride_k + offs*stride_n
          // here in vec units: k * stride_k_vec + vec_idx * 1
          Vec v = input_row[k * stride_k_vec + vec_idx];
#pragma unroll
          for (int i = 0; i < VEC; ++i)
          {
            acc[i] += static_cast<float>(v.val[i]);
          }
        }

        Vec result;
#pragma unroll
        for (int i = 0; i < VEC; ++i)
        {
          if constexpr (std::is_same_v<native_t, float>)
          {
            result.val[i] = acc[i] * scale;
          }
          else
          {
            result.val[i] = b32_to_b16<native_t>(acc[i] * scale);
          }
        }
        out_base[vec_idx] = result;
      }
    }
  }

} // namespace moe_c

// ============================================================
// Host-side wrapper: moe_c_moe_sum_opt_v2
// Accepts an output tensor and writes the result directly into it (no extra allocation).
// This eliminates the D2D copy that happens when the caller does:
//     out_hidden_states[chunk] = ops.moe_sum_opt_v2(...)
// ============================================================
torch::Tensor moe_c_moe_sum_opt_v2(
    torch::Tensor &input,
    torch::Tensor &output,
    double routed_scaling_factor = 1.0)
{

  const int64_t M = input.size(0);
  const int topk = input.size(1);
  const int64_t N = input.size(2);

  TORCH_CHECK(output.size(0) == M && output.size(1) == N,
              "moe_c_moe_sum_opt_v2: output shape mismatch, expected [", M, ", ", N,
              "], got [", output.size(0), ", ", output.size(1), "]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // BLOCK_SIZE=128 matches Triton's best config.
  constexpr int BLOCK_SIZE = 128;

  // VEC_SIZE: two DWORDs (8-byte loads) per thread — fp32:2 elems, fp16/bf16:4 elems
  int vec_size = (input.scalar_type() == at::ScalarType::Float) ? 2 : 4;
  int64_t n_vec = N / vec_size;

  // One block per row (token). Use grid-stride inside kernel for M > 65535.
  int grid_size = static_cast<int>(std::min(M, (int64_t)65535));
  dim3 grid(grid_size);
  dim3 block(BLOCK_SIZE);

  // Precompute strides outside the dispatch macro to avoid scope issues on ROCm/HIP
  const int64_t stride_input_m = input.stride(0);
  const int64_t stride_input_k = input.stride(1);
  const int64_t stride_input_n = input.stride(2);

  switch (topk)
  {
#define DISPATCH_TOPK_MOE_SUM_V2(TOPK_VAL)                                                                                                                                                                                                                                                                                                            \
  case TOPK_VAL:                                                                                                                                                                                                                                                                                                                                      \
    MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel_opt_v2", [&] { moe_c::moe_sum_kernel_opt_v2<scalar_t, TOPK_VAL>                                                                                                                                                                                                                  \
                                                                                        <<<grid, block, 0, stream>>>(                                                                                                                                                                                                                                 \
                                                                                            output.data_ptr<scalar_t>(),                                                                                                                                                                                                                              \
                                                                                            input.data_ptr<scalar_t>(),                                                                                                                                                                                                                               \
                                                                                            n_vec, static_cast<float>(routed_scaling_factor), static_cast<int>(M),                                                                                                                                                                                    \
                                                                                            stride_input_m, stride_input_k, stride_input_n); }); \
    break;

    DISPATCH_TOPK_MOE_SUM_V2(1)
    DISPATCH_TOPK_MOE_SUM_V2(2)
    DISPATCH_TOPK_MOE_SUM_V2(3)
    DISPATCH_TOPK_MOE_SUM_V2(4)
    DISPATCH_TOPK_MOE_SUM_V2(5)
    DISPATCH_TOPK_MOE_SUM_V2(6)
    DISPATCH_TOPK_MOE_SUM_V2(7)
    DISPATCH_TOPK_MOE_SUM_V2(8)
    DISPATCH_TOPK_MOE_SUM_V2(9)
    DISPATCH_TOPK_MOE_SUM_V2(10)

#undef DISPATCH_TOPK_MOE_SUM_V2

  default:
    TORCH_CHECK(false, "moe_c_moe_sum_opt_v2: unsupported topk = ", topk,
                ". Supported values: 1-10.");
  }

  return output;
}
