// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kQuantBlockSize = 256;

__device__ __forceinline__ float moe_quant_abs(float value) {
  return value < 0.0f ? -value : value;
}

__global__ void moe_dynamic_per_token_int8_kernel(
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    const float* __restrict__ input,
    int cols,
    const int32_t* __restrict__ num_rows,
    int num_rows_factor) {
  const int row = static_cast<int>(blockIdx.x);
  const int rows = num_rows == nullptr ? gridDim.x : (*num_rows) * num_rows_factor;
  if (row >= rows) {
    return;
  }

  __shared__ float partial[kQuantBlockSize];
  float max_abs = 1.0e-10f;
  const int64_t offset = static_cast<int64_t>(row) * cols;
  for (int col = static_cast<int>(threadIdx.x); col < cols;
       col += kQuantBlockSize) {
    max_abs = fmaxf(max_abs, moe_quant_abs(input[offset + col]));
  }
  partial[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = kQuantBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float scale = partial[0] / 127.0f;
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }
  const float inverse_scale = 1.0f / scale;
  for (int col = static_cast<int>(threadIdx.x); col < cols;
       col += kQuantBlockSize) {
    const float value = nearbyintf(input[offset + col] * inverse_scale);
    const float clamped = fminf(127.0f, fmaxf(-127.0f, value));
    output[offset + col] = static_cast<int8_t>(clamped);
  }
}

template <typename scalar_t>
__global__ void moe_dynamic_per_token_int8_typed_kernel(
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    const scalar_t* __restrict__ input,
    int cols,
    const int32_t* __restrict__ num_rows,
    int num_rows_factor) {
  const int row = static_cast<int>(blockIdx.x);
  const int rows = num_rows == nullptr ? gridDim.x : (*num_rows) * num_rows_factor;
  if (row >= rows) {
    return;
  }

  __shared__ float partial[kQuantBlockSize];
  float max_abs = 1.0e-10f;
  const int64_t offset = static_cast<int64_t>(row) * cols;
  for (int col = static_cast<int>(threadIdx.x); col < cols;
       col += kQuantBlockSize) {
    max_abs = fmaxf(max_abs, moe_quant_abs(static_cast<float>(input[offset + col])));
  }
  partial[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = kQuantBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float scale = partial[0] / 127.0f;
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }
  const float inverse_scale = 1.0f / scale;
  for (int col = static_cast<int>(threadIdx.x); col < cols;
       col += kQuantBlockSize) {
    const float value = nearbyintf(static_cast<float>(input[offset + col]) * inverse_scale);
    const float clamped = fminf(127.0f, fmaxf(-127.0f, value));
    output[offset + col] = static_cast<int8_t>(clamped);
  }
}

}  // namespace

void moe_c_dynamic_per_token_scaled_quant(
    torch::Tensor output,
    torch::Tensor input,
    torch::Tensor scales,
    c10::optional<torch::Tensor> num_rows,
    int64_t num_rows_factor) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda() && scales.is_cuda(),
              "dynamic quantization tensors must be CUDA/HIP tensors");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && scales.is_contiguous(),
              "dynamic quantization tensors must be contiguous");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Char,
              "dynamic INT8 quantization requires int8 output");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
              "dynamic INT8 quantization requires float scales");
  TORCH_CHECK(input.dim() >= 1 && output.sizes() == input.sizes(),
              "input and output shapes must match");
  const int64_t cols64 = input.size(-1);
  TORCH_CHECK(cols64 > 0 && cols64 <= std::numeric_limits<int>::max(),
              "dynamic quantization requires a non-empty last dimension");
  const int64_t rows64 = input.numel() / cols64;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(),
              "dynamic quantization shape is too large");
  TORCH_CHECK(scales.numel() == rows64,
              "scales must contain one value per input row");
  TORCH_CHECK(num_rows_factor > 0 &&
                  num_rows_factor <= std::numeric_limits<int>::max(),
              "num_rows_factor must be positive and fit in int32");
  if (num_rows.has_value()) {
    TORCH_CHECK(num_rows->device() == input.device() && num_rows->is_cuda() &&
                    num_rows->is_contiguous() &&
                    num_rows->scalar_type() == at::ScalarType::Int &&
                    num_rows->numel() >= 1,
                "num_rows must be a contiguous CUDA int32 tensor on the input device");
  }
  if (rows64 == 0) {
    return;
  }

  const int32_t* rows_ptr = num_rows.has_value() ? num_rows->data_ptr<int32_t>() : nullptr;
  const auto device_guard = at::hip::OptionalHIPGuardMasqueradingAsCUDA(device_of(input));
  const hipStream_t stream = at::hip::getCurrentHIPStream();
  const dim3 grid(static_cast<unsigned int>(rows64));
  const dim3 block(kQuantBlockSize);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "moe_dynamic_per_token_scaled_quant",
      [&] {
        moe_dynamic_per_token_int8_typed_kernel<scalar_t><<<grid, block, 0, stream>>>(
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            input.data_ptr<scalar_t>(),
            static_cast<int>(cols64),
            rows_ptr,
            static_cast<int>(num_rows_factor));
      });
}
