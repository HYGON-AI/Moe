// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <torch/all.h>

#include "moe_wna16_utils.h"
#include "moe_wfp4a16_opt.h"

template <typename scalar_t>
static inline void dispatch_wfp4a16_kernel_map(
    const std::unordered_map<int, KernelFunc_w4a16<scalar_t>>& kernel_map,
    int64_t mode,
    const GemmParams_w4a16<scalar_t>& params,
    const char* timecost_path) {
  auto it = kernel_map.find(mode);
  if (it == kernel_map.end()) {
    printf("wfp4a16 No matching kernel configuration found. mode %ld\n", mode);
    return;
  }

  float milliseconds = 0;
  cudaEvent_t start, stop;
  const char* find_best = std::getenv("WHICH_TO_TEST");
  if (find_best) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  it->second(params);

  if (find_best) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::ofstream ofs(timecost_path, std::ios::app);
    if (ofs.is_open()) {
      ofs << milliseconds << std::endl;
      ofs.close();
    }
  }
}

template <typename scalar_t, typename torch_scalar_t>
static torch::Tensor moe_c_moe_gemm_marlin_wfp4a16_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> b_zeros,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode) {
  const int size_m = input.size(0);
  const int size_k = input.size(1);
  const int size_n = b_scale.size(1);
  constexpr int fp4_delta = 1;
  const bool is_marlin = true;
  const bool first_stage = !topk_weights.has_value();
  torch::Tensor output_alias = output.alias();
  const float* topk_weights_ptr =
      topk_weights.has_value() ? topk_weights.value().data_ptr<float>() : nullptr;
  uint32_t* b_zeros_ptr = b_zeros.has_value()
      ? reinterpret_cast<uint32_t*>(b_zeros.value().data_ptr<uint8_t>())
      : nullptr;

  GemmParams_w4a16<scalar_t> params(
      reinterpret_cast<scalar_t*>(input.data_ptr<torch_scalar_t>()),
      b_qweight.data_ptr<uint32_t>(),
      reinterpret_cast<scalar_t*>(output_alias.data_ptr<torch_scalar_t>()),
      b_zeros_ptr,
      nullptr,
      b_scale.data_ptr<uint8_t>(),
      topk_weights_ptr,
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      0,
      num_tokens_post_pad.data_ptr<int32_t>(),
      size_m,
      size_n,
      size_k,
      sorted_token_ids.size(0),
      top_k,
      fp4_delta,
      is_marlin);

  if (first_stage) {
    const auto& kernel_map = mode >= 500
        ? kernel_maps_gemm1_prefill_wfp4a16<scalar_t>
        : kernel_maps_gemm1_decode_wfp4a16<scalar_t>;
    dispatch_wfp4a16_kernel_map(kernel_map, mode, params, "./wfp4a16_kernel_1_timecost");
  } else {
    const auto& kernel_map = mode >= 500
        ? kernel_maps_gemm2_prefill_wfp4a16<scalar_t>
        : kernel_maps_gemm2_decode_wfp4a16<scalar_t>;
    dispatch_wfp4a16_kernel_map(kernel_map, mode, params, "./wfp4a16_kernel_2_timecost");
  }

  return output;
}

torch::Tensor moe_c_moe_gemm_marlin_wfp4a16(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> b_zeros,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta) {
  (void)delta;
  TORCH_CHECK(b_scale.scalar_type() == at::ScalarType::Byte,
              "wfp4a16 requires uint8 e8m0 scales");

  if (input.scalar_type() == at::ScalarType::Half) {
    return moe_c_moe_gemm_marlin_wfp4a16_impl<half, at::Half>(
        input, b_qweight, output, b_scale, b_zeros, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode);
  }
  if (input.scalar_type() == at::ScalarType::BFloat16) {
    return moe_c_moe_gemm_marlin_wfp4a16_impl<__hip_bfloat16, at::BFloat16>(
        input, b_qweight, output, b_scale, b_zeros, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode);
  }

  TORCH_CHECK(false, "wfp4a16 only supports fp16/bf16 activations");
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def(
      "moe_c_moe_gemm_marlin_wfp4a16("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor b_scale, "
      "Tensor? b_zeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, int mode, "
      "int delta) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_wfp4a16",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_wfp4a16);
}
