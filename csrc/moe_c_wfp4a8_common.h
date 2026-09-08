// SPDX-License-Identifier: MIT

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <torch/all.h>

#include "moe_wfp4a8_opt.h"

template <typename scalar_t>
static inline void dispatch_wfp4a8_kernel_map(
    const std::unordered_map<int, KernelFunc_wfp4a8<scalar_t>>& kernel_map,
    int64_t mode,
    const GemmParams_wfp4a8<scalar_t>& params,
    const char* timecost_path) {
  auto it = kernel_map.find(mode);
  if (it == kernel_map.end()) {
    TORCH_CHECK(false, "unsupported wfp4a8 kernel mode: ", mode);
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
static GemmParams_wfp4a8<scalar_t> make_wfp4a8_params(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    float* b_scale_f32,
    const uint8_t* b_scale_u8,
    int64_t stride_bse,
    int64_t stride_bsn,
    int64_t stride_bsk,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t delta,
    bool force_token_a_scale = false) {
  const int size_m = input.size(0);
  const int size_k = input.size(1);
  const int size_n = b_qweight.size(2) * b_qweight.size(1) / size_k * 2;
  const int stride_asm = a_scale.stride(0);
  const int stride_ask = force_token_a_scale ? 0 : a_scale.stride(1);
  const uint32_t real_topk = delta;
  const float* topk_weights_ptr =
      topk_weights.has_value() ? topk_weights.value().data_ptr<float>() : nullptr;
  torch::Tensor output_alias = output.alias();

  GemmParams_wfp4a8<scalar_t> params(
      reinterpret_cast<const char*>(input.data_ptr()),
      reinterpret_cast<const char*>(b_qweight.data_ptr<uint8_t>()),
      reinterpret_cast<scalar_t*>(output_alias.data_ptr<torch_scalar_t>()),
      reinterpret_cast<float*>(a_scale.data_ptr()),
      b_scale_f32,
      b_scale_u8,
      topk_weights_ptr,
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      0,
      num_tokens_post_pad.data_ptr<int32_t>(),
      size_m,
      size_n,
      size_k,
      stride_asm,
      stride_ask,
      stride_bse,
      stride_bsn,
      stride_bsk,
      sorted_token_ids.size(0),
      top_k,
      real_topk,
      true);
  return params;
}

template <typename scalar_t, typename torch_scalar_t>
static torch::Tensor moe_c_moe_gemm_marlin_wfp4a8_channelwise_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int64_t selected_m) {
  auto params = make_wfp4a8_params<scalar_t, torch_scalar_t>(
      input,
      b_qweight,
      output,
      a_scale,
      reinterpret_cast<float*>(b_scale.data_ptr()),
      nullptr,
      b_scale.stride(0),
      b_scale.stride(1),
      b_scale.stride(2),
      topk_weights,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      top_k,
      delta);

  const bool first_stage = !topk_weights.has_value();
  if (first_stage) {
    const bool prefill = selected_m >= 1024;
    const auto& kernel_map = prefill
        ? kernel_maps_gemm1_prefill_wfp4a8_channelwise<scalar_t>
        : kernel_maps_gemm1_decode_wfp4a8_channelwise<scalar_t>;
    dispatch_wfp4a8_kernel_map(kernel_map, mode, params, "./wfp4a8_kernel_1_timecost");
  } else {
    const bool prefill = selected_m >= 1024;
    const auto& kernel_map = prefill
        ? kernel_maps_gemm2_prefill_wfp4a8_channelwise<scalar_t>
        : kernel_maps_gemm2_decode_wfp4a8_channelwise<scalar_t>;
    dispatch_wfp4a8_kernel_map(kernel_map, mode, params, "./wfp4a8_kernel_2_timecost");
  }

  return output;
}

template <typename scalar_t, typename torch_scalar_t>
static torch::Tensor moe_c_moe_gemm_marlin_wfp4a8_groupwise_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int64_t selected_m) {
  auto params = make_wfp4a8_params<scalar_t, torch_scalar_t>(
      input,
      b_qweight,
      output,
      a_scale,
      nullptr,
      b_scale.data_ptr<uint8_t>(),
      b_scale.stride(0),
      b_scale.stride(1),
      b_scale.stride(2),
      topk_weights,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      top_k,
      delta,
      true);

  const bool first_stage = !topk_weights.has_value();
  if (first_stage) {
    const bool prefill = selected_m >= 1024;
    const auto& kernel_map = prefill
        ? kernel_maps_gemm1_prefill_wfp4a8_groupwise<scalar_t>
        : kernel_maps_gemm1_decode_wfp4a8_groupwise<scalar_t>;
    dispatch_wfp4a8_kernel_map(kernel_map, mode, params, "./wfp4a8_kernel_1_timecost");
  } else {
    const bool prefill = selected_m >= 1024;
    const auto& kernel_map = prefill
        ? kernel_maps_gemm2_prefill_wfp4a8_groupwise<scalar_t>
        : kernel_maps_gemm2_decode_wfp4a8_groupwise<scalar_t>;
    dispatch_wfp4a8_kernel_map(kernel_map, mode, params, "./wfp4a8_kernel_2_timecost");
  }

  return output;
}

template <typename scalar_t, typename torch_scalar_t>
static torch::Tensor moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int64_t selected_m) {
  auto params = make_wfp4a8_params<scalar_t, torch_scalar_t>(
      input,
      b_qweight,
      output,
      a_scale,
      nullptr,
      b_scale.data_ptr<uint8_t>(),
      b_scale.stride(0),
      b_scale.stride(1),
      b_scale.stride(2),
      topk_weights,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      top_k,
      delta,
      false);

  const bool first_stage = !topk_weights.has_value();
  if (first_stage) {
    const bool prefill = selected_m >= 1024;
    const auto& kernel_map = prefill
        ? kernel_maps_gemm1_prefill_wfp4a8_groupwise_qgroup<scalar_t>
        : kernel_maps_gemm1_decode_wfp4a8_groupwise_qgroup<scalar_t>;
    dispatch_wfp4a8_kernel_map(kernel_map, mode, params, "./wfp4a8_kernel_1_timecost");
  } else {
    const bool prefill = selected_m >= 1024;
    const auto& kernel_map = prefill
        ? kernel_maps_gemm2_prefill_wfp4a8_groupwise_qgroup<scalar_t>
        : kernel_maps_gemm2_decode_wfp4a8_groupwise_qgroup<scalar_t>;
    dispatch_wfp4a8_kernel_map(kernel_map, mode, params, "./wfp4a8_kernel_2_timecost");
  }

  return output;
}

static void check_wfp4a8_common(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale) {
  (void)output;
  TORCH_CHECK(input.element_size() == 1,
              "wfp4a8 requires byte-sized fp8 activation input");
  TORCH_CHECK(b_qweight.scalar_type() == at::ScalarType::Byte,
              "wfp4a8 requires uint8 packed fp4 weights");
  TORCH_CHECK(a_scale.scalar_type() == at::ScalarType::Float,
              "wfp4a8 requires fp32 activation scales");
  TORCH_CHECK(b_qweight.dim() == 3,
              "wfp4a8 qweight must have shape [E, N, K/2]");
  TORCH_CHECK(b_scale.dim() == 3,
              "wfp4a8 weight scale must have shape [E, N, scale_k]");
  TORCH_CHECK(input.size(1) % 2 == 0,
              "wfp4a8 only supports even logical K");
  TORCH_CHECK(b_qweight.size(2) * 2 == input.size(1),
              "wfp4a8 qweight packed K mismatch");
}

static torch::Tensor moe_c_moe_gemm_marlin_wfp4a8_channelwise_checked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int64_t size_m) {
  (void)size_m;
  check_wfp4a8_common(input, b_qweight, output, a_scale, b_scale);
  TORCH_CHECK(b_scale.scalar_type() == at::ScalarType::Float,
              "wfp4a8 channelwise requires fp32 weight scales");
  TORCH_CHECK(b_scale.size(2) == 1,
              "wfp4a8 channelwise scale must have shape [E, N, 1]");

  if (output.scalar_type() == at::ScalarType::BFloat16) {
    return moe_c_moe_gemm_marlin_wfp4a8_channelwise_impl<__hip_bfloat16, at::BFloat16>(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, size_m);
  }
  if (output.scalar_type() == at::ScalarType::Half) {
    return moe_c_moe_gemm_marlin_wfp4a8_channelwise_impl<half, at::Half>(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, size_m);
  }

  TORCH_CHECK(false, "wfp4a8 only supports fp16/bf16 output");
}

static torch::Tensor moe_c_moe_gemm_marlin_wfp4a8_groupwise_checked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int64_t size_m) {
  (void)size_m;
  check_wfp4a8_common(input, b_qweight, output, a_scale, b_scale);
  TORCH_CHECK(b_scale.scalar_type() == at::ScalarType::Byte,
              "wfp4a8 groupwise requires uint8 e8m0 weight scales");
  TORCH_CHECK(b_scale.size(2) * 32 == input.size(1),
              "wfp4a8 groupwise scale K mismatch");
  TORCH_CHECK(a_scale.size(-1) == 1,
              "wfp4a8 groupwise per-token activation scale must have last dim 1");

  if (output.scalar_type() == at::ScalarType::BFloat16) {
    return moe_c_moe_gemm_marlin_wfp4a8_groupwise_impl<__hip_bfloat16, at::BFloat16>(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, size_m);
  }
  if (output.scalar_type() == at::ScalarType::Half) {
    return moe_c_moe_gemm_marlin_wfp4a8_groupwise_impl<half, at::Half>(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, size_m);
  }

  TORCH_CHECK(false, "wfp4a8 only supports fp16/bf16 output");
}

static torch::Tensor moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup_checked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int64_t size_m) {
  (void)size_m;
  check_wfp4a8_common(input, b_qweight, output, a_scale, b_scale);
  TORCH_CHECK(b_scale.scalar_type() == at::ScalarType::Byte,
              "wfp4a8 groupwise qgroup requires uint8 e8m0 weight scales");
  TORCH_CHECK(b_scale.size(2) * 32 == input.size(1),
              "wfp4a8 groupwise qgroup scale K mismatch");
  TORCH_CHECK(a_scale.size(-1) * 32 == input.size(1),
              "wfp4a8 groupwise qgroup activation scale K mismatch");

  if (output.scalar_type() == at::ScalarType::BFloat16) {
    return moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup_impl<__hip_bfloat16, at::BFloat16>(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, size_m);
  }
  if (output.scalar_type() == at::ScalarType::Half) {
    return moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup_impl<half, at::Half>(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, size_m);
  }

  TORCH_CHECK(false, "wfp4a8 only supports fp16/bf16 output");
}
