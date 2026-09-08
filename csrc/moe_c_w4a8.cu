// SPDX-License-Identifier: MIT

#include <ATen/cuda/CUDAContext.h>
#include <optional>
#include <torch/all.h>

#include "moe_wna16_utils.h"
#include "moe_w4a8_opt.h"

template <typename scalar_t>
static inline bool dispatch_w4a8_kernel_map(
    const std::unordered_map<int, KernelFunc_w4a8<scalar_t>>& kernel_map,
    int64_t mode,
    const GemmParams_w4a8<char, scalar_t>& params) {
  auto it = kernel_map.find(mode);
  if (it == kernel_map.end()) {
    return false;
  }
  it->second(params);
  return true;
}

template <typename scalar_t>
static GemmParams_w4a8<char, scalar_t> make_w4a8_params(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    const float* topk_weights_ptr,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t delta) {
  const int size_m = input.size(0);
  const int size_k = input.size(1);
  const int stride_asm = a_scale.stride(0);
  const int stride_ask = a_scale.stride(1);
  const int stride_bse = b_scale.stride(0);
  const int stride_bsn = b_scale.stride(1);
  const int stride_bsk = b_scale.stride(2);
  const uint32_t real_topk = delta;
  constexpr int num_pad = 0;
  constexpr bool is_marlin = true;
  torch::Tensor output_alias = output.alias();

  return GemmParams_w4a8<char, scalar_t>(
      reinterpret_cast<const char*>(input.data_ptr<int8_t>()),
      reinterpret_cast<const char*>(b_qweight.data_ptr<int8_t>()),
      reinterpret_cast<scalar_t*>(output_alias.data_ptr()),
      reinterpret_cast<float*>(a_scale.data_ptr()),
      reinterpret_cast<float*>(b_scale.data_ptr()),
      topk_weights_ptr,
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_pad,
      num_tokens_post_pad.data_ptr<int32_t>(),
      size_m,
      stride_bse,
      size_k,
      stride_asm,
      stride_ask,
      stride_bse,
      stride_bsn,
      stride_bsk,
      sorted_token_ids.size(0),
      top_k,
      real_topk,
      is_marlin);
}

template <typename scalar_t>
static void moe_c_moe_gemm_marlin_w4a8_impl(
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
    int64_t config_m) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Char,
              "w4a8 requires int8 activation input");

  const bool first_stage = !topk_weights.has_value();
  const bool gemm1_n256_layout = b_scale.stride(0) == 256;
  const float* topk_weights_ptr =
      topk_weights.has_value() ? topk_weights.value().data_ptr<float>() : nullptr;

  auto params = make_w4a8_params<scalar_t>(
      input,
      b_qweight,
      output,
      a_scale,
      b_scale,
      topk_weights_ptr,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      top_k,
      delta);

  bool launched = false;
  if (first_stage) {
    if (config_m <= 512) {
      launched = dispatch_w4a8_kernel_map(
          kernel_maps_gemm1_decode_w4a8<scalar_t>, mode, params);
      TORCH_CHECK(launched, "unsupported w4a8 GEMM1 decode kernel mode: ", mode);
      return;
    }

    if (gemm1_n256_layout) {
      launched = dispatch_w4a8_kernel_map(
          kernel_maps_gemm1_prefill_w4a8_gemm1n256<scalar_t>, mode, params);
      TORCH_CHECK(launched,
                  "unsupported w4a8 GEMM1N256 prefill kernel mode: ",
                  mode);
      return;
    }

    launched = dispatch_w4a8_kernel_map(
        kernel_maps_gemm1_prefill_w4a8<scalar_t>, mode, params);
    TORCH_CHECK(launched, "unsupported w4a8 GEMM1 prefill kernel mode: ", mode);
    return;
  }

  if (config_m <= 512) {
    launched = dispatch_w4a8_kernel_map(
        kernel_maps_gemm2_decode_w4a8<scalar_t>, mode, params);
    TORCH_CHECK(launched, "unsupported w4a8 GEMM2 decode kernel mode: ", mode);
    return;
  }

  launched = dispatch_w4a8_kernel_map(
      kernel_maps_gemm2_prefill_w4a8<scalar_t>, mode, params);
  TORCH_CHECK(launched, "unsupported w4a8 GEMM2 prefill kernel mode: ", mode);
}

torch::Tensor moe_c_moe_gemm_marlin_w4a8(torch::Tensor input,
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
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    moe_c_moe_gemm_marlin_w4a8_impl<bhalf_t>(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        top_k,
        mode,
        delta,
        size_m);
    return output;
  }

  if (output.scalar_type() == at::ScalarType::Half) {
    moe_c_moe_gemm_marlin_w4a8_impl<half>(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        top_k,
        mode,
        delta,
        size_m);
    return output;
  }

  TORCH_CHECK(false, "w4a8 only supports fp16/bf16 output");
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def(
      "moe_c_moe_gemm_marlin_w4a8("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, "
      "Tensor b_scale, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, int mode, "
      "int delta, int size_m) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_w4a8",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_w4a8);
}
