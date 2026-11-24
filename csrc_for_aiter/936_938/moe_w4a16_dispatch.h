// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <c10/core/DeviceGuard.h>
#include <torch/all.h>
#include <optional>

#include "moe_wna16_utils.h"
#include "moe_w4a16_opt.h"
#include "moe_w4a16_decode.h"

template <typename T, int GS>
static const W4A16Mode<T> *find_w4a16_gemm_mode(bool first_stage, int family, int bm, int profile)
{
  if (first_stage)
    return family == 0 ? find_w4a16_mode(w4a16_gemm1_decode<T, GS>, bm, profile)
                       : find_w4a16_mode(w4a16_gemm1_prefill<T, GS>, bm, profile);
  return family == 0 ? find_w4a16_mode(w4a16_gemm2_decode<T, GS>, bm, profile)
                     : find_w4a16_mode(w4a16_gemm2_prefill<T, GS>, bm, profile);
}

template <typename scalar_t, typename torch_t>
static void run_moe_w4a16_gemm(torch::Tensor &input,
                               torch::Tensor &b_qweight,
                               torch::Tensor &output,
                               torch::Tensor &b_scale,
                               torch::Tensor &b_zeros,
                               const float *topk_weights_ptr,
                               torch::Tensor &sorted_token_ids,
                               torch::Tensor &expert_ids,
                               torch::Tensor &num_tokens_post_pad,
                               int64_t top_k,
                               int64_t mode,
                               int64_t delta,
                               bool first_stage)
{
  const int size_m = input.size(0);
  const int size_k = input.size(1);
  TORCH_CHECK(b_scale.dim() == 3, "W4A16 scales must have shape [E, N, K / group_size]");
  const int size_n = b_scale.size(1);
  const int64_t groups = b_scale.size(2);
  TORCH_CHECK(size_k > 0 && groups > 0 && size_k % groups == 0,
              "W4A16 scale group count ", groups, " does not divide K=", size_k);
  const int group_size = size_k / groups;
  TORCH_CHECK(group_size == 32 || group_size == 128,
              "W4A16 group size must be 32 or 128, got ", group_size);
  const int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
  const int num_pad = 0; // 不读取num_tokens_post_pad[0]，避免device->host拷贝
  const bool is_marlin = true;

  GemmParams_w4a16<scalar_t> params_in(
      (scalar_t *)input.data_ptr<torch_t>(),
      (uint32_t *)b_qweight.data_ptr<uint32_t>(),
      (scalar_t *)output.data_ptr<torch_t>(),
      reinterpret_cast<uint32_t *>(b_zeros.data_ptr<uint8_t>()),
      (scalar_t *)b_scale.data_ptr<torch_t>(),
      topk_weights_ptr,
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_pad,
      num_tokens_post_pad.data_ptr<int32_t>(),
      size_m,
      size_n,
      size_k,
      EM,
      top_k,
      delta,
      is_marlin);

  TORCH_CHECK(is_moe_channel_mode(mode),
              "moe_c_moe_gemm_marlin_w4a16: invalid mode encoding: ", mode);
  TORCH_CHECK(moe_channel_mode_stage(mode) == (first_stage ? 1 : 2),
              "moe_c_moe_gemm_marlin_w4a16: mode stage does not match GEMM", first_stage ? 1 : 2, ": ", mode);
  TORCH_CHECK(moe_channel_mode_sorting(mode) == 1,
              "moe_c_moe_gemm_marlin_w4a16: W4A16 requires moe_align mode: ", mode);
  const int family = moe_channel_mode_family(mode);
  TORCH_CHECK(family == 0 || family == 1,
              "moe_c_moe_gemm_marlin_w4a16: invalid family for mode=", mode);
  const int bm = moe_channel_mode_bm(mode);
  const int profile = moe_channel_mode_profile(mode);
  const auto *entry = group_size == 32
      ? find_w4a16_gemm_mode<scalar_t, 32>(first_stage, family, bm, profile)
      : find_w4a16_gemm_mode<scalar_t, 128>(first_stage, family, bm, profile);
  TORCH_CHECK(entry && entry->launch, "moe_c_moe_gemm_marlin_w4a16: mode=", mode,
              " is not instantiated for group_size=", group_size);
  TORCH_CHECK(top_k > 0 && (first_stage || top_k == 1),
              "moe_c_moe_gemm_marlin_w4a16: GEMM2 requires top_k=1 (one input row per routed slot)");
  TORCH_CHECK(size_n > 0 && (!entry->unfused_mode || size_n % 2 == 0),
              "moe_c_moe_gemm_marlin_w4a16: invalid output N=", size_n, " for mode=", mode);
  const int64_t cols = entry->unfused_mode ? size_n / 2 : size_n;
  TORCH_CHECK(output.is_contiguous() && output.numel() == (int64_t)size_m * top_k * cols,
              "moe_c_moe_gemm_marlin_w4a16: mode=", mode, " needs contiguous output with shape [",
              size_m, ", ", top_k, ", ", cols, "], output has ", output.numel(), " elements");
  if (size_m > 0)
    entry->launch(params_in);
}

torch::Tensor moe_c_moe_gemm_marlin_w4a16(torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor b_scale,
  torch::Tensor b_zeros,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids,
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad,
  int64_t top_k, // gemm1为topk  gemm2为1  因为gemm1输入为[m, k]  gemm2输入为[m*topk, k]
  int64_t mode,
  int64_t delta
  ) {
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    torch::Tensor output_alias = output.alias();

    // GEMM2 kernels read one FP32 routing weight per input row. Keep a typed,
    // contiguous tensor alive through the launch: a raw cast of FP16/BF16
    // storage both misinterprets values and reads past the end of the tensor.
    torch::Tensor topk_weights_fp32;
    const float* topk_weights_ptr = nullptr; // 第一阶段topk_weights为空，第二阶段乘topk权重
    bool first_stage = true;
    if (topk_weights.has_value()){
      const auto &weights = topk_weights.value();
      TORCH_CHECK(weights.is_cuda() && weights.device() == input.device(),
                  "W4A16 topk_weights must be on the same GPU as input");
      TORCH_CHECK(weights.is_floating_point() && weights.numel() == input.size(0),
                  "W4A16 GEMM2 requires one floating-point topk_weight per input row");
      topk_weights_fp32 = weights.to(at::kFloat).contiguous();
      topk_weights_ptr = topk_weights_fp32.data_ptr<float>();
      first_stage = false;
    }

    if (input.scalar_type() == at::ScalarType::Half){
      run_moe_w4a16_gemm<half, at::Half>(
          input, b_qweight, output_alias, b_scale, b_zeros, topk_weights_ptr,
          sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, first_stage);
    } else if (input.scalar_type() == at::ScalarType::BFloat16){
      run_moe_w4a16_gemm<__hip_bfloat16, at::BFloat16>(
          input, b_qweight, output_alias, b_scale, b_zeros, topk_weights_ptr,
          sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta, first_stage);
    } else {
      TORCH_CHECK(false, "moe_c_moe_gemm_marlin_w4a16 only supports fp16/bf16 input");
    }

    return output;
}
