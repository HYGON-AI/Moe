// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "moe_c_common.h"

// mode 格式为 S F K MM VVV：K != 0 为 prefill kernel，K == 0 为 decode kernel；未实例化的
// mode 直接报错，避免静默跳过 kernel 导致输出未初始化（tune 依赖该报错区分非法候选）。
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
  const int size_n = b_scale.size(1);
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
  const bool prefill = moe_channel_mode_family(mode) != 0;
  auto &kernels = first_stage
                      ? (prefill ? kernel_maps_gemm1_prefill_w4a16<scalar_t> : kernel_maps_gemm1_decode_w4a16<scalar_t>)
                      : (prefill ? kernel_maps_gemm2_prefill_w4a16<scalar_t> : kernel_maps_gemm2_decode_w4a16<scalar_t>);
  auto it = kernels.find(mode);
  TORCH_CHECK(it != kernels.end(),
              "moe_c_moe_gemm_marlin_w4a16: GEMM", first_stage ? 1 : 2, " ",
              prefill ? "prefill" : "decode", " mode=", mode, " is not instantiated");
  it->second(params_in);
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

// Modes with a kernel for GEMM<stage> (1 or 2); the tuner enumerates its candidates from this.
torch::Tensor moe_c_w4a16_modes(int64_t stage) {
  TORCH_CHECK(stage == 1 || stage == 2, "moe_c_w4a16_modes: stage must be 1 or 2");
  return torch::tensor(w4a16_registered_modes(stage), torch::kInt64);
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def("moe_c_w4a16_modes(int stage) -> Tensor");
  m.impl("moe_c_w4a16_modes", c10::DispatchKey::CompositeExplicitAutograd, &moe_c_w4a16_modes);
  m.def(
      "moe_c_moe_gemm_marlin_w4a16("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor b_scale, "
      "Tensor b_zeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, int mode, "
      "int delta) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_w4a16",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_w4a16);
}
