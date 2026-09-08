// SPDX-License-Identifier: MIT

#include "moe_c_wfp4a8_common.h"

torch::Tensor moe_c_moe_gemm_marlin_wfp4a8_channelwise(
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
  return moe_c_moe_gemm_marlin_wfp4a8_channelwise_checked(
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
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def(
      "moe_c_moe_gemm_marlin_wfp4a8_channelwise("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, "
      "Tensor b_scale, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, int mode, "
      "int delta, int size_m) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_wfp4a8_channelwise",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_wfp4a8_channelwise);
}
