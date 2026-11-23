// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "moe_w4a16_dispatch.h"

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
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
