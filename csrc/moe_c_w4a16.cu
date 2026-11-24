// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "moe_w4a16_dispatch.h"
#include <vector>

// Modes with a kernel for GEMM<stage> (1 or 2); the tuner enumerates its candidates from this.
torch::Tensor moe_c_w4a16_modes(int64_t stage) {
  TORCH_CHECK(stage == 1 || stage == 2, "moe_c_w4a16_modes: stage must be 1 or 2");
  std::vector<int64_t> modes;
  auto add = [&](const auto &table) {
    for (const auto &entry : table)
      modes.push_back(entry.id);
  };
  if (stage == 1) {
    add(w4a16_gemm1_decode<half, 32>);
    add(w4a16_gemm1_prefill<half, 32>);
  } else {
    add(w4a16_gemm2_decode<half, 32>);
    add(w4a16_gemm2_prefill<half, 32>);
  }
  return torch::tensor(modes, torch::kInt64);
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
