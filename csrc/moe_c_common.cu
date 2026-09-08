// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <torch/all.h>

#include "moe_wna16_utils.h"
#include "moe_align_sum_kernels.h"
#include "moe_sum_kernels_opt_v2.h"
#include "silu_mul_kernels.h"
#include "topk_softmax_kernel.h"

void moe_c_silu_and_mul(torch::Tensor& out,
                        torch::Tensor& input,
                        int64_t rows_per_block = 1,
                        int64_t vec_size = 2) {
  moe_c::silu_and_mul(
      out, input, static_cast<int>(rows_per_block), static_cast<int>(vec_size));
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def(
      "topk_softmax("
      "Tensor topk_weights, Tensor topk_indices, "
      "Tensor token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &moe_c_topk_softmax);

  m.def(
      "moe_align_block_size("
      "Tensor topk_ids, int num_experts, int block_size, "
      "Tensor sorted_token_ids, Tensor experts_ids, "
      "Tensor num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_c_moe_align_block_size);

  m.def(
      "sgl_moe_align_block_size("
      "Tensor topk_ids, int num_experts, int block_size, "
      "Tensor sorted_token_ids, Tensor experts_ids, "
      "Tensor num_tokens_post_pad) -> ()");
  m.impl(
      "sgl_moe_align_block_size",
      torch::kCUDA,
      &moe_c_sgl_moe_align_block_size);

  m.def("moe_sum(Tensor input, Tensor output, Tensor topk_ids) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_c_moe_sum);

  m.def(
      "moe_sum_opt_v2("
      "Tensor input, Tensor output, float routed_scaling_factor) -> Tensor");
  m.impl("moe_sum_opt_v2", torch::kCUDA, &moe_c_moe_sum_opt_v2);

  m.def(
      "silu_and_mul("
      "Tensor out, Tensor input, int rows_per_block=1, int vec_size=2) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &moe_c_silu_and_mul);
}
