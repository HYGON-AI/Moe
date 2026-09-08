// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

#include "moe_wna16_utils.h"
#include "moe_align_sum_kernels.h"

void moe_c_align_moe_align_block_size(torch::Tensor topk_ids,
                                      int64_t num_experts,
                                      int64_t block_size,
                                      torch::Tensor sorted_token_ids,
                                      torch::Tensor experts_ids,
                                      torch::Tensor num_tokens_post_pad) {
  moe_c_moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids,
                             experts_ids, num_tokens_post_pad);
}

void moe_c_align_sgl_moe_align_block_size(torch::Tensor topk_ids,
                                          int64_t num_experts,
                                          int64_t block_size,
                                          torch::Tensor sorted_token_ids,
                                          torch::Tensor experts_ids,
                                          torch::Tensor num_tokens_post_pad) {
  moe_c_sgl_moe_align_block_size(topk_ids, num_experts, block_size,
                                 sorted_token_ids, experts_ids,
                                 num_tokens_post_pad);
}
