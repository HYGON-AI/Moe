// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

#include "moe_wna16_utils.h"
#include "moe_align_sum_kernels.h"
#include "moe_sum_kernels_opt_v2.h"

void moe_c_sum_moe_sum(torch::Tensor& input,
                       torch::Tensor& output,
                       torch::Tensor topk_ids) {
  moe_c_moe_sum(input, output, topk_ids);
}

torch::Tensor moe_c_sum_moe_sum_opt_v2(torch::Tensor& input,
                                       torch::Tensor& output,
                                       double routed_scaling_factor = 1.0) {
  return moe_c_moe_sum_opt_v2(input, output, routed_scaling_factor);
}
