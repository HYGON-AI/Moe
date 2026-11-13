// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <cstdint>
#include <limits>
#include <optional>

#include "moe_sorting_api.hpp"

using at::device_of;

namespace aiter {
namespace native {

namespace {

__global__ void ck_sorted_ids_to_moe_c_kernel(const int32_t* __restrict__ ck_sorted_token_ids,
                                              int32_t* __restrict__ moe_c_sorted_token_ids,
                                              int32_t len,
                                              int32_t num_tokens,
                                              int32_t topk)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= len)
    {
        return;
    }

    const uint32_t encoded = static_cast<uint32_t>(ck_sorted_token_ids[idx]);
    const int32_t token_id = static_cast<int32_t>(encoded & 0x00ffffffu);
    const int32_t route_id = static_cast<int32_t>(encoded >> 24);
    const int32_t padding_id = num_tokens * topk;
    moe_c_sorted_token_ids[idx] =
        (token_id >= num_tokens || route_id >= topk) ? padding_id : token_id * topk + route_id;
}

} // namespace

void moe_sorting_fwd(torch::Tensor &topk_ids,          // [m, topk]
                     torch::Tensor &topk_weights,      // [m, topk]
                     torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                     torch::Tensor &sorted_weights,    // [max_num_tokens_padded]
                     torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                     torch::Tensor &tokens_positions_per_expert, // [experts*2]
                     torch::Tensor &num_valid_ids,     // [1]
                     std::optional<torch::Tensor> moe_buf,  // [max_num_tokens_padded], None to skip zero-fill
                     int64_t num_experts,
                     int64_t unit_size,
                     std::optional<torch::Tensor> local_expert_mask)
{
    TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA/HIP tensor");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must have shape [tokens, topk]");
    TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
                "topk_ids must be int32");
    TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes(),
                "topk_weights must have the same shape as topk_ids");
    TORCH_CHECK(num_experts > 0 && unit_size > 0, "num_experts and unit_size must be positive");
    TORCH_CHECK(topk_ids.size(0) < (1 << 24), "tokens exceed the sorted ID encoding range");
    TORCH_CHECK(topk_ids.size(1) > 0 && topk_ids.size(1) <= 255 &&
                topk_ids.size(1) <= num_experts, "unsupported topk");

    const auto check_tensor = [&](const torch::Tensor& tensor, at::ScalarType dtype,
                                  const char* name) {
        TORCH_CHECK(tensor.device() == topk_ids.device(), name, " must be on the topk_ids device");
        TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    };
    check_tensor(topk_weights, at::ScalarType::Float, "topk_weights");
    check_tensor(sorted_token_ids, at::ScalarType::Int, "sorted_token_ids");
    check_tensor(sorted_weights, at::ScalarType::Float, "sorted_weights");
    check_tensor(sorted_expert_ids, at::ScalarType::Int, "sorted_expert_ids");
    check_tensor(tokens_positions_per_expert, at::ScalarType::Int, "tokens_positions_per_expert");
    check_tensor(num_valid_ids, at::ScalarType::Int, "num_valid_ids");
    int num_tokens = topk_ids.size(0);
    int topk = topk_ids.size(1);
    const int64_t capacity = num_tokens == 0 ? 0 :
        topk_ids.numel() + int64_t(num_experts) * unit_size - topk;
    TORCH_CHECK(capacity <= std::numeric_limits<int32_t>::max(), "sorting output is too large");
    TORCH_CHECK(sorted_token_ids.numel() >= capacity && sorted_weights.numel() >= capacity,
                "sorted token/weight buffers are too small");
    TORCH_CHECK(sorted_expert_ids.numel() >= (capacity + unit_size - 1) / unit_size,
                "sorted_expert_ids buffer is too small");
    TORCH_CHECK(tokens_positions_per_expert.numel() >= int64_t(num_experts) * 2 &&
                num_valid_ids.numel() >= 1, "sorting metadata buffers are too small");
    if(local_expert_mask.has_value())
    {
        check_tensor(*local_expert_mask, at::ScalarType::Int, "local_expert_mask");
        TORCH_CHECK(local_expert_mask->dim() == 1 && local_expert_mask->numel() == num_experts,
                    "local_expert_mask must be a length-num_experts 0/1 mask");
    }
    if(moe_buf.has_value())
    {
        TORCH_CHECK(moe_buf->device() == topk_ids.device() && moe_buf->is_contiguous(),
                    "moe_buf must be contiguous and on the topk_ids device");
        TORCH_CHECK(moe_buf->nbytes() % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(moe_buf->data_ptr()) % 16 == 0,
                    "moe_buf must have a 16-byte aligned address and byte size");
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(topk_ids));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    if(num_tokens == 0)
    {
        tokens_positions_per_expert.zero_();
        num_valid_ids.zero_();
        if(moe_buf.has_value())
            moe_buf->zero_();
        return;
    }

    // Check the byte-count arithmetic before CK's int32 workspace-size helper.
    const int64_t mesh_stride = (int64_t(num_tokens) + 31) / 32 * 32;
    const int64_t mesh_bytes = num_tokens < 512 ? 4 : (topk >= 255 ? 2 : 1);
    const int64_t max_workspace = mesh_stride * num_experts * mesh_bytes +
        (int64_t(num_experts) + 32) / 32 * 32 * sizeof(int32_t) + 128;
    TORCH_CHECK(max_workspace <= std::numeric_limits<int32_t>::max(),
                "sorting workspace exceeds int32 indexing");
    int workspace_size = moe_sorting_get_workspace_size(num_tokens, num_experts, topk);
    torch::Tensor ws;
    void *ws_ptr = nullptr;
    if (workspace_size > 0)
    {
        // Keep ownership through all launches; allocation and use share the current stream.
        ws = torch::zeros({workspace_size},
                          torch::TensorOptions().dtype(torch::kUInt8).device(topk_ids.device()));
        ws_ptr = ws.data_ptr();
    }

    const float status = moe_sorting({
                    "int32",                      // index_type
                    "fp32",                       // weight_type; // currently always float
                    local_expert_mask.has_value() // if mask experts as local expert
                },
                {topk_ids.data_ptr(),     // p_topk_ids
                 topk_weights.data_ptr(), // p_weights
                 local_expert_mask.has_value() ? local_expert_mask.value().data_ptr() : nullptr,
                 sorted_token_ids.data_ptr(),  // p_sorted_token_ids
                 sorted_weights.data_ptr(),    // p_sorted_weights
                 sorted_expert_ids.data_ptr(), // p_sorted_expert_ids
                 tokens_positions_per_expert.data_ptr(), // p_tokens_positions_per_expert
                 num_valid_ids.data_ptr(),     // p_total_tokens_post_pad
                 moe_buf.has_value() ? moe_buf.value().data_ptr() : nullptr,  // p_moe_buf
                 ws_ptr,                       // p_workspace
                 num_tokens, static_cast<int>(unit_size), static_cast<int>(num_experts), topk,
                 moe_buf.has_value() ? static_cast<int64_t>(moe_buf.value().nbytes()) : 0},
                {stream});
    TORCH_CHECK(status >= 0, "unsupported moe_sorting configuration");
}

void moe_sorting_ck_ids_to_moe_c(
                     torch::Tensor &ck_sorted_token_ids,
                     torch::Tensor &moe_c_sorted_token_ids,
                     int64_t num_tokens,
                     int64_t topk)
{
    TORCH_CHECK(ck_sorted_token_ids.scalar_type() == at::ScalarType::Int,
                "ck_sorted_token_ids must be int32");
    TORCH_CHECK(moe_c_sorted_token_ids.scalar_type() == at::ScalarType::Int,
                "moe_c_sorted_token_ids must be int32");
    TORCH_CHECK(ck_sorted_token_ids.is_cuda() && moe_c_sorted_token_ids.is_cuda(),
                "sorted token tensors must be CUDA/HIP tensors");
    TORCH_CHECK(ck_sorted_token_ids.numel() == moe_c_sorted_token_ids.numel(),
                "input/output sorted token tensors must have the same length");
    TORCH_CHECK(num_tokens >= 0 && topk > 0, "invalid num_tokens/topk");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(ck_sorted_token_ids));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int64_t len64 = ck_sorted_token_ids.numel();
    if(len64 == 0)
    {
        return;
    }
    TORCH_CHECK(len64 <= std::numeric_limits<int32_t>::max(),
                "sorted token tensor is too large");

    constexpr int block_size = 256;
    const int len = static_cast<int>(len64);
    const int grid = (len + block_size - 1) / block_size;
    ck_sorted_ids_to_moe_c_kernel<<<grid, block_size, 0, stream>>>(
        ck_sorted_token_ids.data_ptr<int32_t>(),
        moe_c_sorted_token_ids.data_ptr<int32_t>(),
        len,
        static_cast<int32_t>(num_tokens),
        static_cast<int32_t>(topk));
}

} // namespace native
} // namespace aiter
