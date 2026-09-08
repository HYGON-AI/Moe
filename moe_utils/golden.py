# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .utils import moe_kernel_prepare_input, per_token_quant_hip, pertoken_quant
from .fused_moe_c import per_token_group_quant_fp8

from .utils import _apply_activation


def triton_moe_sum(input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float = 1.0) -> None:
    output.copy_(input.to(torch.float32).sum(dim=1).mul_(routed_scaling_factor).to(output.dtype))


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported Triton golden dtype: {dtype}")


def _apply_wfp4a16_torch_activation(
    stage1: torch.Tensor,
    inter: int,
    activation: str,
    compute_dtype: torch.dtype,
    gemm1_alpha: float | None = None,
    gemm1_limit: float | None = None,
) -> torch.Tensor:
    gate = stage1[..., :inter].reshape(-1, inter).to(torch.float32)
    up = stage1[..., inter:].reshape(-1, inter).to(torch.float32)

    if activation == "silu":
        out = F.silu(gate) * up
    elif activation == "situ":
        situ_beta = 4.0 if gemm1_alpha is None else gemm1_alpha
        gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
        if gemm1_limit is not None:
            up = gemm1_limit * torch.tanh(up / gemm1_limit)
        out = gate * up
    elif activation == "gelu":
        out = F.gelu(gate) * up
    elif activation == "gelu_tanh":
        out = F.gelu(gate, approximate="tanh") * up
    else:
        raise ValueError(f"Unsupported WFP4A16 torch golden activation: {activation}")
    return out.to(compute_dtype)


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2
) -> None:
    """Print detailed tensor diff statistics and a few sample mismatches."""
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量（torch.Tensor）")

    if tensor1.shape != tensor2.shape:
        raise ValueError(f"张量形状不匹配！ tensor1形状: {tensor1.shape}, tensor2形状: {tensor2.shape}")

    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
        print(f"警告：张量设备不一致，已将tensor2转移到{tensor1.device}")

    abs_diff = torch.abs(tensor1 - tensor2)
    denom = torch.maximum(torch.abs(tensor1), torch.abs(tensor2))
    rel_diff = abs_diff / (denom + 1e-12)
    match_mask = (abs_diff <= atol) | (rel_diff <= rtol)

    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    abs_diff_flat = abs_diff.flatten()
    match_mask_flat = match_mask.flatten()

    def get_indices_1d(mask: torch.Tensor) -> list:
        indices = torch.nonzero(mask).squeeze(dim=1)
        return indices.tolist() if indices.numel() > 0 else []

    match_indices_1d = get_indices_1d(match_mask_flat)
    mismatch_indices_1d = get_indices_1d(~match_mask_flat)

    total = tensor1_flat.numel()
    matched = len(match_indices_1d)
    mismatched = len(mismatch_indices_1d)
    match_rate = matched / total if total > 0 else 0.0
    max_abs_diff = abs_diff.max().item() if total > 0 else 0.0
    avg_abs_diff = abs_diff.mean().item() if total > 0 else 0.0

    print("=" * 60)
    print("张量比较结果汇总")
    print("=" * 60)
    print(f"张量形状: {tensor1.shape} | 总元素数: {total}")
    print(f"阈值设置: 绝对误差(atol)={atol:.2e}, 相对误差(rtol)={rtol:.2e}")
    print("-" * 60)
    print(f"匹配元素数: {matched} ({match_rate:.2%})")
    print(f"不匹配元素数: {mismatched} ({1 - match_rate:.2%})")
    print(f"最大绝对差异: {max_abs_diff:.6f}")
    print(f"平均绝对差异: {avg_abs_diff:.6f}")
    print("=" * 60)

    def print_sample(name: str, indices_1d: list, max_samples: int = 3, elem_per_sample: int = 10) -> None:
        if not indices_1d:
            print(f"\n【{name}样本】无数据")
            return

        print(f"\n【{name}样本】（最多展示{max_samples}组，每组{elem_per_sample}个元素）")
        print("-" * 50)

        num_samples = min(max_samples, (len(indices_1d) + elem_per_sample - 1) // elem_per_sample)
        for i in range(num_samples):
            start = i * elem_per_sample
            end = start + elem_per_sample
            sample_indices_1d = indices_1d[start:end]
            sample_coords = torch.unravel_index(torch.tensor(sample_indices_1d), tensor1.shape)
            sample_coords_list = list(zip(*[coord.tolist() for coord in sample_coords]))

            print(f"\n第{i + 1}组:")
            print(f"  原始多维坐标: {sample_coords_list}")
            print(f"  tensor1: {[round(tensor1_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  tensor2: {[round(tensor2_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  绝对差异: {[round(abs_diff_flat[idx].item(), 6) for idx in sample_indices_1d]}")

    print_sample("匹配", match_indices_1d, max_samples=2)
    print_sample("不匹配", mismatch_indices_1d, max_samples=3)
    print("\n" + "=" * 60)


def run_w4a8_perchannel_golden(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    out_dtype,
    w1_scale,
    w2_scale,
    routed_scaling_factor=1.0,
    return_intermediates=False,
):
    """Torch golden for W4A8 per-channel MoE.

    w1/w2 are unpacked signed int4 weights:
      - w1: [E, 2 * inter, K]
      - w2: [E, hidden, inter]

    This mirrors the W4A8 kernel numerics: per-token int8 activation
    quantization, per-channel int4 weight scales, silu_and_mul, then routed
    weighted sum.
    """
    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    w1 = w1.to(torch.float32).contiguous()
    w2 = w2.to(torch.float32).contiguous()
    w1_scale = w1_scale.to(torch.float32).contiguous()
    w2_scale = w2_scale.to(torch.float32).contiguous()

    m, _ = hidden_states.shape
    topk = topk_ids.shape[1]
    n1 = w1.shape[1]
    hidden_out = w2.shape[1]
    inter = n1 // 2

    stage1 = torch.empty((m, topk, n1), device=hidden_states.device, dtype=out_dtype)
    stage2 = torch.empty((m, topk, hidden_out), device=hidden_states.device, dtype=out_dtype)

    qinput1, a1_scale = moe_kernel_prepare_input(
        A=hidden_states,
        B=w1,
        A_scale=None,
        B_scale=w1_scale,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=None,
    )
    qinput1 = qinput1.to(torch.float32)
    a1_scale = a1_scale.to(torch.float32)

    for token_idx in range(m):
        activation = qinput1[token_idx] * a1_scale[token_idx]
        for route_idx in range(topk):
            expert = int(topk_ids[token_idx, route_idx].item())
            fc1_weight = w1[expert] * w1_scale[expert]
            stage1[token_idx, route_idx] = activation.matmul(fc1_weight.t()).to(out_dtype)

    stage1_2d = stage1.view(m * topk, n1)
    activated = (
        F.silu(stage1_2d[:, :inter].to(torch.float32))
        * stage1_2d[:, inter:].to(torch.float32)
    ).to(out_dtype)

    qinput2, a2_scale = moe_kernel_prepare_input(
        A=activated.contiguous(),
        B=w2,
        A_scale=None,
        B_scale=w2_scale,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=None,
    )
    qinput2 = qinput2.to(torch.float32)
    a2_scale = a2_scale.to(torch.float32)

    for token_idx in range(m):
        for route_idx in range(topk):
            route = token_idx * topk + route_idx
            expert = int(topk_ids[token_idx, route_idx].item())
            activation = qinput2[route] * a2_scale[route]
            fc2_weight = w2[expert] * w2_scale[expert]
            route_out = activation.matmul(fc2_weight.t())
            route_out = route_out * topk_weights[token_idx, route_idx].to(torch.float32)
            stage2[token_idx, route_idx] = route_out.to(out_dtype)

    out = stage2.sum(dim=1).to(out_dtype)
    if routed_scaling_factor != 1.0:
        out = (out * routed_scaling_factor).to(out_dtype)

    if return_intermediates:
        return out, {
            "stage1": stage1,
            "activated": activated,
            "stage2": stage2,
        }
    return out



def _torch_per_token_quant_int8(
    x: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.to(torch.float32)
    scale = torch.amax(torch.abs(x_f), dim=-1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.round(x_f / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def run_w4a8_perchannel_torch_golden(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    out_dtype,
    w1_scale,
    w2_scale,
    routed_scaling_factor=1.0,
    return_intermediates=False,
    route_chunk_size=16,
):
    """Pure torch fallback golden for W4A8 per-channel MoE.

    This intentionally avoids aiter/Triton helper kernels, including activation
    quantization. It is slower than run_w4a8_perchannel_golden, but is useful as
    an independent correctness reference.

    Args follow run_w4a8_perchannel_golden. w1/w2 are unpacked signed int4:
      - w1: [E, 2 * inter, K]
      - w2: [E, hidden, inter]
    """
    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    w1 = w1.to(torch.float32).contiguous()
    w2 = w2.to(torch.float32).contiguous()
    w1_scale = w1_scale.to(torch.float32).contiguous()
    w2_scale = w2_scale.to(torch.float32).contiguous()

    if w1_scale.shape[-1] == 1:
        w1_scale = w1_scale.squeeze(-1)
    if w2_scale.shape[-1] == 1:
        w2_scale = w2_scale.squeeze(-1)

    m, _ = hidden_states.shape
    topk = topk_ids.shape[1]
    n1 = w1.shape[1]
    inter = n1 // 2
    hidden_out = w2.shape[1]
    num_routes = m * topk

    qinput1, a1_scale = _torch_per_token_quant_int8(hidden_states)
    input1 = qinput1.to(torch.float32) * a1_scale

    flat_experts = topk_ids.reshape(-1).to(torch.long)
    flat_route_weights = topk_weights.reshape(-1).to(torch.float32)
    flat_tokens = (
        torch.arange(m, device=hidden_states.device, dtype=torch.long)
        .repeat_interleave(topk)
    )

    out_acc = torch.zeros((m, hidden_out), device=hidden_states.device, dtype=torch.float32)

    if return_intermediates:
        stage1_all = torch.empty((m, topk, n1), device=hidden_states.device, dtype=out_dtype)
        activated_all = torch.empty((num_routes, inter), device=hidden_states.device, dtype=out_dtype)
        stage2_all = torch.empty((m, topk, hidden_out), device=hidden_states.device, dtype=out_dtype)

    for start in range(0, num_routes, route_chunk_size):
        end = min(start + route_chunk_size, num_routes)
        route_ids = torch.arange(start, end, device=hidden_states.device, dtype=torch.long)
        token_ids = flat_tokens[start:end]
        route_lanes = route_ids - token_ids * topk
        expert_ids = flat_experts[start:end]

        a1 = input1.index_select(0, token_ids)
        fc1_weight = w1.index_select(0, expert_ids) * w1_scale.index_select(0, expert_ids).unsqueeze(-1)
        stage1 = torch.bmm(fc1_weight, a1.unsqueeze(-1)).squeeze(-1).to(out_dtype)

        activated = (
            F.silu(stage1[:, :inter].to(torch.float32))
            * stage1[:, inter:].to(torch.float32)
        ).to(out_dtype)

        qinput2, a2_scale = _torch_per_token_quant_int8(activated)
        a2 = qinput2.to(torch.float32) * a2_scale
        fc2_weight = w2.index_select(0, expert_ids) * w2_scale.index_select(0, expert_ids).unsqueeze(-1)
        route_out = torch.bmm(fc2_weight, a2.unsqueeze(-1)).squeeze(-1)
        route_out = route_out * flat_route_weights[start:end].unsqueeze(-1)

        out_acc.index_add_(0, token_ids, route_out.to(torch.float32))

        if return_intermediates:
            stage1_all[token_ids, route_lanes] = stage1
            activated_all[start:end] = activated
            stage2_all[token_ids, route_lanes] = route_out.to(out_dtype)

    out = out_acc.to(out_dtype)
    if routed_scaling_factor != 1.0:
        out = (out * routed_scaling_factor).to(out_dtype)

    if return_intermediates:
        return out, {
            "stage1": stage1_all,
            "activated": activated_all,
            "stage2": stage2_all,
        }
    return out


@triton.jit
def _wfp4a16_moe_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsk: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    # Adapted from SGLang fused_moe_kernel_gptq_awq's MXFP4/W4A16 path.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_idxs = k0 * BLOCK_SIZE_K + offs_k
        a = tl.load(
            a_ptr
            + (offs_token[:, None] // TOP_K) * stride_am
            + k_idxs[None, :] * stride_ak,
            mask=token_mask[:, None] & (k_idxs[None, :] < K),
            other=0.0,
        )

        packed = tl.load(
            b_ptr
            + off_expert * stride_be
            + offs_n[None, :] * stride_bn
            + (k_idxs[:, None] // 2) * stride_bk,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0,
        )
        nibble = (packed >> ((k_idxs[:, None] % 2) * 4)) & 0xF
        magnitude_code = nibble & 0x7
        magnitude = tl.where(
            magnitude_code <= 4,
            magnitude_code.to(tl.float32) * 0.5,
            tl.where(
                magnitude_code == 5,
                3.0,
                tl.where(magnitude_code == 6, 4.0, 6.0),
            ),
        )
        b = tl.where((nibble & 0x8) == 0, magnitude, -magnitude)

        b_scale = tl.load(
            b_scale_ptr
            + off_expert * stride_bse
            + offs_n[None, :] * stride_bsn
            + (k_idxs[:, None] // group_size) * stride_bsk,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0,
        )
        b = (b * tl.exp2(b_scale.to(tl.float32) - 127.0)).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(
        c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator.to(compute_type),
        mask=token_mask[:, None] & (offs_cn[None, :] < N),
    )


def _align_routes_for_triton_golden(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    flat_ids = topk_ids.reshape(-1).detach().cpu()
    sorted_ids = []
    expert_ids = []
    num_valid = flat_ids.numel()
    for expert in range(num_experts):
        ids = torch.nonzero(flat_ids == expert, as_tuple=False).flatten().tolist()
        if not ids:
            continue
        padded = ((len(ids) + block_size - 1) // block_size) * block_size
        ids.extend([num_valid] * (padded - len(ids)))
        sorted_ids.extend(ids)
        expert_ids.extend([expert] * (padded // block_size))

    if not sorted_ids:
        sorted_ids = [num_valid] * block_size
        expert_ids = [0]

    device = topk_ids.device
    sorted_token_ids = torch.tensor(sorted_ids, device=device, dtype=torch.int64)
    expert_ids = torch.tensor(expert_ids, device=device, dtype=torch.int64)
    num_tokens_post_padded = torch.tensor([len(sorted_ids)], device=device, dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _run_wfp4a16_stage(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_size_m: int,
    group_size: int,
    mul_routed_weight: bool,
    top_k_for_a: int,
    output_shape,
    compute_dtype,
    block_size_n: int = 64,
    block_size_k: int = 64,
) -> torch.Tensor:
    num_experts = b_qweight.shape[0]
    n = b_qweight.shape[1]
    k = a.shape[1]
    sorted_token_ids, expert_ids, num_tokens_post_padded = _align_routes_for_triton_golden(
        topk_ids, num_experts, block_size_m)

    out = torch.empty(output_shape, device=a.device, dtype=compute_dtype)
    flat_out = out.view(-1, n)
    grid = (
        triton.cdiv(sorted_token_ids.numel(), block_size_m)
        * triton.cdiv(n, block_size_n),
    )
    _wfp4a16_moe_gemm_kernel[grid](
        a,
        b_qweight,
        flat_out,
        b_scale,
        topk_weights.reshape(-1).contiguous(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        sorted_token_ids.numel(),
        topk_ids.numel(),
        a.stride(0),
        a.stride(1),
        b_qweight.stride(0),
        b_qweight.stride(1),
        b_qweight.stride(2),
        flat_out.stride(0),
        flat_out.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=8,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        TOP_K=top_k_for_a,
        compute_type=_torch_dtype_to_triton(compute_dtype),
    )
    return out


def _run_wfp4a16_stage_v2(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict,
    group_size: int,
    mul_routed_weight: bool,
    top_k_for_a: int,
    output_shape,
    compute_dtype,
) -> torch.Tensor:
    """SGLang-style WFP4A16 stage: use caller-provided align buffers/config."""
    n = b_qweight.shape[1]
    k = a.shape[1]
    block_size_m = int(config["BLOCK_SIZE_M"])
    block_size_n = int(config.get("BLOCK_SIZE_N", 64))
    block_size_k = int(config.get("BLOCK_SIZE_K", 64))
    group_size_m = int(config.get("GROUP_SIZE_M", 8))

    out = torch.empty(output_shape, device=a.device, dtype=compute_dtype)
    flat_out = out.view(-1, n)
    num_valid_tokens = int(topk_ids.numel())
    em = int(num_tokens_post_padded.item())
    grid = (
        triton.cdiv(em, block_size_m) * triton.cdiv(n, block_size_n),
    )
    _wfp4a16_moe_gemm_kernel[grid](
        a,
        b_qweight,
        flat_out,
        b_scale,
        topk_weights.reshape(-1).contiguous(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        em,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b_qweight.stride(0),
        b_qweight.stride(1),
        b_qweight.stride(2),
        flat_out.stride(0),
        flat_out.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        TOP_K=top_k_for_a,
        compute_type=_torch_dtype_to_triton(compute_dtype),
    )
    return out


def run_wfp4a16_triton_golden(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int = 32,
    block_size_m: int = 16,
    output_dtype: torch.dtype = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Triton golden for packed FP4 E2M1 weight + BF16/FP16 activation MoE.

    Weight layout follows the AITER WFP4A16 test path:
      - w1_qweight: [E, 2 * intermediate, K / 2]
      - w2_qweight: [E, hidden, intermediate / 2]
      - scales: uint8 e8m0 [E, N, K / group_size]
    """
    compute_dtype = output_dtype or hidden_states.dtype
    hidden_states = hidden_states.contiguous()
    w1_qweight = w1_qweight.contiguous()
    w2_qweight = w2_qweight.contiguous()
    w1_scale = w1_scale.contiguous()
    w2_scale = w2_scale.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()

    m, _ = hidden_states.shape
    topk = topk_ids.shape[1]
    n1 = w1_qweight.shape[1]
    inter = n1 // 2
    hidden_out = w2_qweight.shape[1]

    stage1 = _run_wfp4a16_stage(
        hidden_states,
        w1_qweight,
        w1_scale,
        topk_weights,
        topk_ids,
        block_size_m,
        group_size,
        mul_routed_weight=False,
        top_k_for_a=topk,
        output_shape=(m, topk, n1),
        compute_dtype=compute_dtype,
    )
    activated = _apply_wfp4a16_torch_activation(stage1, inter, activation, compute_dtype)

    stage2 = _run_wfp4a16_stage(
        activated.contiguous(),
        w2_qweight,
        w2_scale,
        topk_weights,
        topk_ids,
        block_size_m,
        group_size,
        mul_routed_weight=True,
        top_k_for_a=1,
        output_shape=(m, topk, hidden_out),
        compute_dtype=compute_dtype,
    )
    return stage2.sum(dim=1).to(compute_dtype)


def run_wfp4a16_triton_v2_golden(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict,
    down_config: dict | None = None,
    group_size: int = 32,
    output_dtype: torch.dtype = None,
    activation: str = "silu",
    routed_scaling_factor: float | None = 1.0,
    gemm1_alpha: float | None = None,
    gemm1_limit: float | None = None,
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
) -> torch.Tensor:
    """Triton v2 golden that mirrors SGLang's fused_moe.py sequence.

    Unlike run_wfp4a16_triton_golden, this path consumes the precomputed MoE
    alignment buffers and the exact config dict dumped from the caller.
    """
    compute_dtype = output_dtype or hidden_states.dtype
    hidden_states = hidden_states.contiguous()
    w1_qweight = w1_qweight.contiguous()
    w2_qweight = w2_qweight.contiguous()
    w1_scale = w1_scale.contiguous()
    w2_scale = w2_scale.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    sorted_token_ids = sorted_token_ids.contiguous()
    expert_ids = expert_ids.contiguous()
    num_tokens_post_padded = num_tokens_post_padded.contiguous()

    m = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    n1 = w1_qweight.shape[1]
    inter = n1 // 2
    hidden_out = w2_qweight.shape[1]
    total_tokens = m * topk

    stage1 = _run_wfp4a16_stage_v2(
        hidden_states,
        w1_qweight,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        group_size,
        mul_routed_weight=apply_router_weight_on_input,
        top_k_for_a=topk,
        output_shape=(total_tokens, n1),
        compute_dtype=compute_dtype,
    )
    activated = _apply_wfp4a16_torch_activation(
        stage1,
        inter,
        activation,
        compute_dtype,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
    )

    stage2 = _run_wfp4a16_stage_v2(
        activated.contiguous(),
        w2_qweight,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        down_config or config,
        group_size,
        mul_routed_weight=(not apply_router_weight_on_input and not no_combine),
        top_k_for_a=1,
        output_shape=(m, topk, hidden_out),
        compute_dtype=compute_dtype,
    )
    if no_combine:
        return stage2.to(compute_dtype)

    out = torch.empty((m, hidden_out), device=hidden_states.device, dtype=compute_dtype)
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    triton_moe_sum(stage2, out)
    if scale != 1.0:
        out = (out * scale).to(compute_dtype)
    return out


@triton.jit
def _wfp4a8_moe_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_idxs = k0 * BLOCK_SIZE_K + offs_k
        a = tl.load(
            a_ptr
            + (offs_token[:, None] // TOP_K) * stride_am
            + k_idxs[None, :] * stride_ak,
            mask=token_mask[:, None] & (k_idxs[None, :] < K),
            other=0.0,
        ).to(compute_type)

        packed = tl.load(
            b_ptr
            + off_expert * stride_be
            + offs_n[None, :] * stride_bn
            + (k_idxs[:, None] // 2) * stride_bk,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0,
        )
        nibble = (packed >> ((k_idxs[:, None] % 2) * 4)) & 0xF
        magnitude_code = nibble & 0x7
        magnitude = tl.where(
            magnitude_code <= 4,
            magnitude_code.to(tl.float32) * 0.5,
            tl.where(
                magnitude_code == 5,
                3.0,
                tl.where(magnitude_code == 6, 4.0, 6.0),
            ),
        )
        b = tl.where((nibble & 0x8) == 0, magnitude, -magnitude).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

    a_scale = tl.load(a_scale_ptr + (offs_token // TOP_K), mask=token_mask, other=0.0).to(tl.float32)
    b_scale = tl.load(
        b_scale_ptr
        + off_expert * stride_bse
        + offs_n * stride_bsn,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    accumulator = accumulator * a_scale[:, None] * b_scale[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(
        c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator.to(compute_type),
        mask=token_mask[:, None] & (offs_cn[None, :] < N),
    )


def _run_wfp4a8_stage_v2(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict,
    mul_routed_weight: bool,
    top_k_for_a: int,
    output_shape,
    compute_dtype,
) -> torch.Tensor:
    n = b_qweight.shape[1]
    k = a.shape[1]
    block_size_m = int(config["BLOCK_SIZE_M"])
    block_size_n = int(config.get("BLOCK_SIZE_N", 64))
    block_size_k = int(config.get("BLOCK_SIZE_K", 64))
    group_size_m = int(config.get("GROUP_SIZE_M", 8))

    out = torch.empty(output_shape, device=a.device, dtype=compute_dtype)
    flat_out = out.view(-1, n)
    num_valid_tokens = int(topk_ids.numel())
    em = int(num_tokens_post_padded.item())
    grid = (
        triton.cdiv(em, block_size_m) * triton.cdiv(n, block_size_n),
    )
    _wfp4a8_moe_gemm_kernel[grid](
        a,
        a_scale.reshape(-1).contiguous(),
        b_qweight,
        flat_out,
        b_scale,
        topk_weights.reshape(-1).contiguous(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        em,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b_qweight.stride(0),
        b_qweight.stride(1),
        b_qweight.stride(2),
        flat_out.stride(0),
        flat_out.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        TOP_K=top_k_for_a,
        compute_type=_torch_dtype_to_triton(compute_dtype),
    )
    return out



@triton.jit
def _wfp4a8_groupwise_postscale_moe_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsk: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    A_GROUP_SCALE: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, GROUP_SIZE).to(tl.int64)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for group_k in range(0, tl.cdiv(K, GROUP_SIZE)):
        if A_GROUP_SCALE:
            a_scale = tl.load(
                a_scale_ptr + (offs_token // TOP_K) * stride_asm + group_k * stride_ask,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            a_scale = tl.load(
                a_scale_ptr + (offs_token // TOP_K) * stride_asm,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
        k_idxs = group_k * GROUP_SIZE + offs_k
        a = tl.load(
            a_ptr
            + (offs_token[:, None] // TOP_K) * stride_am
            + k_idxs[None, :] * stride_ak,
            mask=token_mask[:, None] & (k_idxs[None, :] < K),
            other=0.0,
        ).to(compute_type)

        packed = tl.load(
            b_ptr
            + off_expert * stride_be
            + offs_n[None, :] * stride_bn
            + (k_idxs[:, None] // 2) * stride_bk,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0,
        )
        nibble = (packed >> ((k_idxs[:, None] % 2) * 4)) & 0xF
        magnitude_code = nibble & 0x7
        magnitude = tl.where(
            magnitude_code <= 4,
            magnitude_code.to(tl.float32) * 0.5,
            tl.where(
                magnitude_code == 5,
                3.0,
                tl.where(magnitude_code == 6, 4.0, 6.0),
            ),
        )
        b = tl.where((nibble & 0x8) == 0, magnitude, -magnitude).to(compute_type)
        partial = tl.dot(a, b)

        b_scale_u8 = tl.load(
            b_scale_ptr
            + off_expert * stride_bse
            + offs_n * stride_bsn
            + group_k * stride_bsk,
            mask=offs_n < N,
            other=0,
        )
        b_scale = tl.exp2(b_scale_u8.to(tl.float32) - 127.0)
        accumulator += partial * a_scale[:, None] * b_scale[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(
        c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator.to(compute_type),
        mask=token_mask[:, None] & (offs_cn[None, :] < N),
    )


def _run_wfp4a8_groupwise_postscale_stage(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict,
    group_size: int,
    mul_routed_weight: bool,
    top_k_for_a: int,
    output_shape,
    compute_dtype,
) -> torch.Tensor:
    n = b_qweight.shape[1]
    k = a.shape[1]
    block_size_m = int(config["BLOCK_SIZE_M"])
    block_size_n = int(config.get("BLOCK_SIZE_N", 64))
    group_size_m = int(config.get("GROUP_SIZE_M", 8))

    if group_size != 32:
        raise ValueError("WFP4A8 groupwise Triton golden currently expects group_size=32")
    if a_scale.shape[-1] not in (1, triton.cdiv(k, group_size)):
        raise ValueError("WFP4A8 groupwise Triton golden expects per-token or per-token-group A scale")

    out = torch.empty(output_shape, device=a.device, dtype=compute_dtype)
    flat_out = out.view(-1, n)
    num_valid_tokens = int(topk_ids.numel())
    em = int(num_tokens_post_padded.item())
    grid = (
        triton.cdiv(em, block_size_m) * triton.cdiv(n, block_size_n),
    )
    _wfp4a8_groupwise_postscale_moe_gemm_kernel[grid](
        a,
        a_scale.contiguous(),
        b_qweight,
        flat_out,
        b_scale,
        topk_weights.reshape(-1).contiguous(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        em,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        a_scale.stride(0),
        a_scale.stride(1) if a_scale.ndim == 2 else 0,
        b_qweight.stride(0),
        b_qweight.stride(1),
        b_qweight.stride(2),
        flat_out.stride(0),
        flat_out.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        GROUP_SIZE_M=group_size_m,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        TOP_K=top_k_for_a,
        A_GROUP_SCALE=a_scale.shape[-1] != 1,
        compute_type=_torch_dtype_to_triton(compute_dtype),
    )
    return out


def run_wfp4a8_channelwise_triton_golden(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict,
    down_config: dict | None = None,
    output_dtype: torch.dtype = None,
    activation: str = "silu",
    routed_scaling_factor: float | None = 1.0,
    gemm1_alpha: float | None = None,
    gemm1_limit: float | None = None,
) -> torch.Tensor:
    """Triton golden for channelwise WFP4A8 with fp32 per-output scales."""
    if w1_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise ValueError("WFP4A8 channelwise Triton golden expects fp32 scales")

    compute_dtype = output_dtype or hidden_states.dtype
    hidden_states = hidden_states.contiguous()
    w1_qweight = w1_qweight.contiguous()
    w2_qweight = w2_qweight.contiguous()
    w1_scale = w1_scale.contiguous()
    w2_scale = w2_scale.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    sorted_token_ids = sorted_token_ids.contiguous()
    expert_ids = expert_ids.contiguous()
    num_tokens_post_padded = num_tokens_post_padded.contiguous()

    m = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    n1 = w1_qweight.shape[1]
    inter = n1 // 2
    hidden_out = w2_qweight.shape[1]
    total_tokens = m * topk

    qinput1, a1_scale = per_token_quant_hip(
        hidden_states, quant_dtype=torch.float8_e4m3fn
    )
    stage1 = _run_wfp4a8_stage_v2(
        qinput1,
        a1_scale,
        w1_qweight,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight=False,
        top_k_for_a=topk,
        output_shape=(total_tokens, n1),
        compute_dtype=compute_dtype,
    )

    activated = _apply_wfp4a16_torch_activation(
        stage1,
        inter,
        activation,
        compute_dtype,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
    )
    qinput2, a2_scale = per_token_quant_hip(
        activated.contiguous(), quant_dtype=torch.float8_e4m3fn
    )
    stage2 = _run_wfp4a8_stage_v2(
        qinput2,
        a2_scale,
        w2_qweight,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        down_config or config,
        mul_routed_weight=True,
        top_k_for_a=1,
        output_shape=(m, topk, hidden_out),
        compute_dtype=compute_dtype,
    )

    out = torch.empty((m, hidden_out), device=hidden_states.device, dtype=compute_dtype)
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    triton_moe_sum(stage2, out)
    if scale != 1.0:
        out = (out * scale).to(compute_dtype)
    return out


def run_wfp4a8_groupwise_triton_golden(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict,
    down_config: dict | None = None,
    group_size: int = 32,
    output_dtype: torch.dtype = None,
    activation: str = "silu",
    routed_scaling_factor: float | None = 1.0,
    gemm1_alpha: float | None = None,
    gemm1_limit: float | None = None,
    a_quant: str = "per_token_group",
) -> torch.Tensor:
    """Triton golden for groupwise WFP4A8 MoE with packed FP4 E2M1 weights.

    Weight and scale layout match WFP4A16/MXFP4:
      - qweight: uint8 packed E2M1 FP4, two values per byte
      - scale: uint8 e8m0, one scale per K group

    This mirrors the C++ WFP4A8 groupwise flow: packed FP4 E2M1 weights are
    used without pre-applying e8m0, then each K32 partial is post-scaled by
    activation scale and weight scale. Set a_quant="per_token_group" to match
    block_shape=[0, 32] activation quantization.
    """
    if w1_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise ValueError("WFP4A8 groupwise Triton golden expects uint8 e8m0 scales")

    compute_dtype = output_dtype or hidden_states.dtype
    hidden_states = hidden_states.contiguous()
    w1_qweight = w1_qweight.contiguous()
    w2_qweight = w2_qweight.contiguous()
    w1_scale = w1_scale.contiguous()
    w2_scale = w2_scale.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    sorted_token_ids = sorted_token_ids.contiguous()
    expert_ids = expert_ids.contiguous()
    num_tokens_post_padded = num_tokens_post_padded.contiguous()

    m = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    n1 = w1_qweight.shape[1]
    inter = n1 // 2
    hidden_out = w2_qweight.shape[1]
    total_tokens = m * topk

    if a_quant == "per_token_group":
        qinput1, a1_scale = per_token_group_quant_fp8(hidden_states, group_size)
    elif a_quant == "per_token":
        qinput1, a1_scale = per_token_quant_hip(
            hidden_states, quant_dtype=torch.float8_e4m3fn
        )
    else:
        raise ValueError(f"Unsupported WFP4A8 groupwise golden A quant: {a_quant}")
    stage1 = _run_wfp4a8_groupwise_postscale_stage(
        qinput1,
        a1_scale,
        w1_qweight,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        group_size,
        mul_routed_weight=False,
        top_k_for_a=topk,
        output_shape=(total_tokens, n1),
        compute_dtype=compute_dtype,
    )

    activated = _apply_wfp4a16_torch_activation(
        stage1,
        inter,
        activation,
        compute_dtype,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
    )
    if a_quant == "per_token_group":
        qinput2, a2_scale = per_token_group_quant_fp8(activated.contiguous(), group_size)
    else:
        qinput2, a2_scale = per_token_quant_hip(
            activated.contiguous(), quant_dtype=torch.float8_e4m3fn
        )
    stage2 = _run_wfp4a8_groupwise_postscale_stage(
        qinput2,
        a2_scale,
        w2_qweight,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        down_config or config,
        group_size,
        mul_routed_weight=True,
        top_k_for_a=1,
        output_shape=(m, topk, hidden_out),
        compute_dtype=compute_dtype,
    )

    out = torch.empty((m, hidden_out), device=hidden_states.device, dtype=compute_dtype)
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    triton_moe_sum(stage2, out)
    if scale != 1.0:
        out = (out * scale).to(compute_dtype)
    return out


run_wfp4a8_triton_golden = run_wfp4a8_groupwise_triton_golden


@triton.jit
def _fp8_channelwise_stage1_kernel(
    hidden_ptr,
    w1_ptr,
    topk_ids_ptr,
    a_scale_ptr,
    w1_scale_ptr,
    stage1_ptr,
    K: tl.constexpr,
    N1: tl.constexpr,
    TOPK: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_w1e: tl.constexpr,
    stride_w1n: tl.constexpr,
    stride_w1k: tl.constexpr,
    stride_w1se: tl.constexpr,
    stride_w1sn: tl.constexpr,
    stride_s1m: tl.constexpr,
    stride_s1n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    token_id = route_id // TOPK
    expert_id = tl.load(topk_ids_ptr + route_id).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        a = tl.load(
            hidden_ptr + token_id * stride_hm + k_idxs * stride_hk,
            mask=k_idxs < K,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w1_ptr + expert_id * stride_w1e + offs_n[:, None] * stride_w1n + k_idxs[None, :] * stride_w1k,
            mask=(offs_n[:, None] < N1) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(b * a[None, :], axis=1)

    a_scale = tl.load(a_scale_ptr + token_id).to(tl.float32)
    w_scale = tl.load(
        w1_scale_ptr + expert_id * stride_w1se + offs_n * stride_w1sn,
        mask=offs_n < N1,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        stage1_ptr + route_id * stride_s1m + offs_n * stride_s1n,
        acc * a_scale * w_scale,
        mask=offs_n < N1,
    )


@triton.jit
def _fp8_channelwise_act_quant_kernel(
    stage1_ptr,
    bridge_ptr,
    bridge_scale_ptr,
    N: tl.constexpr,
    stride_s1m: tl.constexpr,
    stride_s1n: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    route_id = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    gate = tl.load(
        stage1_ptr + route_id * stride_s1m + offs_n * stride_s1n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        stage1_ptr + route_id * stride_s1m + (offs_n + N) * stride_s1n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    act = (gate / (1.0 + tl.exp(-gate))) * up
    max_abs = tl.max(tl.abs(act), axis=0)
    scale = max_abs / FP8_MAX
    scale = tl.where(scale == 0.0, 1.0, scale)
    tl.store(bridge_scale_ptr + route_id, scale)
    tl.store(
        bridge_ptr + route_id * stride_bm + offs_n * stride_bn,
        act / scale,
        mask=mask,
    )


@triton.jit
def _fp8_channelwise_stage2_routes_kernel(
    bridge_ptr,
    bridge_scale_ptr,
    w2_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    w2_scale_ptr,
    route_out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_w2e: tl.constexpr,
    stride_w2k: tl.constexpr,
    stride_w2n: tl.constexpr,
    stride_w2se: tl.constexpr,
    stride_w2sk: tl.constexpr,
    stride_rom: tl.constexpr,
    stride_rot: tl.constexpr,
    stride_rok: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    route_id = tl.program_id(0)
    pid_k = tl.program_id(1)
    token_id = route_id // TOPK
    topk_lane = route_id - token_id * TOPK
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    expert_id = tl.load(topk_ids_ptr + route_id).to(tl.int64)
    route_weight = tl.load(topk_weights_ptr + route_id).to(tl.float32)
    bridge_scale = tl.load(bridge_scale_ptr + route_id).to(tl.float32)
    route_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for n0 in range(0, N, BLOCK_N):
        n_idxs = n0 + offs_n
        act = tl.load(
            bridge_ptr + route_id * stride_bm + n_idxs * stride_bn,
            mask=n_idxs < N,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w2_ptr + expert_id * stride_w2e + offs_k[:, None] * stride_w2k + n_idxs[None, :] * stride_w2n,
            mask=(offs_k[:, None] < K) & (n_idxs[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        route_acc += tl.sum(b * act[None, :], axis=1)

    w_scale = tl.load(
        w2_scale_ptr + expert_id * stride_w2se + offs_k * stride_w2sk,
        mask=offs_k < K,
        other=0.0,
    ).to(tl.float32)
    route_out = route_acc * bridge_scale * w_scale * route_weight

    tl.store(
        route_out_ptr + token_id * stride_rom + topk_lane * stride_rot + offs_k * stride_rok,
        route_out,
        mask=offs_k < K,
    )


def _run_fp8_channelwise_golden_impl(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    out_dtype,
    w1_scale,
    w2_scale,
    a1_scale,
    global_num_experts,
):
    del global_num_experts
    hidden_states = hidden_states.contiguous()
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    if a1_scale is None:
        hidden_states, a1_scale = pertoken_quant(hidden_states, quant_dtype=w1.dtype)
    else:
        hidden_states = hidden_states.to(w1.dtype)
    hidden_states = hidden_states.contiguous()
    a1_scale = a1_scale.reshape(-1).contiguous()
    w1_scale = w1_scale.contiguous()
    w2_scale = w2_scale.contiguous()

    m, k = hidden_states.shape
    topk = topk_ids.shape[1]
    n = w2.shape[2]
    n1 = w1.shape[1]
    assert n1 == 2 * n

    stage1 = torch.empty((m * topk, n1), device=hidden_states.device, dtype=out_dtype)
    out = torch.empty((m, k), device=hidden_states.device, dtype=out_dtype)
    block_n = 32
    block_k = 64

    _fp8_channelwise_stage1_kernel[(m * topk, triton.cdiv(n1, block_n))](
        hidden_states,
        w1,
        topk_ids,
        a1_scale,
        w1_scale,
        stage1,
        K=k,
        N1=n1,
        TOPK=topk,
        stride_hm=hidden_states.stride(0),
        stride_hk=hidden_states.stride(1),
        stride_w1e=w1.stride(0),
        stride_w1n=w1.stride(1),
        stride_w1k=w1.stride(2),
        stride_w1se=w1_scale.stride(0),
        stride_w1sn=w1_scale.stride(1),
        stride_s1m=stage1.stride(0),
        stride_s1n=stage1.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    activated = torch.empty((m * topk, n), device=hidden_states.device, dtype=out_dtype)
    _apply_activation(
        activation="silu",
        is_gated=True,
        activated_out=activated,
        ffn1_out_2d=stage1,
        gemm1_alpha=None,
        gemm1_limit=None,
    )
    bridge, bridge_scale = per_token_quant_hip(activated, quant_dtype=torch.float8_e4m3fn)

    route_out = torch.empty((m, topk, k), device=hidden_states.device, dtype=out_dtype)
    _fp8_channelwise_stage2_routes_kernel[(m * topk, triton.cdiv(k, block_k))](
        bridge,
        bridge_scale,
        w2,
        topk_weights,
        topk_ids,
        w2_scale,
        route_out,
        K=k,
        N=n,
        TOPK=topk,
        stride_bm=bridge.stride(0),
        stride_bn=bridge.stride(1),
        stride_w2e=w2.stride(0),
        stride_w2k=w2.stride(1),
        stride_w2n=w2.stride(2),
        stride_w2se=w2_scale.stride(0),
        stride_w2sk=w2_scale.stride(1),
        stride_rom=route_out.stride(0),
        stride_rot=route_out.stride(1),
        stride_rok=route_out.stride(2),
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )

    triton_moe_sum(route_out, out, routed_scaling_factor=1.0)
    return out


def run_fp8_channelwise_golden(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    out_dtype,
    w1_scale,
    w2_scale,
    a1_scale,
    global_num_experts,
):
    m = hidden_states.shape[0]
    # The route-wise Triton stage2 launch is (m * topk, cdiv(k, 64)).
    # Very large token counts can exceed practical ROCm launch/memory limits
    # for the golden path, even though the ASM path itself is fine.
    chunk_m = 8192
    if m <= chunk_m:
        return _run_fp8_channelwise_golden_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            out_dtype,
            w1_scale,
            w2_scale,
            a1_scale,
            global_num_experts,
        )

    outs = []
    a1_scale_flat = None if a1_scale is None else a1_scale.reshape(-1)
    for start in range(0, m, chunk_m):
        end = min(start + chunk_m, m)
        chunk_a1_scale = None
        if a1_scale_flat is not None:
            chunk_a1_scale = a1_scale_flat[start:end].contiguous()
        outs.append(
            _run_fp8_channelwise_golden_impl(
                hidden_states[start:end].contiguous(),
                w1,
                w2,
                topk_weights[start:end].contiguous(),
                topk_ids[start:end].contiguous(),
                out_dtype,
                w1_scale,
                w2_scale,
                chunk_a1_scale,
                global_num_experts,
            )
        )
    return torch.cat(outs, dim=0)
