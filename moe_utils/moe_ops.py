# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Python wrappers for MOE_C torch.library operators."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


_MOE_C_LOADED = False


def load_moe_c_library() -> None:
    global _MOE_C_LOADED
    if _MOE_C_LOADED:
        return

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = Path(os.environ.get("MOE_C_BUILD_DIR", repo_root / "build"))
    split_libs = (
        "libmoe_c_common.so",
        "libmoe_w8a8_int8.so",
        "libmoe_w8a8_fp8.so",
        "libmoe_w4a8.so",
        "libmoe_w4a16.so",
        "libmoe_w8a16.so",
        "libmoe_wna16_only.so",
        "libmoe_wfp4a16_groupwise.so",
        "libmoe_wfp4a8_channelwise.so",
        "libmoe_wfp4a8_groupwise.so",
    )

    loaded = False
    for lib_name in split_libs:
        lib_path = build_dir / lib_name
        if lib_path.exists():
            torch.ops.load_library(str(lib_path))
            loaded = True

    if not loaded:
        raise OSError(
            "Could not find MOE build artifacts. Run build.sh first, or set "
            f"MOE_C_BUILD_DIR. Looked under: {build_dir}"
        )
    _MOE_C_LOADED = True


def _call_moe_wna16_op(op_name: str, *args, **kwargs):
    load_moe_c_library()
    try:
        op = getattr(torch.ops.moe_wna16, op_name)
    except AttributeError as exc:
        raise NotImplementedError(
            f"MOE op '{op_name}' is not registered by this repository build."
        ) from exc
    return op(*args, **kwargs)


def _call_marlin_op(op_name: str, *args):
    return _call_moe_wna16_op(op_name, *args)


def moe_sum(input: torch.Tensor, output: torch.Tensor, topk_ids: torch.Tensor) -> None:
    _call_moe_wna16_op("moe_sum", input, output, topk_ids)


def moe_sum_opt_v2(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    return _call_moe_wna16_op("moe_sum_opt_v2", input, output, routed_scaling_factor)


def silu_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
    rows_per_block: int = 1,
    vec_size: int = 2,
) -> None:
    _call_moe_wna16_op("silu_and_mul", out, input, rows_per_block, vec_size)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = True,
) -> None:
    load_moe_c_library()
    try:
        torch.ops.moe_wna16.topk_softmax(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )
    except RuntimeError:
        torch.ops.moe_wna16.topk_softmax(
            topk_weights, topk_ids, token_expert_indices, gating_output
        )


def moe_align_block_size(topk_ids: torch.Tensor, *args):
    if len(args) == 3:
        block_size, num_experts, _expert_map = args
        max_num_tokens_padded = topk_ids.numel() + int(num_experts) * (int(block_size) - 1)
        sorted_token_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
        sorted_token_ids.fill_(topk_ids.numel())
        max_num_m_blocks = (max_num_tokens_padded + int(block_size) - 1) // int(block_size)
        expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
        num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    elif len(args) == 5:
        num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad = args
    else:
        raise TypeError(
            "moe_align_block_size expects either "
            "(topk_ids, block_size, num_experts, expert_map) or "
            "(topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad)"
        )

    op_name = "sgl_moe_align_block_size" if int(num_experts) == 256 else "moe_align_block_size"
    _call_moe_wna16_op(
        op_name,
        topk_ids,
        int(num_experts),
        int(block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    if len(args) == 3:
        return sorted_token_ids, expert_ids, num_tokens_post_pad
    return None


def moe_c_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> None:
    topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output)


def moe_c_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    _call_moe_wna16_op(
        "moe_align_block_size",
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )


def moe_c_sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    _call_moe_wna16_op(
        "sgl_moe_align_block_size",
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )


moe_c_moe_sum = moe_sum
moe_c_moe_sum_opt_v2 = moe_sum_opt_v2
moe_c_silu_and_mul = silu_and_mul


def moe_c_moe_w8a16_gemm_block_wise(*args):
    return _call_moe_wna16_op("moe_c_moe_w8a16_gemm_block_wise", *args)


def moe_c_moe_w8a16_gemm_awq(*args):
    return _call_moe_wna16_op("moe_c_moe_w8a16_gemm_awq", *args)


def moe_c_moe_wna16_gemm(*args):
    return _call_moe_wna16_op("moe_c_moe_wna16_gemm", *args)


def moe_c_moe_wna16_gemm_2(*args):
    return _call_moe_wna16_op("moe_c_moe_wna16_gemm_2", *args)


def moe_c_moe_wna16_gemm_base(*args):
    return _call_moe_wna16_op("moe_c_moe_wna16_gemm_base", *args)


def moe_c_moe_w8a8_gemm_block_wise(*args):
    return _call_moe_wna16_op("moe_c_moe_w8a8_gemm_block_wise", *args)


def moe_c_moe_w8a8_gemm_block_wise_kernel2(*args):
    return _call_moe_wna16_op("moe_c_moe_w8a8_gemm_block_wise_kernel2", *args)


def moe_c_moe_w8a8_gemm_block_wise_fp8(*args):
    return _call_moe_wna16_op("moe_c_moe_w8a8_gemm_block_wise_fp8", *args)


def moe_c_moe_w8a8_gemm_block_wise_kernel2_fp8(*args):
    return _call_moe_wna16_op("moe_c_moe_w8a8_gemm_block_wise_kernel2_fp8", *args)


def moe_c_moe_gemm_marlin_w16a16(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w16a16", *args)


def moe_c_moe_gemm_marlin_w16a16_asm(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w16a16_asm", *args)


def moe_c_moe_gemm_marlin_w8a16(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w8a16", *args)


def moe_c_moe_gemm_marlin_w4a16(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w4a16", *args)


def moe_c_moe_gemm_marlin_wfp4a16(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_wfp4a16", *args)


def moe_c_moe_gemm_marlin_w8a8(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w8a8", *args[:13])


def moe_c_moe_gemm_marlin_w8a8_tensorwise(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w8a8_tensorwise", *args[:13])


def moe_c_moe_gemm_marlin_w4a8(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w4a8", *args[:13])


def moe_c_moe_gemm_marlin_w8a8_fp8(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w8a8_fp8", *args[:15])


def moe_c_moe_gemm_marlin_w8a8_fp8_tensorwise(*args):
    return _call_marlin_op("moe_c_moe_gemm_marlin_w8a8_fp8_tensorwise", *args[:15])


def moe_c_moe_gemm_marlin_wfp4a8_channelwise(*args):
    normalized_args = args[:12] + args[14:15] if len(args) >= 15 else args[:13]
    return _call_marlin_op("moe_c_moe_gemm_marlin_wfp4a8_channelwise", *normalized_args)


def moe_c_moe_gemm_marlin_wfp4a8_groupwise(*args):
    normalized_args = args[:12] + args[14:15] if len(args) >= 15 else args[:13]
    return _call_marlin_op("moe_c_moe_gemm_marlin_wfp4a8_groupwise", *normalized_args)


def moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup(*args):
    return moe_c_moe_gemm_marlin_wfp4a8_groupwise(*args)


def moe_sorting_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    tokens_positions_per_expert: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    block_size: int,
    expert_mask: Optional[torch.Tensor] = None,
) -> None:
    from .ck_sorting_sim import simulate_ck_moe_sorting

    sorting = simulate_ck_moe_sorting(
        topk_ids,
        topk_weights,
        int(num_experts),
        int(block_size),
        expert_mask,
    )
    sorted_ids.copy_(sorting.sorted_token_ids.to(sorted_ids.dtype))
    sorted_weights.copy_(sorting.sorted_weights.to(sorted_weights.dtype))
    sorted_expert_ids.copy_(sorting.sorted_expert_ids.to(sorted_expert_ids.dtype))
    tokens_positions_per_expert.copy_(
        sorting.tokens_positions_per_expert.to(tokens_positions_per_expert.dtype)
    )
    num_valid_ids.copy_(sorting.num_valid_ids.to(num_valid_ids.dtype))
    moe_buf.zero_()


def moe_sorting_ck_ids_to_moe_c(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "moe_sorting_ck_ids_to_moe_c is an AITER sorting helper and is not "
        "implemented in the standalone MOE repository."
    )


def moe_c_situ_glu(
    out: torch.Tensor,
    input: torch.Tensor,
    beta1: float = 4.0,
    beta2: float = 25.0,
    rows_per_block: int = 1,
    vec_size: int = 2,
) -> None:
    del rows_per_block, vec_size
    gate, up = input.chunk(2, dim=-1)
    gate_f = gate.to(torch.float32)
    up_f = up.to(torch.float32)
    value = beta1 * torch.tanh(gate_f / beta1) * torch.sigmoid(gate_f)
    value = value * (beta2 * torch.tanh(up_f / beta2))
    out.copy_(value.to(out.dtype))
