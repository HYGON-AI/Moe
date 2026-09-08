# Test for local get_moe_config and moe with W4A8.

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe_utils.golden import run_w4a8_perchannel_golden
from moe_utils.utils import (
    ActivationType,
    checkAllclose,
    compare_tensors,
    dtypes,
    fused_topk,
    logger,
    moe_kernel_prepare_input,
    perftest,
    torch_moe,
)
from moe_utils.moe_ops import moe_align_block_size, moe_sum_opt_v2, silu_and_mul
from moe_utils.moe import get_moe_config, moe, moe_shfl_weight, MoeSolutionType, MoeQuantType
import torch.nn.functional as F
import triton.language as tl
import triton
from typing import  List, Optional,  Type


torch.set_default_device("cuda")


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    delta: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    """
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    if GROUP_SIZE_M ==1:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m // delta).to(tl.int64)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]

        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel_int8(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: Optional[torch.Tensor],
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: Optional[List[int]] = None,
                            use_nn_moe: Optional[bool]=False) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w8a8 is True

    M = A.shape[0]
    num_tokens = M * top_k

    triton_config = {   "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "kpack": 1,
        # "num_stages": 0, #triton后端更新 =0时错误改为等于2
        "num_stages": 2, #triton后端更新 =0时错误改为等于2
        "num_warps": 2
    }
    triton_config["BLOCK_SIZE_M"] = 16

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < triton_config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * triton_config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1] if not use_nn_moe else B.shape[2], META['BLOCK_SIZE_N']), )

    fused_moe_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1] if not use_nn_moe else B.shape[2],
            A.shape[1],
            EM,
            num_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2) if not use_nn_moe else B.stride(1),
            B.stride(1) if not use_nn_moe else B.stride(2),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2)
            if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            delta=1,
            **triton_config
        )

def perchannel_w8a8_triton(input,
        weight1,
        weight2,
        weight_scale1,
        weight_scale2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        out_dtype,
        per_channel_quant
        ):

    if out_dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif out_dtype == torch.float16:
        compute_type = tl.float16
    elif out_dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {out_dtype}")

    n1 = weight1.shape[1]
    top_k = topk_weights.shape[1]
    m = input.shape[0]


    output_shape = (m, top_k, n1)
    output_triton = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
    output_shape2 = (m  , top_k, weight2.shape[1])
    output_triton2 = torch.zeros(output_shape2, device=input.device, dtype=input.dtype)
    input_gemm2 = torch.empty((m * top_k, n1 // 2), device=input.device, dtype=input.dtype)


    qinput1, qa_scale1 = moe_kernel_prepare_input(
            A=input,
            B=weight1,
            A_scale=None,
            B_scale=weight_scale1,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=per_channel_quant,
            block_shape=None
        )

    invoke_fused_moe_kernel_int8(
        qinput1,
        weight1,
        output_triton,
        qa_scale1,
        weight_scale1,
        None,  # B_zp
        topk_weights,
        topk_ids,  # topk_ids (不需要)
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # mul_routed_weight
        top_k ,
        compute_type,
        False,  # use_fp8_w8a8
        True,   # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        per_channel_quant,
        None,   # block_shape
        False   # use_nn_moe
    )

    # print("*******************************************************output_triton")
    # print(output_triton)


    silu_and_mul(input_gemm2, output_triton.view(-1, n1))
    # print(input_gemm2)




    # input_gemm2 = torch.ones_like(input_gemm2)
    qinput2, qa_scale2 = moe_kernel_prepare_input(
            A=input_gemm2,
            B=weight2,
            A_scale=None,
            B_scale=weight_scale2,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=per_channel_quant,
            block_shape=None
        )

    # print("golden 量化后：")
    # print(qinput2)


    # print("**********************************************qinput2")
    # print(qinput2)

    invoke_fused_moe_kernel_int8(
        qinput2,
        weight2,
        output_triton2,
        qa_scale2,
        weight_scale2,
        None,  # B_zp
        topk_weights,
        topk_ids,  # topk_ids (不需要)
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,  # mul_routed_weight
        1,
        compute_type,
        False,  # use_fp8_w8a8
        True,   # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        per_channel_quant,
        None,   # block_shape
        False   # use_nn_moe
    )




    out_hidden_states = torch.empty_like(input)
    moe_sum_opt_v2(output_triton2.view(*output_triton2.shape), out_hidden_states, 1.0)
    # print("**************************************triton")
    # print(out_hidden_states)

    return out_hidden_states


@perftest(num_warmup=1, num_iters=2)
def _run_torch_ref(hidden_states, w1, w2, topk_weights, topk_ids, dtype, block_shape, w1_scale, w2_scale):

    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        None,
        None,
        None,
        ActivationType.Silu,
    )

def _run_triton_ref(hidden_states, w1, w2, topk_weights, topk_ids, dtype, block_shape, w1_scale, w2_scale,E):

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(topk_ids, 16, E, None)
        )
        return perchannel_w8a8_triton(hidden_states,w1,w2,w1_scale,w2_scale,topk_weights, topk_ids,sorted_token_ids, expert_ids, num_tokens_post_padded,dtype,True)

@perftest(num_warmup=10, num_iters=100, num_rotate_args=1,testGraph = True)
def _run_moe_perf(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    moe_config,
    inplace,
    activation,
    w1_scale,
    w2_scale,
    w1_zp,
    w2_zp,
    a1_scale,
    a2_scale,
    block_shape,
    global_num_experts,
    expert_map,
):

    if inplace:
        mortal_input = hidden_states.clone()  # 保证inplace操作的正确性
    else:
        mortal_input = hidden_states

    return moe(
        mortal_input,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_config,
        inplace,
        activation,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        global_num_experts,
        expert_map,
    )
def generate_unique_int_tensor(m, topk, low=0, high=256):
    """生成[m, topk]的整数张量，每行元素不重复"""
    if topk > (high - low):
        raise ValueError(f"topk ({topk}) 不能超过范围 ({high - low})")

    tensor = torch.zeros((m, topk), dtype=torch.int)
    for i in range(m):
        row = torch.randperm(high - low)[:topk] + low
        tensor[i] = row
    return tensor


def pack_int4_on_k(weight):
    # 假设 weight 形状为 (E, n2, k2)
    E, n2, k2 = weight.shape
    assert k2 % 2 == 0, "K 维度必须是偶数才能成对打包"


    w_unsigned = weight.to(torch.uint8) & 0x0F


    original_shape = w_unsigned.shape
    temp_view = w_unsigned.view(*original_shape[:-1], -1, 8)
    selected_low = temp_view[..., :4]
    w_low = selected_low.reshape(*original_shape[:-1], -1)
    selected_high = temp_view[..., 4:]
    w_high = selected_high.reshape(*original_shape[:-1], -1)




    packed_weight = (w_low << 4) | w_high

    # 5. 转回 int8 存储
    return packed_weight.to(torch.int8)

def pack_weight_k_contiguous(weight):
    # weight shape: [E, n2, k2]
    E, n2, k2 = weight.shape
    assert k2 % 2 == 0, "K 维度必须是偶数才能成对打包"

    # 1. 确保数据在 [0, 15] 范围内（4-bit）
    w_unsigned = weight.to(torch.uint8) & 0x0F

    # 2. 将 K 维度拆解为 [k2//2, 2]
    w_reshaped = w_unsigned.view(E, n2, k2 // 2, 2)

    # 3. 提取相邻的两个元素
    # w_low 取第 0 个元素，w_high 取第 1 个元素
    w_0 = w_reshaped[..., 0]
    w_1 = w_reshaped[..., 1]

    # 4. 执行打包逻辑 (K_even << 4 | K_odd)
    packed_weight = (w_0 << 4) | w_1

    # 5. 转回 int8 存储，最终 shape 为 [E, n2, k2 // 2]
    return packed_weight.to(torch.int8)


def prepare_w4a8_inputs(m, k, n, E, topk,  out_dtype, device: str = "cuda", per_channel_quant: bool = True):

    torch.manual_seed(0)

    n2 = k
    k2 = n
    n = 2 * n
    k = k



      # """生成测试数据"""
    #输入初始化
    input_data = (torch.rand((m, k), device=device) ).to(dtype=out_dtype)  / 10000

    #权重初始化
    #weight1
    weight1_ori = torch.randint(0,15,(E, n, k ), device=device, dtype=torch.int8)
    ref_weight1 = torch.where(weight1_ori > 7, weight1_ori - 16, weight1_ori)

    #weight2
    weight2_ori = torch.randint(0,15,(E, n2, k2 ), device=device, dtype=torch.int8)
    ref_weight2 = torch.where(weight2_ori > 7, weight2_ori - 16, weight2_ori)


    #对齐正常模型pack后的权重shape
    weight1_packed = pack_weight_k_contiguous(weight1_ori)
    weight2_packed = pack_weight_k_contiguous(weight2_ori)


    if per_channel_quant:
        weight1_scale = torch.randn((E, n, 1), dtype=torch.float32, device=device)
        weight2_scale = torch.randn((E, n2, 1), dtype=torch.float32, device=device)
    else:
        # 这里可以添加block-wise量化的支持
        weight1_scale = torch.randn((E, n, 1), device=device, dtype=torch.float32)
        weight2_scale = torch.randn((E, n2, 1), device=device, dtype=torch.float32)


    tensor = torch.rand((m, topk), device=device)
    topk_weights = F.softmax(tensor, dim=1)
    topk_ids = generate_unique_int_tensor(m=m, topk=topk, low=0, high=E).cuda()

    return {
        "input": input_data,
        "w1_ref": ref_weight1,
        "w2_ref": ref_weight2,
        "w1_qweight": weight1_packed,
        "w2_qweight": weight2_packed,
        "w1_scales": weight1_scale,
        "w2_scales": weight2_scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def test_get_config(m, k, n, e, topk, dtype):



    status, moe_cfg = get_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=None,
        dtype=dtype,
        quant_type=MoeQuantType.W4A8,
    )


    if status:
        assert moe_cfg.quant_type == MoeQuantType.W4A8
        assert moe_cfg.solution_type in (
            MoeSolutionType.MOE_C,
        )
        assert moe_cfg.config is not None
        logger.info(
            f"[get_moe_config_w4a8] {m=}, solution={moe_cfg.solution_type}, config keys={list(moe_cfg.config.keys())}"
        )
    else:
        assert moe_cfg.solution_type is None
        assert moe_cfg.config is None
        logger.info(f"[get_moe_config_w4a8] {m=}, no solution found")

    return status, moe_cfg


def test_moe_w4a8(m, k, n, e, topk, dtype, show_compare=False):

    status, moe_cfg = get_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=None,
        dtype=dtype,
        quant_type=MoeQuantType.W4A8,
    )

    data = prepare_w4a8_inputs(m, k, n, e, topk, dtype)
    ref_out = run_w4a8_perchannel_golden(
        data["input"],
        data["w1_ref"],
        data["w2_ref"],
        data["topk_weights"],
        data["topk_ids"],
        dtype,
        data["w1_scales"],
        data["w2_scales"],
    )

    ret = {
        "m": m,
        "moe_backend": moe_cfg.solution_type if status else None,
        "moe_us": None,
    }
    moe_us = 1.0
    if not status:
        logger.info(f"[moe_w4a8] SKIP {m=}: no backend available")
        return ret

    w1_run, w2_run = data["w1_qweight"], data["w2_qweight"]
    if moe_cfg.need_shuffle:
        w1_run, w2_run = moe_shfl_weight(w1_run, w2_run, moe_cfg)

    moe_out, moe_us = _run_moe_perf(
        hidden_states=data["input"],
        w1=w1_run,
        w2=w2_run,
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        moe_config=moe_cfg,
        inplace=False,
        activation="silu",
        w1_scale=data["w1_scales"] / 16,
        w2_scale=data["w2_scales"] / 16,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        global_num_experts=e,
        expert_map=None,
    )

    msg = f"[moe_w4a8] {m=}, backend={moe_cfg.solution_type}"
  
    assert torch.isfinite(moe_out).all(), (
        "Non-finite output in test_moe_w4a8_perchannel: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {dtype=}, "
        f"backend={moe_cfg.solution_type}"
    )
    check_ret = checkAllclose(ref_out, moe_out, rtol=0.01, atol=0.2, msg=msg)
    if show_compare:
        print(f"\ncompare_tensors output for m={m}", flush=True)
        compare_tensors(ref_out, moe_out, rtol=0.01, atol=0.2)
    assert check_ret <= 0.03, (
        "Accuracy check failed in test_moe_w4a8_perchannel: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {dtype=}, "
        f"backend={moe_cfg.solution_type}, need_shuffle={moe_cfg.need_shuffle}, "
        f"error_ratio={check_ret:.6f}, tolerance=0.03"
    )
    ret["moe_accuracy"] = "passed" if check_ret == 0 else (1-check_ret)
    ret["moe_us"] = moe_us
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=None)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    failed_cases = []

    def run_accuracy_case(test_func, *args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except AssertionError as exc:
            failed_cases.append(str(exc))
            print(f"[ACCURACY FAILED] {exc}", flush=True)
            return None
    dtype = dtypes.bf16


    e = 256
    topk = 8
    k = 7168
    n = 256

    logger.info("=" * 60)
    logger.info("Part 1: Testing get_moe_config for w4a8")
    logger.info("=" * 60)
    test_tokens = args.tokens or [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,6144,8192,16384]
    # test_tokens = [4096]

    for m in test_tokens:
        test_get_config(m, k, n, e, topk,  dtype)

    logger.info("=" * 60)
    logger.info("Part 2: Testing moe end-to-end for w4a8")
    logger.info("=" * 60)
    df = []
    for m in test_tokens:
        ret = run_accuracy_case(
            test_moe_w4a8, m, k, n, e, topk, dtype, args.compare
        )
        if ret is not None:
            df.append(ret)
    if df:
        df = pd.DataFrame(df)
        logger.info(f"summary(端到端耗时):\n{df}")
        df.to_csv(f"w4a8_perchannel_TP{2048 // n}.csv", index=False)
    
        logger.info(f"CSV 文件已保存至: w4a8_perchannel_TP{2048 // n}.csv")

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
