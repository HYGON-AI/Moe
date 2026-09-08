# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager

import time
import torch
import triton
import triton.language as tl
import statistics
import logging

import time
from .utils import (
    _apply_activation,
    _normalize_activation_and_gate,
    dtypes,
    per_token_quant_hip,
    perftest,
    torch_compile_guard,
)
from .moe_ops import (
    moe_c_moe_align_block_size,
    moe_c_moe_gemm_marlin_w4a8,
    moe_c_moe_gemm_marlin_w4a16,
    moe_c_moe_gemm_marlin_w8a8,
    moe_c_moe_gemm_marlin_w8a8_fp8,
    moe_c_moe_gemm_marlin_w8a8_fp8_tensorwise,
    moe_c_moe_gemm_marlin_w8a8_tensorwise,
    moe_c_moe_gemm_marlin_w8a16,
    moe_c_moe_gemm_marlin_w16a16,
    moe_c_moe_gemm_marlin_w16a16_asm,
    moe_c_moe_gemm_marlin_wfp4a8_channelwise,
    moe_c_moe_gemm_marlin_wfp4a8_groupwise,
    moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup,
    moe_c_moe_gemm_marlin_wfp4a16,
    moe_c_moe_sum,
    moe_c_moe_sum_opt_v2,
    moe_c_moe_w8a8_gemm_block_wise,
    moe_c_moe_w8a8_gemm_block_wise_fp8,
    moe_c_moe_w8a8_gemm_block_wise_kernel2,
    moe_c_moe_w8a8_gemm_block_wise_kernel2_fp8,
    moe_c_moe_w8a16_gemm_awq,
    moe_c_moe_w8a16_gemm_block_wise,
    moe_c_moe_wna16_gemm,
    moe_c_moe_wna16_gemm_2,
    moe_c_moe_wna16_gemm_base,
    moe_c_sgl_moe_align_block_size,
    moe_c_silu_and_mul,
    moe_c_situ_glu,
    moe_c_topk_softmax,
    moe_sorting_ck_ids_to_moe_c,
    moe_sorting_fwd,
)
from triton.language.extra import libdevice


logger = logging.getLogger(__name__)

_MOEC_WFP4_GROUP_SIZE = 32


def _check_moec_quant_legal(
    use_int4_w4a16: bool = False,
    use_fp4_w4a16: bool = False,
    use_int8_w4a8: bool = False,
    use_fp4_w4a8: bool = False,
    block_shape: Optional[List[int]] = None,
) -> bool:
    use_fp4_w4a8_groupwise = False

    if use_int4_w4a16:
        assert block_shape is not None, (
            "W4A16 moe_c currently requires groupwise block_shape=[0, 32]."
        )
        assert len(block_shape) == 2, (
            f"W4A16 moe_c currently only supports block_shape=[0, 32], got {block_shape}."
        )
        block_n, block_k = block_shape
        assert block_n == 0 and block_k == _MOEC_WFP4_GROUP_SIZE, (
            f"W4A16 moe_c currently only supports group_size=32 "
            f"(expected block_shape=[0, 32], got block_shape={block_shape})."
        )

    if use_fp4_w4a16:
        assert block_shape is not None, (
            "WFP4A16 moe_c currently requires groupwise block_shape=[0, 32]."
        )
        assert len(block_shape) == 2, (
            f"WFP4A16 moe_c currently only supports block_shape=[0, 32], got {block_shape}."
        )
        block_n, block_k = block_shape
        assert block_n == 0 and block_k == _MOEC_WFP4_GROUP_SIZE, (
            f"WFP4A16 moe_c currently only supports group_size=32 "
            f"(expected block_shape=[0, 32], got block_shape={block_shape})."
        )

    if use_int8_w4a8:
        assert block_shape is None, (
            f"W4A8 moe_c currently only supports channelwise block_shape=None, got {block_shape}."
        )

    if use_fp4_w4a8:
        if block_shape is None:
            return use_fp4_w4a8_groupwise
        assert len(block_shape) == 2, (
            f"WFP4A8 moe_c currently only supports block_shape=None or [0, 32], got {block_shape}."
        )
        block_n, block_k = block_shape
        assert block_n == 0 and block_k == _MOEC_WFP4_GROUP_SIZE, (
            f"WFP4A8 moe_c currently only supports channelwise or group_size=32 "
            f"(expected block_shape=None or [0, 32], got block_shape={block_shape})."
        )
        use_fp4_w4a8_groupwise = True

    return use_fp4_w4a8_groupwise

if not logger.handlers:
    # 设置日志级别（DEBUG/INFO/WARNING/ERROR）
    logger.setLevel(logging.INFO)
    # 定义日志格式（包含时间、模块、级别、内容）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 添加控制台输出 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

_global_config: Optional[Dict] = None

@contextmanager
def override_config(config: Dict):
    global _global_config
    old_config = _global_config  # 保存当前配置
    _global_config = config      # 应用新配置
    yield                       # 执行上下文内的代码
    _global_config = old_config  # 退出上下文时恢复原配置

def get_config() -> Optional[Dict]:
    return _global_config

def scaled_fp8_quant(A: torch.tensor, A_scale: torch.tensor):  # todo
    pass

def get_device_name():
    if torch.cuda.is_available():
        
        device_name = torch.cuda.get_device_name(0)
    else:
        
        device_name = "cpu"
    
    return device_name.replace(" ", "_")


#int8 per token量化
@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols,
                mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    # x_q = tl.extra.cuda.libdevice.nearbyint(x_q).to(tl.int8)
    x_q = libdevice.nearbyint(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,),
                         device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    assert x.is_contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    return x_q, scales

# ck_sorting算子
def moe_sorting_ck(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=32,
    expert_mask=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=dtypes.i32, device=device
    )
    tokens_positions_per_expert = torch.empty(
        (num_experts*2,), dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty((1), dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

# for now, moe_sorting_fwd only support int32 topk_ids
    if topk_ids.dtype != dtypes.i32:
        topk_ids = topk_ids.to(dtypes.i32)

    moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        tokens_positions_per_expert,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size,
        expert_mask,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, tokens_positions_per_expert, moe_buf


def moe_align_block_size_w16a16_ck(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = topk_ids.device
    _, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=dtypes.i32, device=device
    )
    tokens_positions_per_expert = torch.empty(
        (num_experts * 2,), dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty((1,), dtype=dtypes.i32, device=device)

    if topk_ids.dtype != dtypes.i32:
        topk_ids = topk_ids.to(dtypes.i32)

    expert_mask = (expert_map >= 0).to(dtypes.i32) if expert_map is not None else None
    moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        tokens_positions_per_expert,
        num_valid_ids,
        None,
        num_experts,
        block_size,
        expert_mask,
    )
    return sorted_ids, sorted_expert_ids, num_valid_ids


def moe_kernel_prepare_input(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool= False,
    use_int8_w4a8: bool= False,
    use_int8_w8a16: bool= False,
    use_int4_w4a16: bool= False,
    use_fp4_w4a16: bool= False,
    per_channel_quant: bool= False,
    block_shape: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """准备MOE kernel的输入"""
    if use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # 激活channel-wise int8量化
            assert (per_channel_quant), "int8 quantization only supports block or channel-wise"
            A, A_scale = per_token_quant_int8(A)
        else:
            # 激活block-wise int8量化
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    elif use_int8_w4a8:
        assert B_scale is not None
        if block_shape is None:
            # 激活channel-wise int8量化
            assert (per_channel_quant), "int8 quantization only supports block or channel-wise"
            A, A_scale = per_token_quant_int8(A)
        else:
            # 激活block-wise int8量化
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    elif use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # 激活channel-wise int8量化
            assert (per_channel_quant), "int8 quantization only supports block or channel-wise"
            block_k = A.shape[-1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
        else:
            # 激活block-wise int8量化
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16 or use_fp4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    return A, A_scale

def get_fp8_dtype():
    if not torch.cuda.is_available():
        return None  # CPU 不支持 FP8

    # 检查设备是否支持 FP8（需 CUDA 11.8+ 且 GPU 为 Hopper 及以上架构）
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) >= (9, 0):  # Hopper 及以上架构（如 H100）
        return torch.float8_e4m3fn  # 常用 FP8 类型
    else:
        return None  # 不支持 FP8 的设备返回 None

def get_compile_backend():
    if torch.cuda.is_available():
        # CUDA 设备优先用 inductor（PyTorch 默认高效后端）
        return "inductor"
    else:
        # CPU 可用 aot_eager 或 inductor
        return "aot_eager"

def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor for quantization.
    """
    dtype = torch.float8_e4m3fn #current_platform.fp8_dtype() if dtype is None else dtype
    assert (x.shape[-1] % group_size == 0), (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=torch.float32)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device,
                          dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    x_q = x_q.to(torch.float8_e4m3fn)

    return x_q, x_s

@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0)
    y = tl.cast(y,tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8`
            is supported for now.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
            scaling factor for quantization.
    """
    assert (x.shape[-1] % group_size == 0
            ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size, ),
        device=x.device,
        dtype=torch.float32,
    )

    M = x.numel()
    configs = get_w8a8_group_quant_configs(M, group_size)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = {
            "BLOCK_SIZE": 128,
            "num_warps": 1,
            "num_stages": 1,
        }

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE']),
    )
    _per_token_group_quant_int8[grid](
        x,
        x_q,
        x_s,
        M,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        GROUP_SIZE=group_size,
        **config
    )

    return x_q, x_s


@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    M,
    # Avoid to divide zero
    eps,
    int8_min,
    int8_max,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    g_id = tl.program_id(0)
    y_ptr += g_id * BLOCK_SIZE
    y_q_ptr += g_id * BLOCK_SIZE
    S_NUM: tl.constexpr = BLOCK_SIZE // GROUP_SIZE
    y_s_ptr += g_id * S_NUM

    cols = tl.arange(0, BLOCK_SIZE)  # N <= BLOCK_SIZE
    s_cols = tl.arange(0, S_NUM)
    mask = g_id * BLOCK_SIZE + cols < M

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.reshape(y, (S_NUM, GROUP_SIZE))
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = (_absmax / int8_max).reshape(S_NUM, 1)
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    y_q = tl.reshape(y_q, (S_NUM * GROUP_SIZE))
    y_s = tl.reshape(y_s, (S_NUM))

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr + s_cols, y_s.to(y_s_ptr.dtype.element_ty))

@functools.lru_cache
def get_w8a8_group_quant_configs(
    M: int, groupSize: int
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    config_file_path = get_w8a8_group_quant_config_filepath(M, groupSize)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            #logger.info(
            #    "Using configuration from %s for W8A8 GROUP QUANT kernel.",
            #    config_file_path,
            #)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default W8A8 GROUP QUANT kernel config. Performance might "
            "be sub-optimal! Config file not found at %s"
        ),
        config_file_path,
    )
    return None

@functools.lru_cache
def get_w8a8_group_quant_config_filepath(M: int, GROUP_SIZE: int) -> str:
    device_name = get_device_name()

    if device_name.lower().startswith("bw"):
        device_name = "BW200"
    json_file_name = f"w8a8_per_token_group_quant_device_name={device_name},group_size={GROUP_SIZE}.json"
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../configs/quant_configs", json_file_name
    )
    return config_file_path

# def per_token_quant_int8(x):
#     M = x.numel() // x.shape[-1]
#     N = x.shape[-1]
#     x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
#     scales = torch.empty(x.shape[:-1] + (1,),
#                          device=x.device, dtype=torch.float32)
#     BLOCK = triton.next_power_of_2(N)
#     # heuristics for number of warps
#     num_warps = min(max(BLOCK // 256, 1), 8)

#     assert x.is_contiguous()
#     _per_token_quant_int8[(M,)](
#         x,
#         x_q,
#         scales,
#         stride_x=x.stride(-2),
#         stride_xq=x_q.stride(-2),
#         N=N,
#         BLOCK=BLOCK,
#         num_warps=num_warps,
#         num_stages=1,
#     )

#     return x_q, scales

# @triton.jit
# def _per_token_quant_int8(
#     x_ptr,
#     xq_ptr,
#     scale_ptr,
#     stride_x,
#     stride_xq,
#     N,
#     BLOCK: tl.constexpr,
# ):
#     row_id = tl.program_id(0)

#     cols = tl.arange(0, BLOCK)
#     mask = cols < N

#     x = tl.load(x_ptr + row_id * stride_x + cols,
#                 mask=mask, other=0.0).to(tl.float32)
#     absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
#     scale_x = absmax / 127
#     x_q = x * (127 / absmax)
#     # x_q = tl.extra.cuda.libdevice.nearbyint(x_q).to(tl.int8)
#     x_q = libdevice.nearbyint(x_q).to(tl.int8)

#     tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
#     tl.store(scale_ptr + row_id, scale_x)

@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]).to(tl.int64)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        b_zp_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N: tl.constexpr,
        K: tl.constexpr,
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
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        block_k_diviable: tl.constexpr,
        block_n_diviable: tl.constexpr,
        group_size: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        COMBINE_SCALE_LOAD: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        USE_ADDR_OFFSET_INT64_A: tl.constexpr,
        USE_ADDR_OFFSET_INT64_C: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bsn >= 0)
    tl.assume(stride_bze >= 0)
    tl.assume(stride_bzk >= 0)
    tl.assume(stride_bzn >= 0)

    # to notify the compiler that sorted_token_ids_ptr is a pointer to the memory,
    # and all value in the memory is non-negative.
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    tl.static_assert(COMBINE_SCALE_LOAD == False, "COMBINE_SCALE_LOAD not support for awq!")
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    tl.assume(off_experts >= 0)

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N)).to(tl.int32) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int32)


    if USE_ADDR_OFFSET_INT64_A:
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                        offs_k[None, :] * stride_ak).to(tl.int64)
    else:
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                        offs_k[None, :] * stride_ak).to(tl.int32)

    if use_int4_w4a16:
        b_ptrs = b_ptr + (off_experts * stride_be + \
            (offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * \
                stride_bn).to(tl.int32)
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = b_ptr + (off_experts * stride_be + \
            offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn).to(tl.int32)

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = b_scale_ptr + (off_experts * stride_bse + \
            offs_bn[None, :] * stride_bsn + \
            ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * \
                stride_bsk).to(tl.int32)
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + (off_experts * stride_bze + \
                (offs_bn[None, :] // 2) * stride_bzn + \
                offs_k_true * stride_bzk).to(tl.int32)
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + (off_experts * stride_bze + \
                offs_bn[None, :] * stride_bzn + \
                offs_k_true * stride_bzk).to(tl.int32)
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)

        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if USE_ADDR_OFFSET_INT64_C:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]).to(tl.int64)
    else:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]).to(tl.int32)

    if block_n_diviable:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


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
        block_k_diviable: tl.constexpr,
        block_n_diviable: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        COMBINE_SCALE_LOAD: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        USE_ADDR_OFFSET_INT64_A: tl.constexpr,
        USE_ADDR_OFFSET_INT64_C: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bsn >= 0)

    # to notify the compiler that sorted_token_ids_ptr is a pointer to the memory,
    # and all value in the memory is non-negative.
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    if group_k > 0:
        tl.static_assert(BLOCK_SIZE_K <= group_k and group_k % BLOCK_SIZE_K == 0,
            "BLOCK_SIZE_K must be divisible by GROUP_SIZE_K")
    if COMBINE_SCALE_LOAD: # used for use_int8_w8a8
        tl.static_assert(stride_ask == 1,
            "COMBINE_SCALE_LOAD implictly stride_ask == 1!")
        tl.static_assert(MUL_ROUTED_WEIGHT == False,
            "COMBINE_SCALE_LOAD and MUL_ROUTED_WEIGHT cannot be both true due to w1_scale and w2_scale diff layout!")
        tl.static_assert(block_k_diviable == True and BLOCK_SIZE_K == group_k,
            "COMBINE_SCALE_LOAD only add and verify on block_k_diviable!")
        tl.static_assert(use_int8_w8a8 == True,
            "COMBINE_SCALE_LOAD only add and verify on use_int8_w8a8!")
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int32)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int32)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if USE_ADDR_OFFSET_INT64_A:
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                offs_k[None, :] * stride_ak).to(tl.int64)
    else:
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                offs_k[None, :] * stride_ak).to(tl.int32)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn).to(tl.int32)

    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            if COMBINE_SCALE_LOAD:
                a_scale_ptrs = a_scale_ptr + (offs_token[:, None] // top_k) * stride_asm
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                offs_bsn[:, None] * stride_bsn)
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                offs_bsn * stride_bsn)

        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if COMBINE_SCALE_LOAD:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K), 2):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if not block_k_diviable:
                a0 = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0)
                b0 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            else:
                a0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b0 = tl.load(b_ptrs)

            # We accumulate along the K dimension.
            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
                tl.static_assert(False, "Not implemented")
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k + tl.arange(0, 2)
                    a_scale = tl.load(a_scale_ptrs + offs_ks[None, :] * stride_ask,
                                    mask=token_mask[:, None],
                                    other=0.0)
                    b_scale = tl.load(b_scale_ptrs + offs_ks[None, :] * stride_bsk)
                    a_scale_0, a_scale_1 = tl.split(a_scale)
                    b_scale_0, b_scale_1 = tl.split(b_scale)

                    accumulator += tl.dot(a0, b0) * a_scale_0[:,
                                                        None] * b_scale_0[None, :]

                    if not block_k_diviable:
                        a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak,
                                    mask=token_mask[:, None] & (offs_k[None, :] < K - (k + 1) * BLOCK_SIZE_K),
                                    other=0.0)
                        b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk,
                                    mask=offs_k[:, None] < K - (k + 1) * BLOCK_SIZE_K, other=0.0)
                    else:
                        a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak, mask=token_mask[:, None], other=0.0)
                        b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk)
                    accumulator += tl.dot(a1, b1) * a_scale_1[:,
                                                        None] * b_scale_1[None, :]
                else:
                    accumulator = tl.dot(a, b, acc=accumulator)
                    tl.static_assert(False, "Not implemented")
            else:
                accumulator += tl.dot(a, b)
                tl.static_assert(False, "Not implemented")

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak * 2
            b_ptrs += BLOCK_SIZE_K * stride_bk * 2

    else: # non-COMBINE_SCALE_LOAD

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if not block_k_diviable:
                a = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            else:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)

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
                    accumulator = tl.dot(a, b, acc=accumulator)
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
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if USE_ADDR_OFFSET_INT64_C:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]).to(tl.int64)
    else:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]).to(tl.int32)
    if not block_n_diviable:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    else:
        c_mask = token_mask[:, None]

    tl.store(c_ptrs, accumulator, mask=c_mask)



def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread,
                                         numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


# Triton implementation based on:
# https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0
def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts, )
    tokens_cnts = torch.zeros((num_experts + 1, num_experts),
                              dtype=torch.int32,
                              device=topk_ids.device)
    cumsum = torch.zeros((num_experts + 1, ),
                         dtype=torch.int32,
                         device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1, )](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    if num_experts >= 224:
        if num_experts not in (256, 896):
            # Triton fallback only writes valid blocks. Keep padding/tail safe
            # without changing the Triton kernel semantics.
            sorted_ids.fill_(topk_ids.numel())
            expert_ids.zero_()
            moe_align_block_size_triton(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )
        else:
            # Currently supports num_experts=256/896.
            moe_c_sgl_moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )
    else:
        moe_c_moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                                 expert_ids, num_tokens_post_pad)
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad



# def generate_sum_configs():
#     configs = []
#     for block_m in [16, 32, 64, 128]:
#         for block_n in [32, 64, 128, 256]:
#             for num_warps in [2, 4, 8, 16]:
#                 for num_stages in [1, 2]:
#                     config = triton.Config({
#                         'BLOCK_SIZE_M': block_m,
#                         'BLOCK_SIZE_N': block_n,
#                     }, num_warps=num_warps, num_stages=num_stages)
#                 configs.append(config)
#     return configs

# @triton.autotune(
#     key=['M', 'N', 'top_k','compute_type'],
#     # configs=generate_sum_configs(),
#     configs = [
#                 triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32  }, num_warps=4),
#                 triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64 },  num_warps=8),
#                 triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128 }, num_warps=8),
#                 triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128 }, num_warps=16),
#               ],
#     # perf_debug=True,
# )


device_name = get_device_name()
if device_name=='K100_AI':
    moe_sum_best_configs = {
        # M, topK, N
        (1,    8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32,  "num_warps" : 4 },
        (4,    8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, "num_warps" : 16},
        (16,   8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
        (32,   8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
        (64,   8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
        (128,  8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
    }
elif device_name=='BW200' or device_name.upper().startswith('BW'):
    moe_sum_best_configs = {
        # M, topK, N
        (1,    8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32,  "num_warps" : 4} ,
        (4,    8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, "num_warps" : 16},
        (16,   8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
        (32,   8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
        (64,   8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
        (128,  8, 7168): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, "num_warps" : 16},
    }
else:
    moe_sum_best_configs = {}

def get_moe_sum_config(M, top_k, N):

    if moe_sum_best_configs:
        # config = moe_sum_best_configs[min(moe_sum_best_configs.keys(), key=lambda x: abs(x[0] - M))] #torch.compile不支持
        best_key = None
        min_diff = float('inf')
        for key in moe_sum_best_configs.keys():
            diff = abs(key[0] - M)
            if diff < min_diff:
                min_diff = diff
                best_key = key
        config = moe_sum_best_configs[best_key]

    else:
        if M < 32:
            config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64,  "num_warps" : 4}
        else:
            config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, "num_warps" : 8}

    return config

@triton.heuristics({
    'block_m_dividable': lambda nargs: nargs['M'] % nargs['BLOCK_SIZE_M'] == 0,
    'block_n_dividable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
})


@triton.jit
def moe_sum_kernel(
    # Pointers to matrices
    output_ptr,             # [M, N]
    input_ptr,              # [M, top_k, N]
    # Matrix dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    top_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    stride_output_m,
    stride_output_n,
    stride_input_m,
    stride_input_k,
    stride_input_n,
    compute_type: tl.constexpr,
    block_m_dividable: tl.constexpr,
    block_n_dividable: tl.constexpr,
):
    """
    Args:
        output_ptr: shape [M, N]
        input_ptr: shape[M, top_k, N]
    """
    tl.assume(stride_output_m >= 0)
    tl.assume(stride_output_n >= 0)
    tl.assume(stride_input_m >= 0)
    tl.assume(stride_input_k >= 0)
    tl.assume(stride_input_n >= 0)

    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(top_k):
        input_ptrs = input_ptr + (offs_m[:, None] * stride_input_m +
                                 k * stride_input_k + offs_n[None, :] * stride_input_n).to(tl.int32)
        if block_m_dividable:
            x = tl.load(input_ptrs)
        else:
            x = tl.load(input_ptrs,
                        mask=mask_m[:, None] & mask_n[None, :],
                        other=0.0)

        acc += x

    acc = acc.to(compute_type)

    output_ptrs = output_ptr + (offs_m[:, None] * stride_output_m +
                                offs_n[None, :] * stride_output_n).to(tl.int32)

    if block_m_dividable and block_n_dividable:
        tl.store(output_ptrs, acc)
    else:
        tl.store(output_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_moe_sum_noaiter(input_tensor,
                   output_tensor,
                   routed_scaling_factor: float = 1.0):
    """
    Args:
        input_tensor: [M, top_k, N]
        output_tensor: [M, N]
    """
    M, top_k, N = input_tensor.shape

    # 计算grid
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
    #                                  N, META['BLOCK_SIZE_N']), )
    config = get_moe_sum_config(M, top_k, N)
    grid = (triton.cdiv(M, config['BLOCK_SIZE_M']) * triton.cdiv(N, config['BLOCK_SIZE_N']),)

    # Check constraints.
    assert output_tensor.dtype == torch.float16 or \
           output_tensor.dtype == torch.bfloat16 or \
           output_tensor.dtype == torch.float32

    if output_tensor.dtype == torch.float16:
        compute_type = tl.float16
    elif output_tensor.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif output_tensor.dtype == torch.float32:
        compute_type = tl.float32

    assert input_tensor.is_contiguous()
    assert output_tensor.is_contiguous()
    assert input_tensor.shape[0] == output_tensor.shape[0]
    assert input_tensor.shape[2] == output_tensor.shape[1]

    moe_sum_kernel[grid](
        output_tensor,
        input_tensor,
        M,
        N,
        top_k,
        stride_output_m=output_tensor.stride(0),
        stride_output_n=output_tensor.stride(1),
        stride_input_m=input_tensor.stride(0),
        stride_input_k=input_tensor.stride(1),
        stride_input_n=input_tensor.stride(2),
        compute_type=compute_type,
        **config,
    )
    if routed_scaling_factor != 1.0:
        output_tensor.mul_(routed_scaling_factor)

    return output_tensor

def invoke_fused_moe_kernel(A: torch.Tensor,
                            B: torch.Tensor,
                            B_new: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: torch.Tensor,
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: Dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            use_int4_w4a16_base: bool,
                            BM: int,
                            BN: int,
                            BK: int,
                            kloops: int,
                            nloops: int,
                            is_bottom: bool,
                            block_shape: Optional[List[int]] = None) -> None:
    find_best = os.getenv("WHICH_TO_TEST")

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert B_scale is not None

        assert len(block_shape) == 2
        block_n, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
        assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
        assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            A, A_scale = per_token_quant_int8(A)
        else:
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)

            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        # assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    EM = sorted_token_ids.shape[0]

    if(find_best):
        if A.shape[0] < BM:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique, so
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(sorted_token_ids.shape[0],
                    A.shape[0] * top_k * BM)
    else:
        if A.shape[0] < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique, so
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(sorted_token_ids.shape[0],
                    A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1], META['BLOCK_SIZE_N']), )

    if (use_int8_w8a16 or use_int4_w4a16 or use_int8_w8a8 or use_fp8_w8a8) and \
            block_shape is not None and block_shape[1] > 0:
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=topk_ids.numel(),
            group_size=block_shape[1],
            num_experts=B.shape[0],
            bit=4 if use_int4_w4a16 else 8)
        config = config.copy()
        config.update(
            get_moe_wna16_block_config(config=config,
                                       use_moe_wna16_cuda=use_moe_wna16_cuda,
                                       num_valid_tokens=topk_ids.numel(),
                                       size_k=A.shape[1],
                                       size_n=B.shape[1],
                                       num_experts=B.shape[1],
                                       group_size=block_shape[1],
                                       real_top_k=topk_ids.shape[1],
                                       block_size_m=config["BLOCK_SIZE_M"]))

        if(find_best):
            config["BLOCK_SIZE_M"] = BM
            config["BLOCK_SIZE_N"] = BN
            config["BLOCK_SIZE_K"] = BK
            config["kloops"] = kloops
            config["nloops"] = nloops
        else:
            config.setdefault("kloops", kloops)
            config.setdefault("nloops", nloops)
        if use_moe_wna16_cuda:
        # if True:
            # print("calling adding path -------")
            if block_shape[0] > 1 and block_shape[1] > 1:
                bit = 4 if use_int4_w4a16 else 8
                if use_int8_w8a16:
                    # print("calling w8a16  block wise -------")

                    moe_c_moe_w8a16_gemm_block_wise(A, C, B, B_scale, B_zp,
                                    topk_weights if mul_routed_weight else None,
                                    sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                    config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                    config["BLOCK_SIZE_K"], bit)
                elif use_int8_w8a8:
                    # print("calling w8a8  block wise kernel2 -------")
                    if is_bottom:
                        moe_c_moe_w8a8_gemm_block_wise_kernel2(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"], config["kloops"],config["nloops"],bit)
                    else :
                        moe_c_moe_w8a8_gemm_block_wise(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"], config["kloops"],config["nloops"],bit)
                    # print("finished calling w8a8  block wise kernel2 -------")
                elif use_fp8_w8a8:
                    if is_bottom:
                        # print("calling w8a8  block wise -------")
                        A = A.view(torch.int8)
                        B = B.view(torch.int8)
                        moe_c_moe_w8a8_gemm_block_wise_kernel2_fp8(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"],config["kloops"],config["nloops"],bit)
                        # print("finished calling w8a8  block wise -------")
                    else :
                        # print("calling w8a8  block wise -------")
                        A = A.view(torch.int8)
                        B = B.view(torch.int8)
                        moe_c_moe_w8a8_gemm_block_wise_fp8(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"],config["kloops"],config["nloops"],bit)
                        # print("finished calling w8a8  block wise -------")
                return


            else :
                bit = 4 if use_int4_w4a16 else 8
                # print("calling wna16 awq -------")
                if (bit == 8 and use_int8_w8a16) :
                    # print("calling w8a16 awq -------")
                    moe_c_moe_w8a16_gemm_awq(A, C, B, B_scale, B_zp,
                                    topk_weights if mul_routed_weight else None,
                                    sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, top_k,
                                    config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                    config["BLOCK_SIZE_K"], bit)
                    return

                elif use_int4_w4a16_base :
                    # print("calling w4a16 awq base -------")
                    # ops.moe_gemm_marlin_w4a16(A, B_new, C, B_scale, B_zp, None,
                    #                         sorted_token_ids, expert_ids, num_tokens_post_padded,8, 54, 1)
                    moe_c_moe_wna16_gemm_base(A, C, B, B_scale, B_zp,
                                    topk_weights if mul_routed_weight else None,
                                    sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, top_k,
                                    config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                    config["BLOCK_SIZE_K"], bit)
                    return
                else :
                    if is_bottom:
                        # print("calling w4a16 awq -------")
                        # ops.moe_gemm_marlin_w4a16(A, B_new, C, B_scale, B_zp, topk_weights,
                        #     sorted_token_ids, expert_ids, num_tokens_post_padded,1, 54, 1)
                        moe_c_moe_wna16_gemm_2(A, C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"], config["kloops"], config["nloops"], bit)
                    else :
                        # print(f"calling w4a16 awq {top_k}-------")
                        # ops.moe_gemm_marlin_w4a16(A, B_new, C, B_scale, B_zp, None,
                        #                     sorted_token_ids, expert_ids, num_tokens_post_padded,8, 54, 1)
                        moe_c_moe_wna16_gemm(A, C, B_new, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"], config["kloops"], config["nloops"], bit)
                        # print("calling w4a16 awq end-------")
                    return


        offset_max = 2**31 - 1
        use_addr_offset_int64_a = A.numel() * A.element_size() >= offset_max
        use_addr_offset_int64_c = C.numel() * C.element_size() >= offset_max
        if (A.numel() * A.element_size() >= offset_max or
            B.numel() * B.element_size() >= offset_max or
            C.numel() * C.element_size() >= offset_max):
            logger.warning(
                ("A,B,C numel：%ld, %ld, %ld has out of range for fused_moe_kernel kernel!"
                "Use int64 for address offset."), A.numel(), B.numel(), C.numel())

        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
            block_n_diviable=B.shape[1] % config["BLOCK_SIZE_N"] == 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
            USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            **config,
        )
    else:
        # simple check out of range, not accurate.
        offset_max = 2**31 - 1
        use_addr_offset_int64_a = A.numel() * A.element_size() >= offset_max
        use_addr_offset_int64_c = C.numel() * C.element_size() >= offset_max
        if (A.numel() * A.element_size() >= offset_max or
            B.numel() * B.element_size() >= offset_max or
            C.numel() * C.element_size() >= offset_max):
            logger.warning(
                ("A,B,C numel：%ld, %ld, %ld has out of range for fused_moe_kernel kernel!"
                "Use int64 for address offset."), A.numel(), B.numel(), C.numel())


        config = config.copy()
        BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
        if block_shape is not None:
            BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0],
                                                 block_shape[1]))
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
            B.shape[1],
            B.shape[2],
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
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
            block_k_diviable=A.shape[1] % BLOCK_SIZE_K == 0,
            block_n_diviable=B.shape[1] % config["BLOCK_SIZE_N"] == 0,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
            USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            COMBINE_SCALE_LOAD=config.pop("COMBINE_SCALE_LOAD", None),
            **config,
        )

_FLOAT_MOE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_QUANTIZED_ACTIVATION_DTYPES = (torch.int8, torch.float8_e4m3fn)


def _is_prequantized_activation(
    hidden_states_dtype: torch.dtype,
    a_scale: Optional[torch.Tensor],
) -> bool:
    return (
        a_scale is not None
        or hidden_states_dtype in _QUANTIZED_ACTIVATION_DTYPES
    )


def _resolve_moe_compute_dtype(
    hidden_states_dtype: torch.dtype,
    compute_dtype: Optional[torch.dtype] = None,
    *,
    prequantized: bool = False,
) -> torch.dtype:
    """Resolve fp16/bf16/fp32 dtype for GEMM outputs, caches, and compute_type."""
    if hidden_states_dtype in _FLOAT_MOE_DTYPES:
        return hidden_states_dtype
    if prequantized or hidden_states_dtype in _QUANTIZED_ACTIVATION_DTYPES:
        if compute_dtype is not None:
            assert compute_dtype in _FLOAT_MOE_DTYPES, (
                f"compute_dtype must be fp16/bf16/fp32, got {compute_dtype}")
            return compute_dtype
        return torch.bfloat16
    raise ValueError(
        f"Unsupported hidden_states dtype: {hidden_states_dtype}")


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported compute dtype for triton: {dtype}")


def _is_marlin_tensorwise_scale(
    B_scale: Optional[torch.Tensor],
    num_experts: int,
) -> bool:
    # Marlin W8A8 tensorwise path expects one scale per expert,
    # represented as (E, 1, 1) to stay compatible with existing 3D scale checks.
    return B_scale is not None and B_scale.shape == (num_experts, 1, 1)


def _make_w16a16_config_pair(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config1 = dict(config1)
    config2 = dict(config2)
    bm1 = int(config1["BLOCK_SIZE_M"])
    bm2 = int(config2["BLOCK_SIZE_M"])
    assert bm1 >= bm2 and bm1 % bm2 == 0
    config1["DELTA"] = 1
    config2["DELTA"] = bm1 // bm2
    return config1, config2


def _validate_prequant_marlin_activation(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    expected_dtype,
    block_shape: Optional[List[int]],
    B: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    """Validate pre-quantized activation (A, A_scale) for marlin MoE kernels."""
    assert A_scale is not None
    allowed = (expected_dtype,) if isinstance(expected_dtype, torch.dtype) else tuple(expected_dtype)
    assert A.dtype in allowed, (
        f"pre-quantized A must be one of {allowed}, got {A.dtype}")
    if block_shape is None:
        assert A_scale.shape[-1] == 1, (
            f"per-token A_scale last dim must be 1, got shape {A_scale.shape}")
    else:
        assert len(block_shape) == 2
        block_n, block_k = block_shape[0], block_shape[1]
        assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
        assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
        assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]


def _wfp4a8_use_a_quant_group() -> bool:
    return os.getenv("AITER_WFP4A8_A_QUANT", "0").strip() == "1"


def invoke_fused_moe_kernel_marlin(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: torch.Tensor,
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            MODE: int,
                            config: Dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w4a8 : bool,
                            use_fp4_w4a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            use_fp4_w4a16: bool,
                            use_int4_w4a16_base: bool,
                            use_w16a16: bool,
                            is_bottom: bool,
                            key_selected:int,
                            block_shape: Optional[List[int]] = None,
                            real_size_k: Optional[int] = None) -> None:
    find_best = os.getenv("WHICH_TO_TEST")
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1


    use_fp4_w4a8_groupwise = _check_moec_quant_legal(
        use_int4_w4a16=use_int4_w4a16,
        use_fp4_w4a16=use_fp4_w4a16,
        use_int8_w4a8=use_int8_w4a8,
        use_fp4_w4a8=use_fp4_w4a8,
        block_shape=block_shape,
    )
    use_wfp4a8_a_quant_group = use_fp4_w4a8_groupwise and _wfp4a8_use_a_quant_group()
    if use_fp8_w8a8 or use_fp4_w4a8:
        assert B_scale is not None
        if A_scale is not None:
            # Pre-quantized fp8 activation; skip internal quantization.
            _validate_prequant_marlin_activation(
                A, A_scale, (torch.float8_e4m3fn, torch.int8),
                block_shape, B, B_scale)
        elif use_wfp4a8_a_quant_group:
            block_k = block_shape[1]
            A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
        elif block_shape is None:
            if real_size_k is not None and real_size_k != A.shape[-1]:
                A, A_scale = per_token_group_quant_fp8(A, A.shape[-1])
            else:
                A, A_scale = per_token_quant_hip(A, quant_dtype=torch.float8_e4m3fn)
        elif use_fp4_w4a8_groupwise:
            A, A_scale = per_token_quant_hip(A, quant_dtype=torch.float8_e4m3fn)
        else:
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
        if use_fp4_w4a8_groupwise:
            assert B_scale.dtype == torch.uint8
            assert B_scale.shape[-1] * 32 == A.shape[-1]
    elif use_int8_w8a8:
        assert B_scale is not None
        if A_scale is not None:
            # Pre-quantized int8 activation; skip internal quantization.
            _validate_prequant_marlin_activation(
                A, A_scale, torch.int8, block_shape, B, B_scale)
        elif block_shape is None:
            A, A_scale = moe_kernel_prepare_input(
                    A=A,
                    B=B,
                    A_scale=None,
                    B_scale=B_scale,
                    use_fp8_w8a8=False,
                    use_int8_w8a8=True,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    use_fp4_w4a16=False,
                    per_channel_quant=True,
                    block_shape=None
                )
        else:
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w4a8:
        assert B_scale is not None
        if A_scale is not None:
            # Pre-quantized int8 activation; skip internal quantization.
            _validate_prequant_marlin_activation(
                A, A_scale, torch.int8, block_shape, B, B_scale)
        elif block_shape is None:
            A, A_scale = moe_kernel_prepare_input(
                    A=A,
                    B=B,
                    A_scale=None,
                    B_scale=B_scale,
                    use_fp8_w8a8=False,
                    use_int8_w8a8=False,
                    use_int8_w4a8=True,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    use_fp4_w4a16=False,
                    per_channel_quant=True,
                    block_shape=None
                )
        else:
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16 or use_fp4_w4a16:
        assert B_scale is not None
        # assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    if use_w16a16:
        if(find_best):
            config["MODE"] = MODE
        gemm_op = (
            moe_c_moe_gemm_marlin_w16a16_asm
            if int(config["MODE"]) >= 1000
            else moe_c_moe_gemm_marlin_w16a16
        )
        gemm_top_k = top_k
        if is_bottom and int(config["MODE"]) >= 400:
            gemm_top_k = 1
        if is_bottom:
            gemm_op(
                A, B, C, topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                gemm_top_k, config["MODE"], int(config.get("DELTA", 1)))
        else:
            gemm_op(
                A, B, C, None,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                top_k, config["MODE"], int(config.get("DELTA", 1)))
        return

    if (use_int8_w8a16 or use_int4_w4a16 or use_fp4_w4a16 or use_fp4_w4a8 or use_fp8_w8a8 or use_int8_w8a8 or use_int8_w4a8) :
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3

        # use_moe_wna16_cuda = should_moe_wna16_use_cuda(
        #     num_valid_tokens=topk_ids.numel(),
        #     group_size=block_shape[1],
        #     num_experts=B.shape[0],
        #     bit=4 if use_int4_w4a16 else 8)
        use_moe_wna16_cuda = True

        if(find_best):
            config["MODE"] = MODE

        if use_moe_wna16_cuda:
            if block_shape and block_shape[0] > 1 and block_shape[1] > 1:
                bit = 4 if (use_int4_w4a16 or use_fp4_w4a16 or use_fp4_w4a8) else 8

                if use_int8_w8a16:
                    moe_c_moe_w8a16_gemm_block_wise(A, C, B, B_scale, B_zp,
                                    topk_weights if mul_routed_weight else None,
                                    sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                    config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                    config["BLOCK_SIZE_K"], bit)
                elif use_int8_w8a8:
                    if is_bottom:
                        moe_c_moe_w8a8_gemm_block_wise_kernel2(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"], config["kloops"],config["nloops"],bit)
                    else :
                        moe_c_moe_w8a8_gemm_block_wise(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"], config["kloops"],config["nloops"],bit)
                elif use_fp8_w8a8:
                    if is_bottom:
                        A = A.view(torch.int8)
                        B = B.view(torch.int8)
                        moe_c_moe_w8a8_gemm_block_wise_kernel2_fp8(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"],config["kloops"],config["nloops"],bit)
                    else :
                        A = A.view(torch.int8)
                        B = B.view(torch.int8)
                        moe_c_moe_w8a8_gemm_block_wise_fp8(A, A_scale,C, B, B_scale, B_zp,
                                        topk_weights if mul_routed_weight else None,
                                        sorted_token_ids, expert_ids,
                                        num_tokens_post_padded, block_shape[0], block_shape[1], top_k,
                                        config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                        config["BLOCK_SIZE_K"],config["kloops"],config["nloops"],bit)
                        # print("finished calling w8a8  block wise -------")
                return



            else :
                bit = 4 if (use_int4_w4a16 or use_fp4_w4a16 or use_fp4_w4a8 or use_int4_w4a16_base)  else 8
                if (bit == 8 and use_int8_w8a16) :
                    B = B.view(torch.uint32)
                    if is_bottom:
                        moe_c_moe_gemm_marlin_w8a16(A, B , C, B_scale, topk_weights, #B处应该传shuffle权重 待修改
                            sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"], 1)
                    else:
                        moe_c_moe_gemm_marlin_w8a16(A, B, C, B_scale, None, #B处应该传shuffle权重 待修改
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,8, config["MODE"], 1)
                    return

                elif use_int4_w4a16_base :
                    moe_c_moe_wna16_gemm_base(A, C, B, B_scale, B_zp,
                                    topk_weights if mul_routed_weight else None,
                                    sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, top_k,
                                    config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                                    config["BLOCK_SIZE_K"], bit)
                    return
                else :


                    if use_int4_w4a16 or use_fp4_w4a16:
                        B = B.view(torch.uint32)
                        gemm_op_w4 = (
                            moe_c_moe_gemm_marlin_wfp4a16
                            if use_fp4_w4a16 else
                            moe_c_moe_gemm_marlin_w4a16
                        )
                        if is_bottom:
                            gemm_op_w4(A, B, C, B_scale, B_zp, topk_weights,
                                sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"], 1)
                        else :
                            gemm_op_w4(A, B, C, B_scale, B_zp, None,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, config["MODE"], 1)
                        return
                    elif (use_int8_w8a8 and block_shape == None):
                        if _is_marlin_tensorwise_scale(B_scale, B.shape[0]):
                            if is_bottom:
                                moe_c_moe_gemm_marlin_w8a8_tensorwise(A, B, C, A_scale, B_scale, topk_weights,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded, 1, config["MODE"], top_k, key_selected)
                            else:
                                moe_c_moe_gemm_marlin_w8a8_tensorwise(A, B, C, A_scale, B_scale, None,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, config["MODE"], top_k, key_selected)
                        else:
                            if is_bottom:
                                assert B.shape[1] in [7168,6144,4096,3072,2048] , f" K = {B.shape[1]} is not in support"
                                assert B.shape[2] in [128,256,384,512,768,1024,2048] , f" N = {B.shape[2]} is not in support"
                                moe_c_moe_gemm_marlin_w8a8(A, B, C, A_scale, B_scale,topk_weights,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"],  top_k,key_selected)


                            else :
                                moe_c_moe_gemm_marlin_w8a8(A, B, C, A_scale, B_scale, None,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded,top_k, config["MODE"],  top_k,key_selected)
                        return
                    elif (use_int8_w4a8 and block_shape == None):
                        if is_bottom:
                            moe_c_moe_gemm_marlin_w4a8(A, B, C, A_scale, B_scale,topk_weights,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"], top_k,key_selected)

                        else :
                            moe_c_moe_gemm_marlin_w4a8(A, B, C, A_scale, B_scale, None,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,top_k, config["MODE"], top_k,key_selected)

                        return
                    elif (use_fp4_w4a8 and block_shape is None):
                        if is_bottom:
                            moe_c_moe_gemm_marlin_wfp4a8_channelwise(A, B, C, A_scale, B_scale, topk_weights,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"], top_k,key_selected)
                        else:
                            moe_c_moe_gemm_marlin_wfp4a8_channelwise(A, B, C, A_scale, B_scale, None,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,top_k, config["MODE"], top_k,key_selected)
                        return
                    elif (use_fp4_w4a8 and block_shape is not None and block_shape[0] == 0 and block_shape[1] == 32):
                        gemm_op_wfp4a8_groupwise = (
                            moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup
                            if use_wfp4a8_a_quant_group else
                            moe_c_moe_gemm_marlin_wfp4a8_groupwise
                        )
                        if is_bottom:
                            gemm_op_wfp4a8_groupwise(A, B, C, A_scale, B_scale, topk_weights,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"], top_k,key_selected)
                        else:
                            gemm_op_wfp4a8_groupwise(A, B, C, A_scale, B_scale, None,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,top_k, config["MODE"], top_k,key_selected)
                        return
                    elif (use_fp8_w8a8 and block_shape == None):
                        gemm_real_size_k = int(real_size_k or A.shape[1])
                        if _is_marlin_tensorwise_scale(B_scale, B.shape[0]):
                            if is_bottom:
                                moe_c_moe_gemm_marlin_w8a8_fp8_tensorwise(A, B, C, A_scale, B_scale,topk_weights,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"],  top_k,key_selected, gemm_real_size_k)


                            else :
                                moe_c_moe_gemm_marlin_w8a8_fp8_tensorwise(A, B, C, A_scale, B_scale, None,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded,top_k, config["MODE"],  top_k,key_selected, gemm_real_size_k)
                        else:
                            if is_bottom:
                                moe_c_moe_gemm_marlin_w8a8_fp8(A, B, C, A_scale, B_scale,topk_weights,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded,1, config["MODE"],  top_k,key_selected, gemm_real_size_k)


                            else :
                                moe_c_moe_gemm_marlin_w8a8_fp8(A, B, C, A_scale, B_scale, None,
                                                sorted_token_ids, expert_ids, num_tokens_post_padded,top_k, config["MODE"],  top_k,key_selected, gemm_real_size_k)
                        return




# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[List[int]] = None,
                         is_bottom: bool = False,
                         use_moe_wna16_cuda: bool = False) -> str:
    device_name = get_device_name()
    # device_name = "BW200"
    if device_name == 'BW200' or device_name.upper().startswith('BW'):
        device_name = 'BW200'
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    is_bottom_selector = ("" if is_bottom == False else ",is_bottom=True")
    block_shape_selector = ("" if not block_shape or not all(block_shape) else
                            f",block_shape={block_shape}").replace(" ", "")
    is_cuda_kernel_selector = ("" if use_moe_wna16_cuda == False else ",is_cuda_kernel=True")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{is_bottom_selector}{block_shape_selector}{is_cuda_kernel_selector}.json"  # noqa: E501


def _get_gfx_version() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]


def get_config_file_name_marlin(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[List[int]] = None,
                         is_bottom: bool = False,
                         use_moe_wna16_cuda: bool = False,
                         K: Optional[int] = None) -> str:
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    k_selector = "" if K is None else f",K={K}"
    is_bottom_selector = ("" if is_bottom == False else ",is_bottom=True")
    return f"E={E},N={N}{k_selector}{dtype_selector}{is_bottom_selector}.json"  # noqa: E501


def get_legacy_config_file_name_marlin(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[List[int]] = None,
                         is_bottom: bool = False,
                         use_moe_wna16_cuda: bool = False,
                         K: Optional[int] = None) -> str:
    gfx_version = _get_gfx_version()
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    k_selector = "" if K is None else f",K={K}"
    is_bottom_selector = ("" if is_bottom == False else ",is_bottom=True")
    return f"E={E},N={N}{k_selector},gfx_version={gfx_version}{dtype_selector}{is_bottom_selector}.json"  # noqa: E501


def _moe_c_config_root() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "moe_c_configs")


def _moe_c_config_category(dtype: Optional[str], block_shape: Optional[List[int]]) -> Optional[str]:
    if not dtype:
        return None
    if dtype == "fp4_w4a8" and block_shape and block_shape[0] == 0 and block_shape[1]:
        return f"fp4_w4a8_groupwise{block_shape[1]}"
    return dtype


def _find_moe_c_config_file(
    json_file_name: str,
    dtype: Optional[str],
    block_shape: Optional[List[int]],
    arch: Optional[str] = None,
) -> str:
    config_root = _moe_c_config_root()
    category = _moe_c_config_category(dtype, block_shape)
    if arch and category:
        arch_category_path = os.path.join(config_root, arch, category, json_file_name)
        if os.path.exists(arch_category_path):
            return arch_category_path
    if arch:
        arch_path = os.path.join(config_root, arch, json_file_name)
        if os.path.exists(arch_path):
            return arch_path
    if category:
        category_path = os.path.join(config_root, category, json_file_name)
        if os.path.exists(category_path):
            return category_path
    return os.path.join(config_root, json_file_name)


def _load_moe_c_config_file(config_file_path: str) -> Optional[Dict[int, Any]]:
    if not os.path.exists(config_file_path):
        return None
    with open(config_file_path) as f:
        logger.info("Using configuration from %s for MoE layer.", config_file_path)
        return {int(key): val for key, val in json.load(f).items()}


def _load_first_moe_c_config_file(config_file_paths: List[str]) -> Optional[Dict[int, Any]]:
    seen = set()
    for config_file_path in config_file_paths:
        if config_file_path in seen:
            continue
        seen.add(config_file_path)
        config = _load_moe_c_config_file(config_file_path)
        if config is not None:
            return config
    return None


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_bottom: bool = False,
    use_moe_wna16_cuda: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    block_shape = [block_n if block_n is not None else 0, block_k] if block_k is not None else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape, is_bottom, use_moe_wna16_cuda)

    config_file_path = _find_moe_c_config_file(json_file_name, dtype, block_shape)
    config = _load_moe_c_config_file(config_file_path)
    if config is not None:
        return config

    if is_bottom:
        fallback_json_file_name = get_config_file_name(E, N, dtype, block_shape)
        fallback_config_file_path = _find_moe_c_config_file(fallback_json_file_name, dtype, block_shape)
        config = _load_moe_c_config_file(fallback_config_file_path)
        if config is not None:
            return config

    logger.warning(
        ("Using default MoE config. Performance might be sub-optimal! "
         "Config file not found at %s"), config_file_path)
    return None

@torch._dynamo.disable
@functools.lru_cache
def get_moe_configs_marlin(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_bottom: bool = False,
    use_moe_wna16_cuda: bool = False,
    K: Optional[int] = None,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    block_shape = [block_n if block_n is not None else 0, block_k] if block_k is not None else None
    arch = _get_gfx_version()
    json_file_names = []
    if K is not None:
        json_file_names.append(get_config_file_name_marlin(
            E, N, dtype, block_shape, is_bottom, use_moe_wna16_cuda, K))
    json_file_names.append(get_config_file_name_marlin(
        E, N, dtype, block_shape, is_bottom, use_moe_wna16_cuda))

    config_file_paths = [
        _find_moe_c_config_file(json_file_name, dtype, block_shape, arch)
        for json_file_name in json_file_names
    ]
    config_file_path = config_file_paths[0]
    config = _load_first_moe_c_config_file(config_file_paths)
    if config is not None:
        return config

    legacy_json_file_names = []
    if K is not None:
        legacy_json_file_names.append(get_legacy_config_file_name_marlin(
            E, N, dtype, block_shape, is_bottom, use_moe_wna16_cuda, K))
    legacy_json_file_names.append(get_legacy_config_file_name_marlin(
        E, N, dtype, block_shape, is_bottom, use_moe_wna16_cuda))
    legacy_config_file_paths = [
        _find_moe_c_config_file(legacy_json_file_name, dtype, block_shape)
        for legacy_json_file_name in legacy_json_file_names
    ]
    config = _load_first_moe_c_config_file(legacy_config_file_paths)
    if config is not None:
        return config

    if is_bottom:
        fallback_json_file_names = []
        if K is not None:
            fallback_json_file_names.append(get_config_file_name_marlin(
                E, N, dtype, block_shape, K=K))
        fallback_json_file_names.append(get_config_file_name_marlin(
            E, N, dtype, block_shape))
        fallback_config_file_paths = [
            _find_moe_c_config_file(fallback_json_file_name, dtype, block_shape, arch)
            for fallback_json_file_name in fallback_json_file_names
        ]
        config = _load_first_moe_c_config_file(fallback_config_file_paths)
        if config is not None:
            return config

        legacy_fallback_json_file_names = []
        if K is not None:
            legacy_fallback_json_file_names.append(get_legacy_config_file_name_marlin(
                E, N, dtype, block_shape, K=K))
        legacy_fallback_json_file_names.append(get_legacy_config_file_name_marlin(
            E, N, dtype, block_shape))
        legacy_fallback_config_file_paths = [
            _find_moe_c_config_file(legacy_fallback_json_file_name, dtype, block_shape)
            for legacy_fallback_json_file_name in legacy_fallback_json_file_names
        ]
        config = _load_first_moe_c_config_file(legacy_fallback_config_file_paths)
        if config is not None:
            return config

    logger.warning(
        ("Using default MoE config. Performance might be sub-optimal! "
         "Config file not found at %s"), config_file_path)
    return None


def get_moe_wna16_block_config(config: Dict[str,
                                            int], use_moe_wna16_cuda: bool,
                               num_valid_tokens: int, size_k: int, size_n: int,
                               num_experts: int, group_size: int,
                               real_top_k: int, block_size_m: int):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        # optimal block config is set
        return {}
    if not use_moe_wna16_cuda:
        # triton moe wna16 kernel
        if num_valid_tokens // real_top_k == 1:
            # if bs=1, use a smaller BLOCK_SIZE_N
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        # cuda moe wna16 kernel
        # set default block_size 128, and increase them when num_blocks
        # is too large.
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (num_valid_tokens + block_size_m - 1) / block_size_m + \
            num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and \
                block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if num_m_blocks <= 16 and size_k % (block_size_k * 2) == 0 and \
                size_k % (block_size_k * 2) == 0 and block_size_k <= 512 and \
                num_blocks >= 512:
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            # The kernel performance got much better with BLOCK_SIZE_N=1024
            # when num_blocks is large, event when N is small.
            # Not sure why, maybe it force the CUDA SM process only one block
            # at the same time.
            block_size_n = 1024

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def should_moe_wna16_use_cuda(num_valid_tokens: int, group_size: int,
                              num_experts: int, bit: int):
    # return bit == 4 and group_size in [32, 64, 128] and \
    #    num_valid_tokens / num_experts <= 6
    #暂时为False
    return True

def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
    is_bottom: bool = False,
) -> Dict[str, int]:
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "COMBINE_SCALE_LOAD": False,
            "num_warps": 4,
            "num_stages": 3,
        }
    elif dtype in ["int4_w4a16", "fp4_w4a16", "int8_w8a16"] and block_shape is not None:
        # moe wna16 kernels
        # only set BLOCK_SIZE_M
        # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
        bit = 4 if dtype in ("int4_w4a16", "fp4_w4a16") else 8
        use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk,
                                                       block_shape[1], E, bit)
        if use_moe_wna16_cuda:
            config = {"BLOCK_SIZE_M": min(16, M)}
        elif M <= 20:
            config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False}
        elif M <= 40:
            config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False}
        else:
            config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False}
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "COMBINE_SCALE_LOAD": False,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "COMBINE_SCALE_LOAD": False,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
    is_bottom: bool = False,
    use_moe_wna16_cuda: bool = False,
):
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        if dtype == "int4_w4a16":
            N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k, is_bottom,use_moe_wna16_cuda)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype,
                                        is_marlin, block_shape, is_bottom)
    return config

_FP8_W8A8_PREFILL_BM_CHOICES = {32, 48, 64, 80, 96, 112, 128}
_FP8_W8A8_PREFILL_NLOOP_CHOICES = {1, 2, 3, 4, 5}
_FP8_W8A8_PREFILL_MODE_BASE = 80000
_FP8_W8A8_PREFILL_MODE_BASE_CHOICES = {70000, 80000, 90000, 100000}
_FP8_W8A8_DECODE_MODE_BASE = 60000
_FP8_W8A8_DECODE_MODE_BASE_CHOICES = {60000}
_FP8_W8A8_MODE_BASE_CHOICES = (
    _FP8_W8A8_DECODE_MODE_BASE_CHOICES | _FP8_W8A8_PREFILL_MODE_BASE_CHOICES
)


def _strip_fp8_w8a8_mode_base(mode: int) -> int:
    for mode_base in _FP8_W8A8_MODE_BASE_CHOICES:
        if mode_base <= mode < mode_base + 10000:
            return mode - mode_base
    return mode


def _encode_fp8_w8a8_mode(config: dict, mode_base: int, mode_suffix: int) -> dict:
    if mode_base not in _FP8_W8A8_MODE_BASE_CHOICES:
        raise ValueError(
            f"Unsupported fp8_w8a8 mode base={mode_base}; "
            f"supported values are {sorted(_FP8_W8A8_MODE_BASE_CHOICES)}"
        )

    config = dict(config)
    config["MODE"] = mode_base + mode_suffix
    return config


def _maybe_encode_fp8_w8a8_decode_mode(
    config: dict,
    dtype: Optional[str],
    block_shape: Optional[List[int]],
    key_selected: int,
    is_bottom: bool,
) -> dict:
    if dtype != "fp8_w8a8" or block_shape is not None or key_selected > 512:
        return config

    final_mode = config.get("MODE")
    if final_mode is not None and int(final_mode) >= _FP8_W8A8_DECODE_MODE_BASE:
        return config

    decode_mode = config.get("DECODE_MODE", final_mode)
    if decode_mode is None:
        return config

    decode_mode = _strip_fp8_w8a8_mode_base(int(decode_mode))

    mode_base = int(config.get("MODE_BASE", config.get("DECODE_MODE_BASE", _FP8_W8A8_DECODE_MODE_BASE)))
    if mode_base not in _FP8_W8A8_DECODE_MODE_BASE_CHOICES:
        raise ValueError(
            f"Unsupported fp8_w8a8 decode mode base={mode_base}; "
            f"supported values are {sorted(_FP8_W8A8_DECODE_MODE_BASE_CHOICES)}"
        )
    config = _encode_fp8_w8a8_mode(config, mode_base, decode_mode)
    stage_nloop_key = "N_LOOP2" if is_bottom else "N_LOOP1"
    config[stage_nloop_key] = int(config.get(stage_nloop_key, config.get("N_LOOP", 1)))
    config["DECODE_MODE"] = decode_mode
    config["DECODE_MODE_BASE"] = mode_base
    return config


def _maybe_encode_fp8_w8a8_prefill_mode(
    config: dict,
    dtype: Optional[str],
    block_shape: Optional[List[int]],
    key_selected: int,
    is_bottom: bool,
) -> dict:
    if dtype != "fp8_w8a8" or block_shape is not None or key_selected <= 512:
        return config

    final_mode = config.get("MODE")
    if final_mode is not None and int(final_mode) >= min(_FP8_W8A8_PREFILL_MODE_BASE_CHOICES):
        return config

    block_size_m = config.get("BLOCK_SIZE_M")
    if block_size_m not in _FP8_W8A8_PREFILL_BM_CHOICES:
        return config

    stage_nloop_key = "N_LOOP2" if is_bottom else "N_LOOP1"
    n_loop = config.get(stage_nloop_key, config.get("N_LOOP"))
    if n_loop is None:
        return config

    n_loop = int(n_loop)
    if n_loop not in _FP8_W8A8_PREFILL_NLOOP_CHOICES:
        raise ValueError(
            f"Unsupported fp8_w8a8 prefill N_LOOP={n_loop}; "
            f"supported values are {sorted(_FP8_W8A8_PREFILL_NLOOP_CHOICES)}"
        )

    mode_base = int(config.get("MODE_BASE", config.get("PREFILL_MODE_BASE", _FP8_W8A8_PREFILL_MODE_BASE)))
    if mode_base not in _FP8_W8A8_PREFILL_MODE_BASE_CHOICES:
        raise ValueError(
            f"Unsupported fp8_w8a8 prefill mode base={mode_base}; "
            f"supported values are {sorted(_FP8_W8A8_PREFILL_MODE_BASE_CHOICES)}"
        )
    mode_suffix = int(block_size_m) * 10 + n_loop
    config = _encode_fp8_w8a8_mode(config, mode_base, mode_suffix)
    config[stage_nloop_key] = n_loop
    config["PREFILL_MODE_BASE"] = mode_base
    return config

def try_get_optimal_moe_config_marlin(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
    is_bottom: bool = False,
    use_moe_wna16_cuda: bool = False,
):
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape

        if dtype == "w16a16":
            logical_n = w2_shape[1] if is_bottom else w1_shape[1]
            logical_k = w2_shape[2] if is_bottom else w1_shape[2]
            configs = get_moe_configs_marlin(
                E,
                logical_n,
                dtype,
                is_bottom=is_bottom,
                use_moe_wna16_cuda=use_moe_wna16_cuda,
                K=logical_k,
            )
            if configs:
                key_selected = min(configs.keys(), key=lambda x: abs(x - M))
                config = dict(configs[key_selected])
                config["key_selected"] = key_selected
            else:
                logger.warning("W16A16 Marlin config fallback for M=%s", M)
                config = get_default_config(
                    M, E, logical_n, logical_k, top_k, dtype,
                    is_marlin, block_shape, is_bottom)
            return config

        logical_k = None
        if dtype in ("int8_w4a8", "fp4_w4a8"):
            logical_k = w2_shape[1]
        elif dtype == "fp8_w8a8" and block_shape is None:
            logical_k = w2_shape[2] if is_bottom else w1_shape[2]

        if dtype in ("int4_w4a16", "fp4_w4a16", "int8_w4a8", "fp4_w4a8"):
            N = N * 2

        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs_marlin(
            E, N, dtype, block_n, block_k, is_bottom, use_moe_wna16_cuda,
            K=logical_k)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            key_selected = min(configs.keys(), key=lambda x: abs(x - M))
            config = dict(configs[key_selected])
            config = _maybe_encode_fp8_w8a8_decode_mode(
                config, dtype, block_shape, key_selected, is_bottom)
            config = _maybe_encode_fp8_w8a8_prefill_mode(
                config, dtype, block_shape, key_selected, is_bottom)
            config["key_selected"] = key_selected
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype,
                                        is_marlin, block_shape, is_bottom)
    return config


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    warmup:int = 0,
    rep:int = 1,
    renormalize: bool = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

    # fused_moe_times = []

    # with torch.inference_mode():
    fn = lambda: moe_c_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    fn()
            # ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        # ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        # fused_moe_times.append(ms)
    # Calculate statistics (skip warmup)
    # fused_moe_times = fused_moe_times[warmup:]
    # fused_moe_avg = ms #statistics.mean(fused_moe_times)  # ms

    # topk_weights_size_bytes = topk_weights.numel() * topk_weights.element_size()
    # topk_ids_size_bytes = topk_ids.numel() * topk_ids.element_size()
    # token_expert_indicies_size_bytes = token_expert_indicies.numel() * token_expert_indicies.element_size()
    # gating_output_size_bytes = gating_output.numel() * gating_output.element_size()
    # data_size_bytes =  topk_weights_size_bytes + topk_ids_size_bytes + token_expert_indicies_size_bytes + gating_output_size_bytes # 总字节数
    # data_size_gb = data_size_bytes / (1024 *1024*1024)    # 转为GB
    # bandwidth = (data_size_gb*1000) / fused_moe_avg   # GB/s

    # del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(dynamic=True, backend=get_compile_backend)
def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def get_config_dtype_str(dtype: torch.dtype,
                         use_int4_w4a16: Optional[bool] = False,
                         use_fp4_w4a16: Optional[bool] = False,
                         use_fp4_w4a8: Optional[bool] = False,
                         use_int8_w8a16: Optional[bool] = False,
                         use_fp8_w8a8: Optional[bool] = False,
                         use_int8_w8a8: Optional[bool] = False,
                         use_int8_w4a8: Optional[bool] = False,
                         use_w16a16: Optional[bool] = False):
    if use_w16a16:
        return "w16a16"
    elif use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int8_w4a8:
        return "int8_w4a8"
    elif use_fp4_w4a8:
        return "fp4_w4a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_fp4_w4a16:
        return "fp4_w4a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def inplace_fused_experts(hidden_states: torch.Tensor,
                        w1: torch.Tensor,
                        w2: torch.Tensor,
                        topk_weights: torch.Tensor,
                        topk_ids: torch.Tensor,
                        MODE1: int,
                        MODE2: int,
                        BM: int,
                        BN: int,
                        BK: int,
                        kloops: int,
                        nloops: int,
                        BN2: int ,
                        BK2: int ,
                        kloops2: int,
                        nloops2: int,
                        activation: Optional[str] = None,
                        is_gated: Optional[bool] = None,
                        use_fp8_w8a8: bool = False,
                        use_int8_w8a8: bool = False,
                        use_int8_w4a8: bool = False,
                        use_fp4_w4a8: bool = False,
                        use_int8_w8a16: bool = False,
                        use_int4_w4a16: bool = False,
                        use_fp4_w4a16: bool = False,
                        use_int4_w4a16_base: bool = False,
                        use_w16a16: bool = False,
                        global_num_experts: int = -1,
                        expert_map: Optional[torch.Tensor] = None,
                        w1_scale: Optional[torch.Tensor] = None,
                        w2_scale: Optional[torch.Tensor] = None,
                        w1_zp: Optional[torch.Tensor] = None,
                        w2_zp: Optional[torch.Tensor] = None,
                        a1_scale: Optional[torch.Tensor] = None,
                        a2_scale: Optional[torch.Tensor] = None,
                        block_shape: Optional[List[int]] = None,
                        routed_scaling_factor: Optional[float] = 1.0,
                        gemm1_alpha: Optional[float] = None,
                        gemm1_limit: Optional[float] = None,
                        compute_dtype: Optional[torch.dtype] = None,
                        w16a16_config: Optional[List[int]] = None,
                        fp8_w8a8_config: Optional[List[int]] = None) -> None:
    if activation is None:
        activation = "silu"



    use_fp4_w4a8_groupwise = _check_moec_quant_legal(
        use_int4_w4a16=use_int4_w4a16,
        use_fp4_w4a16=use_fp4_w4a16,
        use_int8_w4a8=use_int8_w4a8,
        use_fp4_w4a8=use_fp4_w4a8,
        block_shape=block_shape,
    )
    if (use_w16a16 or use_int4_w4a16 or use_fp4_w4a16 or (use_int8_w8a8 and block_shape == None) or (use_fp8_w8a8 and block_shape == None) or (use_int8_w8a16 and block_shape == None) or (use_int8_w4a8 and block_shape == None) or (use_fp4_w4a8 and block_shape == None) or use_fp4_w4a8_groupwise):

        fused_experts_impl_marlin(hidden_states, w1, w2,  topk_weights, topk_ids, MODE1, MODE2, BM,
                            True, activation,is_gated, use_fp8_w8a8, use_int8_w8a8,use_int8_w4a8, use_fp4_w4a8, use_int8_w8a16,
                            use_int4_w4a16, use_fp4_w4a16, use_int4_w4a16_base, use_w16a16, global_num_experts, expert_map,
                            w1_scale, w2_scale, w1_zp, w2_zp, a1_scale,
                            a2_scale, block_shape, routed_scaling_factor,gemm1_alpha,gemm1_limit,
                            compute_dtype=compute_dtype,
                            w16a16_config=w16a16_config,
                            fp8_w8a8_config=fp8_w8a8_config)
    else:
        fused_experts_impl(hidden_states, w1, w2,  topk_weights, topk_ids ,BM,BN,BK,kloops, nloops,BN2,
                            BK2,kloops2,nloops2,True,
                        activation,  use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
                        use_int4_w4a16, use_int4_w4a16_base, global_num_experts, expert_map,
                        w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale,
                        block_shape, routed_scaling_factor)


def inplace_fused_experts_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        routed_scaling_factor: Optional[float] = 1.0) -> None:
    pass



def outplace_fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        MODE1: int,
        MODE2: int,
        BM: int,
        BN: int,
        BK: int,
        kloops: int ,
        nloops: int,
        BN2: int ,
        BK2: int ,
        kloops2: int,
        nloops2: int,
        activation: Optional[str] = None,
        is_gated: Optional[bool] = None,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w4a8: bool = False,
        use_fp4_w4a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        use_fp4_w4a16: bool = False,
        use_int4_w4a16_base: bool = False,
        use_w16a16: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        routed_scaling_factor: Optional[float] = 1.0,
        gemm1_alpha: Optional[float] = None,
        gemm1_limit: Optional[float] = None,
        compute_dtype: Optional[torch.dtype] = None,
        w16a16_config: Optional[List[int]] = None,
        fp8_w8a8_config: Optional[List[int]] = None) -> torch.Tensor:
    if activation is None:
        activation = "silu"


    use_fp4_w4a8_groupwise = _check_moec_quant_legal(
        use_int4_w4a16=use_int4_w4a16,
        use_fp4_w4a16=use_fp4_w4a16,
        use_int8_w4a8=use_int8_w4a8,
        use_fp4_w4a8=use_fp4_w4a8,
        block_shape=block_shape,
    )
    if (use_w16a16 or use_int4_w4a16 or use_fp4_w4a16 or (use_int8_w8a8 and block_shape == None) or (use_fp8_w8a8 and block_shape == None) or (use_int8_w8a16 and block_shape == None) or (use_int8_w4a8 and block_shape == None) or (use_fp4_w4a8 and block_shape == None) or use_fp4_w4a8_groupwise):
        return fused_experts_impl_marlin(hidden_states, w1, w2,  topk_weights, topk_ids, MODE1, MODE2, BM,
                            False, activation,is_gated, use_fp8_w8a8, use_int8_w8a8,use_int8_w4a8, use_fp4_w4a8, use_int8_w8a16,
                            use_int4_w4a16, use_fp4_w4a16, use_int4_w4a16_base, use_w16a16, global_num_experts, expert_map,
                            w1_scale, w2_scale, w1_zp, w2_zp, a1_scale,
                            a2_scale, block_shape, routed_scaling_factor,gemm1_alpha,gemm1_limit,
                            compute_dtype=compute_dtype,
                            w16a16_config=w16a16_config,
                            fp8_w8a8_config=fp8_w8a8_config)


    return fused_experts_impl(hidden_states, w1, w2,  topk_weights, topk_ids,BM,BN,BK,kloops,nloops,BN2,
                                BK2,kloops2,nloops2,
                              False, activation, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
                              use_int4_w4a16, use_int4_w4a16_base, global_num_experts, expert_map,
                              w1_scale, w2_scale, w1_zp, w2_zp, a1_scale,
                              a2_scale, block_shape, routed_scaling_factor)


def outplace_fused_experts_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        routed_scaling_factor: Optional[float] = 1.0,
        *args,
        **kwargs) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@perftest(num_warmup=5, num_iters=10,testGraph=False)
def moe_c_fused_experts_bench(hidden_states: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                MODE1: int = 1,
                MODE2: int = 1,
                BM: int = 1,
                BN: int = 1,
                BK: int = 1,
                kloops: int = 1,
                nloops: int = 1,
                BN2: int = 1,
                BK2: int  = 1,
                kloops2: int = 1,
                nloops2: int = 1,
                inplace: bool = False,
                activation: Optional[str] = None,
                use_fp8_w8a8: bool = False,
                use_int8_w8a8: bool = False,
                use_int8_w4a8: bool = False,
                use_fp4_w4a8: bool = False,
                use_int8_w8a16: bool = False,
                use_int4_w4a16: bool = False,
                use_fp4_w4a16: bool = False,
                use_int4_w4a16_base: bool = False,
                use_w16a16: bool = False,
                global_num_experts: int = -1,
                expert_map: Optional[torch.Tensor] = None,
                w1_scale: Optional[torch.Tensor] = None,
                w2_scale: Optional[torch.Tensor] = None,
                w1_zp: Optional[torch.Tensor] = None,
                w2_zp: Optional[torch.Tensor] = None,
                a1_scale: Optional[torch.Tensor] = None,
                a2_scale: Optional[torch.Tensor] = None,
                block_shape: Optional[List[int]] = None,
                w16a16_config: Optional[List[int]] = None,
                fp8_w8a8_config: Optional[List[int]] = None) -> torch.Tensor:

    return moe_c_fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weights,
                         topk_ids,
                         MODE1,
                         MODE2,
                         BM ,BN,BK,
                         kloops,nloops,
                         BN2,
                         BK2,
                         kloops2,
                         nloops2,
                         inplace=inplace,
                         activation=activation,
                         use_fp8_w8a8=use_fp8_w8a8,
                         use_int8_w8a8=use_int8_w8a8,
                         use_int8_w4a8=use_int8_w4a8,
                         use_fp4_w4a8=use_fp4_w4a8,
                         use_int8_w8a16=use_int8_w8a16,
                         use_int4_w4a16=use_int4_w4a16,
                         use_fp4_w4a16=use_fp4_w4a16,
                         use_int4_w4a16_base=use_int4_w4a16_base,
                         use_w16a16=use_w16a16,
                         global_num_experts=global_num_experts,
                         expert_map=expert_map,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         w1_zp=w1_zp,
                         w2_zp=w2_zp,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale,
                         block_shape=block_shape,
                         w16a16_config=w16a16_config,
                         fp8_w8a8_config=fp8_w8a8_config)

@torch_compile_guard(gen_fake=outplace_fused_experts_fake)
def moe_c_fused_experts(hidden_states: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                MODE1: int = 1,
                MODE2: int = 1,
                BM: int = 1,
                BN: int = 1,
                BK: int = 1,
                kloops: int = 1,
                nloops: int = 1,
                BN2: int = 1,
                BK2: int  = 1,
                kloops2: int = 1,
                nloops2: int = 1,
                inplace: bool = False,
                activation: Optional[str] = None,
                is_gated: Optional[bool] = None,
                use_fp8_w8a8: bool = False,
                use_int8_w8a8: bool = False,
                use_int8_w4a8: bool = False,
                use_fp4_w4a8: bool = False,
                use_int8_w8a16: bool = False,
                use_int4_w4a16: bool = False,
                use_fp4_w4a16: bool = False,
                use_int4_w4a16_base: bool = False,
                use_w16a16: bool = False,
                global_num_experts: int = -1,
                expert_map: Optional[torch.Tensor] = None,
                w1_scale: Optional[torch.Tensor] = None,
                w2_scale: Optional[torch.Tensor] = None,
                w1_zp: Optional[torch.Tensor] = None,
                w2_zp: Optional[torch.Tensor] = None,
                a1_scale: Optional[torch.Tensor] = None,
                a2_scale: Optional[torch.Tensor] = None,
                block_shape: Optional[List[int]] = None,
                routed_scaling_factor: Optional[float] = 1.0,
                gemm1_alpha: Optional[float] = None,
                gemm1_limit: Optional[float] = None,
                compute_dtype: Optional[torch.dtype] = None,
                w16a16_config: Optional[List[int]] = None,
                fp8_w8a8_config: Optional[List[int]] = None,
                ) -> torch.Tensor:

    # assert  not (use_int8_w4a8 and hidden_states.shape[0] < 1024) , "only support M >= 1024"

    # print("*******************************use_int8_w4a8",use_int8_w4a8)
    if activation is None:
        activation = 'silu'
    if inplace:

        inplace_fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids,MODE1,MODE2,BM,BN,BK,kloops,nloops,BN2,
            BK2,kloops2,nloops2,activation,is_gated,
            use_fp8_w8a8, use_int8_w8a8, use_int8_w4a8,use_fp4_w4a8, use_int8_w8a16, use_int4_w4a16, use_fp4_w4a16, use_int4_w4a16_base, use_w16a16, global_num_experts,
            expert_map, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale,
            block_shape, routed_scaling_factor,gemm1_alpha,gemm1_limit,
            compute_dtype=compute_dtype,
            w16a16_config=w16a16_config,
            fp8_w8a8_config=fp8_w8a8_config)
        # print("hidden_states",hidden_states)
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids,MODE1,MODE2,BM,BN,BK,kloops,nloops,BN2,
            BK2,kloops2,nloops2,activation,is_gated,
            use_fp8_w8a8, use_int8_w8a8,use_int8_w4a8, use_fp4_w4a8, use_int8_w8a16, use_int4_w4a16, use_fp4_w4a16, use_int4_w4a16_base, use_w16a16, global_num_experts,
            expert_map, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale,
            block_shape, routed_scaling_factor,gemm1_alpha,gemm1_limit,
            compute_dtype=compute_dtype,
            w16a16_config=w16a16_config,
            fp8_w8a8_config=fp8_w8a8_config)


# def moe_sum(input_tensor: torch.tensor, output_tensor: torch.tensor, ):
#     torch.ops.sgl_kernel.moe_sum.default(input_tensor, output_tensor, )



def fused_experts_impl_channelwise_w8a8(hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    w1_new: torch.Tensor,
                    w2_new: torch.Tensor,
                    topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor,
                    BM: int,
                    BN: int,
                    BK: int,
                    kloops: int,
                    nloops: int,
                    BN2: int ,
                    BK2: int ,
                    kloops2: int,
                    nloops2: int,
                    inplace: bool = False,
                    activation: str = "silu",
                    use_fp8_w8a8: bool = False,
                    use_int8_w8a8: bool = False,
                    use_int8_w8a16: bool = False,
                    use_int4_w4a16: bool = False,
                    use_int4_w4a16_base: bool = False,
                    global_num_experts: int = -1,
                    expert_map: Optional[torch.Tensor] = None,
                    w1_scale: Optional[torch.Tensor] = None,
                    w2_scale: Optional[torch.Tensor] = None,
                    w1_zp: Optional[torch.Tensor] = None,
                    w2_zp: Optional[torch.Tensor] = None,
                    a1_scale: Optional[torch.Tensor] = None,
                    a2_scale: Optional[torch.Tensor] = None,
                    block_shape: Optional[List[int]] = None):

        m = hidden_states.shape[0]
        topk = topk_ids.shape[1]

        e, n1, _ = w1.shape
        if global_num_experts == -1:
            global_num_experts = e
        per_channel_quant = True


        if inplace:
            out_hidden_states = hidden_states
        else:

            out_hidden_states = torch.empty_like(hidden_states)



        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(topk_ids, 16, global_num_experts, expert_map)
                )


        qinput1, qa1_scale = moe_kernel_prepare_input(
                    A=hidden_states,
                    B=w1,
                    A_scale=None,
                    B_scale=w1_scale,
                    use_fp8_w8a8=False,
                    use_int8_w8a8=True,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    use_fp4_w4a16=False,
                    per_channel_quant=per_channel_quant,
                    block_shape=None
                )

        cache13 = torch.empty(m * topk * max(n1, w2.shape[1]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
        intermediate_cache1 = cache13[:m * topk * n1].view(
            (m, topk_ids.shape[1], n1))
        intermediate_cache3 = cache13[:m * topk * w2.shape[1]].view(
            (m, topk_ids.shape[1], w2.shape[1]))


        # # intermediate_cache1 = torch.zeros((m, topk, n1), device=hidden_states.device, dtype=hidden_states.dtype)


        moe_c_moe_gemm_marlin_w8a8(qinput1, w1_new, intermediate_cache1, qa1_scale, w1_scale, None,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded,topk, 54, 1)
        intermediate_cache2 = torch.empty((m * topk, n1 // 2), device=hidden_states.device, dtype=hidden_states.dtype)



        # # torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, n1))

        moe_c_silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, n1))

        # torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, n1))
        # print("moe_kernel_prepare_input************************************")
        # print(intermediate_cache1)
        # print(intermediate_cache2)
        # print(w2)
        # intermediate_cache2 = intermediate_cache2 /51380224
        # print(intermediate_cache2)
        # start = time.perf_counter()

        qinput2, qa2_scale = moe_kernel_prepare_input(
            A=intermediate_cache2,
            B=w2,
            A_scale=None,
            B_scale=w2_scale,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            use_fp4_w4a16=False,
            per_channel_quant=per_channel_quant,
            block_shape=None
        )



        moe_c_moe_gemm_marlin_w8a8(qinput2, w2_new, intermediate_cache3, qa2_scale, w2_scale, topk_weights,
                                            sorted_token_ids, expert_ids, num_tokens_post_padded, 1, 54, 1)


        # print(intermediate_cache3)
        mode_use_triton_moe_sum = out_hidden_states.dtype == torch.float16 or  \
                                  out_hidden_states.dtype == torch.bfloat16 or \
                                  out_hidden_states.dtype == torch.float32

        mode_use_triton_moe_sum = True

        if mode_use_triton_moe_sum:

            triton_moe_sum_noaiter(intermediate_cache3.view(*intermediate_cache3.shape),
                           out_hidden_states)
        else:
            moe_c_moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,topk_ids)


        return out_hidden_states


def fused_experts_impl_marlin(hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor,
                    MODE1: int,
                    MODE2: int,
                    BM: int,
                    inplace: bool = False,
                    activation: str = "silu",
                    is_gated: Optional[bool] = None,
                    use_fp8_w8a8: bool = False,
                    use_int8_w8a8: bool = False,
                    use_int8_w4a8: bool = False,
                    use_fp4_w4a8: bool = False,
                    use_int8_w8a16: bool = False,
                    use_int4_w4a16: bool = False,
                    use_fp4_w4a16: bool = False,
                    use_int4_w4a16_base: bool = False,
                    use_w16a16: bool = False,
                    global_num_experts: int = -1,
                    expert_map: Optional[torch.Tensor] = None,
                    w1_scale: Optional[torch.Tensor] = None,
                    w2_scale: Optional[torch.Tensor] = None,
                    w1_zp: Optional[torch.Tensor] = None,
                    w2_zp: Optional[torch.Tensor] = None,
                    a1_scale: Optional[torch.Tensor] = None,
                    a2_scale: Optional[torch.Tensor] = None,
                    block_shape: Optional[List[int]] = None,
                    routed_scaling_factor: Optional[float] = 1.0,
                    gemm1_alpha: Optional[float] = None,
                    gemm1_limit: Optional[float] = None,
                    compute_dtype: Optional[torch.dtype] = None,
                    w16a16_config: Optional[List[int]] = None,
                    fp8_w8a8_config: Optional[List[int]] = None):

    activation, is_gated = _normalize_activation_and_gate(activation, is_gated)

    prequantized_input = _is_prequantized_activation(
        hidden_states.dtype, a1_scale)
    if prequantized_input:
        assert use_fp8_w8a8 or use_int8_w8a8 or use_int8_w4a8 or use_fp4_w4a8, (
            "pre-quantized activation requires use_fp8_w8a8, "
            "use_int8_w8a8, use_int8_w4a8, or use_fp4_w4a8")
    else:
        assert hidden_states.dtype in _FLOAT_MOE_DTYPES, (
            f"hidden_states must be fp16/bf16/fp32, got {hidden_states.dtype}")

    compute_dtype = _resolve_moe_compute_dtype(
        hidden_states.dtype,
        compute_dtype,
        prequantized=prequantized_input,
    )

    # Check constraints.
    fp8_original_k = None
    fp8_padded_k = None
    if use_fp8_w8a8 and fp8_w8a8_config is not None:
        fp8_original_k = int(fp8_w8a8_config[0])
        fp8_padded_k = int(fp8_w8a8_config[1])
    w16a16_original_k = None
    w16a16_padded_k = None
    if use_w16a16 and w16a16_config is not None and len(w16a16_config) >= 8:
        w16a16_original_k = int(w16a16_config[6])
        w16a16_padded_k = int(w16a16_config[7])

    if use_int4_w4a16 or use_fp4_w4a16 or use_int8_w4a8 or use_fp4_w4a8:
        assert hidden_states.shape[1] // 2 == w1.shape[
            2], "Hidden size mismatch"
    elif use_fp8_w8a8 and fp8_padded_k is not None:
        assert hidden_states.shape[1] <= fp8_original_k, "Hidden size mismatch"
        assert w1.shape[2] == fp8_padded_k, "Padded W1 K mismatch"
        assert w2.shape[1] == fp8_padded_k, "Padded W2 output mismatch"
    elif use_w16a16 and w16a16_padded_k is not None:
        assert hidden_states.shape[1] <= w16a16_original_k, "Hidden size mismatch"
        assert w1.shape[2] == w16a16_padded_k, "Padded W1 K mismatch"
        assert w2.shape[1] == w16a16_padded_k, "Padded W2 output mismatch"
        assert topk_ids.shape[1] == 1, "Padded W16A16 MOE_C only supports topk=1"
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"

    num_tokens, _ = hidden_states.shape
    model_dim = hidden_states.shape[1]
    E = w1.shape[0]
    N = w1.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    CHUNK_SIZE = 32768

    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a8=use_int8_w8a8,
                                        use_int8_w4a8=use_int8_w4a8,
                                        use_fp4_w4a8=use_fp4_w4a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int4_w4a16=use_int4_w4a16,
                                        use_fp4_w4a16=use_fp4_w4a16,
                                        use_w16a16=use_w16a16,
                                        dtype=compute_dtype)

    get_config_func = functools.partial(
        try_get_optimal_moe_config_marlin,
        w1.shape,
        w2.shape,
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )
    # use_moe_wna16_cuda = should_moe_wna16_use_cuda(
    #         num_valid_tokens=topk_ids.numel(),
    #         group_size=block_shape[1] if block_shape else 0,
    #         num_experts=w1.shape[0],
    #         bit=4 if use_int4_w4a16 else 8)
    use_moe_wna16_cuda = True

    def _align_w16a16_topk_ids(
        curr_topk_ids: torch.Tensor,
        curr_topk_weights: torch.Tensor,
        block_size_m: int,
    ):
        return moe_align_block_size_w16a16_ck(
            curr_topk_ids,
            curr_topk_weights,
            block_size_m,
            global_num_experts,
            expert_map,
        )

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, w2.shape[1]),
                          device=hidden_states.device,
                          dtype=compute_dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(
        (M, topk_ids.shape[1], N))
    intermediate_cache3 = cache13[:M * top_k_num * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]))

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=compute_dtype)

    compute_type = _torch_dtype_to_triton(compute_dtype)

    if inplace:
        assert not prequantized_input, (
            "inplace is not supported when hidden_states is pre-quantized")
        assert hidden_states.dtype == compute_dtype
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty(
            (num_tokens, hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=compute_dtype,
        )



    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if w16a16_padded_k is not None:
            padded_hidden_states = curr_hidden_states.new_zeros(
                (tokens_in_chunk, w16a16_padded_k)
            )
            padded_hidden_states[:, :model_dim].copy_(curr_hidden_states)
            curr_hidden_states = padded_hidden_states

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        if use_w16a16 and w16a16_config is not None:
            config, config2 = _make_w16a16_config_pair({
                "MODE": int(w16a16_config[0]),
                "BLOCK_SIZE_M": int(w16a16_config[1]),
                "DELTA": int(w16a16_config[2]),
                "key_selected": tokens_in_chunk,
            }, {
                "MODE": int(w16a16_config[3]),
                "BLOCK_SIZE_M": int(w16a16_config[4]),
                "DELTA": int(w16a16_config[5]),
                "key_selected": tokens_in_chunk,
            })
        elif use_w16a16:
            config = get_config_func(
                tokens_in_chunk,
                is_bottom=False,
                use_moe_wna16_cuda=use_moe_wna16_cuda,
            )
            config2 = get_config_func(
                tokens_in_chunk,
                is_bottom=True,
                use_moe_wna16_cuda=use_moe_wna16_cuda,
            )
            config, config2 = _make_w16a16_config_pair(config, config2)
        else:
            if use_moe_wna16_cuda and config_dtype in ["int4_w4a16","fp4_w4a16","int8_w8a8","fp8_w8a8","int8_w8a16", "int8_w4a8", "fp4_w4a8"]:
                config = get_config_func(tokens_in_chunk, is_bottom=False, use_moe_wna16_cuda=use_moe_wna16_cuda)
            else:
                config = get_config_func(tokens_in_chunk, is_bottom=False)
            config2 = None
        if use_fp4_w4a8:
            config2 = get_config_func(
                tokens_in_chunk,
                is_bottom=True,
                use_moe_wna16_cuda=use_moe_wna16_cuda,
            )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        curr_a1_scale = (a1_scale[begin_chunk_idx:end_chunk_idx]
                         if a1_scale is not None else None)
        curr_a2_scale = (a2_scale[begin_chunk_idx:end_chunk_idx]
                         if a2_scale is not None else None)

        find_best = os.environ.get("WHICH_TO_TEST")
        sorting_method = config.get("SORTING_METHOD")
        use_moe_align_sorting = (
            sorting_method == "moe_align"
            or (sorting_method is None and use_fp8_w8a8)
        )
        if use_w16a16:
            assert config2 is not None
            block_size_m1 = int(BM if find_best else config["BLOCK_SIZE_M"])
            block_size_m2 = int(BM if find_best else config2["BLOCK_SIZE_M"])
            assert block_size_m1 >= block_size_m2
            assert block_size_m1 % block_size_m2 == 0
            config["DELTA"] = 1
            config2["DELTA"] = block_size_m1 // block_size_m2
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                _align_w16a16_topk_ids(curr_topk_ids, curr_topk_weights, block_size_m1)
            )
            moe_c_sorted_token_ids = None
            if int(config["MODE"]) >= 1000 or int(config2["MODE"]) >= 1000:
                moe_c_sorted_token_ids = torch.empty_like(sorted_token_ids)
                moe_sorting_ck_ids_to_moe_c(
                    sorted_token_ids,
                    moe_c_sorted_token_ids,
                    tokens_in_chunk,
                    top_k_num,
                )
            sorted_token_ids2 = (
                moe_c_sorted_token_ids if int(config2["MODE"]) >= 1000 else sorted_token_ids
            )
            sorted_token_ids = (
                moe_c_sorted_token_ids if int(config["MODE"]) >= 1000 else sorted_token_ids
            )
        elif(find_best):
            if(use_int4_w4a16 or use_fp4_w4a16 or use_int8_w4a8 or use_fp4_w4a8 or use_int8_w8a16):
                sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(curr_topk_ids, BM,
                                        global_num_experts, expert_map))
            else:
                if (use_fp8_w8a8 or use_int8_w8a8) and use_moe_align_sorting:
                    sorted_token_ids, expert_ids, num_tokens_post_padded = (
                        moe_align_block_size(curr_topk_ids, BM,
                                            global_num_experts, expert_map))
                else:
                    sorted_token_ids,_,expert_ids, num_tokens_post_padded,_,_  =  moe_sorting_ck(
                                                                                    curr_topk_ids,
                                                                                    curr_topk_weights,
                                                                                    global_num_experts,
                                                                                    model_dim,
                                                                                    compute_dtype,
                                                                                    BM,
                                                                                    expert_mask=None,
                                                                                )

        else:
            if(use_int4_w4a16 or use_fp4_w4a16 or use_int8_w4a8 or use_fp4_w4a8 or use_int8_w8a16):
                sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(curr_topk_ids, config["BLOCK_SIZE_M"],
                                        global_num_experts, expert_map))
            else:
                if (use_fp8_w8a8 or use_int8_w8a8) and use_moe_align_sorting:
                    sorted_token_ids, expert_ids, num_tokens_post_padded = (
                        moe_align_block_size(curr_topk_ids, config["BLOCK_SIZE_M"],
                                            global_num_experts, expert_map))
                else:
                    sorted_token_ids,_,expert_ids, num_tokens_post_padded,_,_  =  moe_sorting_ck(
                                                                                    curr_topk_ids,
                                                                                    curr_topk_weights,
                                                                                    global_num_experts,
                                                                                    model_dim,
                                                                                    compute_dtype,
                                                                                    config["BLOCK_SIZE_M"],
                                                                                    expert_mask=None,
                                                                                )

        invoke_fused_moe_kernel_marlin(curr_hidden_states,
                                w1,
                                intermediate_cache1,
                                curr_a1_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                top_k_num,
                                int(config["MODE"]),
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w4a8=use_int8_w4a8,
                                use_fp4_w4a8=use_fp4_w4a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                use_fp4_w4a16=use_fp4_w4a16,
                                use_int4_w4a16_base=use_int4_w4a16_base,
                                use_w16a16=use_w16a16,
                                is_bottom = False,
                                key_selected=config.get("key_selected"),
                                block_shape=block_shape,
                                real_size_k=fp8_original_k if use_fp8_w8a8 else None)

        if activation == "silu" :
                moe_c_silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        elif activation == "situ":
                moe_c_situ_glu(
                        intermediate_cache2,
                        intermediate_cache1.view(-1, N),
                        beta1 = 4.0,
                        beta2= 25.0
                )

        else:
                _apply_activation(
                    activation=activation,
                    is_gated=is_gated,
                    activated_out=intermediate_cache2,
                    ffn1_out_2d=intermediate_cache1.view(-1, N),
                    gemm1_alpha=gemm1_alpha,
                    gemm1_limit=gemm1_limit,
                )
        

        if use_w16a16:
            assert config2 is not None
            config = config2
            sorted_token_ids_for_gemm2 = sorted_token_ids2
        elif use_fp4_w4a8 and config2 is not None:
            config = config2
            sorted_token_ids_for_gemm2 = sorted_token_ids
        else:
            if use_moe_wna16_cuda and config_dtype in ["int4_w4a16","fp4_w4a16","int8_w8a8","fp8_w8a8","int8_w8a16", "int8_w4a8", "fp4_w4a8"] :
                config = get_config_func(tokens_in_chunk,is_bottom=True,use_moe_wna16_cuda=use_moe_wna16_cuda)
            else:
                config = get_config_func(tokens_in_chunk, is_bottom=True)
            sorted_token_ids_for_gemm2 = sorted_token_ids


        invoke_fused_moe_kernel_marlin(intermediate_cache2,
                                w2,
                                intermediate_cache3,
                                curr_a2_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids_for_gemm2,
                                expert_ids,
                                num_tokens_post_padded,
                                True,
                                top_k_num,
                                int(config["MODE"]),
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w4a8=use_int8_w4a8,
                                use_fp4_w4a8=use_fp4_w4a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                use_fp4_w4a16=use_fp4_w4a16,
                                use_int4_w4a16_base=use_int4_w4a16_base,
                                use_w16a16=use_w16a16,
                                is_bottom = True,
                                key_selected=config.get("key_selected") ,
                                block_shape=block_shape,
                                real_size_k=None)
        mode_use_triton_moe_sum = out_hidden_states.dtype == torch.float16 or  \
                                  out_hidden_states.dtype == torch.bfloat16 or \
                                  out_hidden_states.dtype == torch.float32

        mode_use_triton_moe_sum = True

        if top_k_num == 1:
            moe_out = intermediate_cache3[:, 0, :out_hidden_states.shape[1]]
            out_hidden_states[begin_chunk_idx:end_chunk_idx].copy_(moe_out)
            if routed_scaling_factor != 1.0:
                out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor)
        elif mode_use_triton_moe_sum:
            # triton_moe_sum_noaiter(intermediate_cache3.view(*intermediate_cache3.shape),
            #                out_hidden_states[begin_chunk_idx:end_chunk_idx])
            triton_moe_sum_noaiter(intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states[begin_chunk_idx:end_chunk_idx] , routed_scaling_factor)
        else:
            moe_c_moe_sum_opt_v2(intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],routed_scaling_factor)



    return out_hidden_states
















def fused_experts_impl(hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor,
                    BM: int,
                    BN: int,
                    BK: int,
                    kloops: int,
                    nloops: int,
                    BN2: int ,
                    BK2: int ,
                    kloops2: int,
                    nloops2: int,
                    inplace: bool = False,
                    activation: str = "silu",
                    use_fp8_w8a8: bool = False,
                    use_int8_w8a8: bool = False,
                    use_int8_w8a16: bool = False,
                    use_int4_w4a16: bool = False,
                    use_int4_w4a16_base: bool = False,
                    global_num_experts: int = -1,
                    expert_map: Optional[torch.Tensor] = None,
                    w1_scale: Optional[torch.Tensor] = None,
                    w2_scale: Optional[torch.Tensor] = None,
                    w1_zp: Optional[torch.Tensor] = None,
                    w2_zp: Optional[torch.Tensor] = None,
                    a1_scale: Optional[torch.Tensor] = None,
                    a2_scale: Optional[torch.Tensor] = None,
                    block_shape: Optional[List[int]] = None,
                    routed_scaling_factor: Optional[float] = 1.0):

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[
            2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    CHUNK_SIZE = 32768

    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a8=use_int8_w8a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int4_w4a16=use_int4_w4a16,
                                        dtype=hidden_states.dtype)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )
    use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=topk_ids.numel(),
            group_size=block_shape[1],
            num_experts=w1.shape[0],
            bit=4 if use_int4_w4a16 else 8)
    if use_moe_wna16_cuda and config_dtype in ["int4_w4a16","int8_w8a8","fp8_w8a8","int8_w8a16"] :
        config = get_config_func(M,use_moe_wna16_cuda=use_moe_wna16_cuda)
    else:
        config = get_config_func(M)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, w2.shape[1]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(
        (M, topk_ids.shape[1], N))
    intermediate_cache3 = cache13[:M * top_k_num * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]))

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]


        find_best = os.environ.get("WHICH_TO_TEST")
        if(find_best):
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, BM,
                                    global_num_experts, expert_map))
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config["BLOCK_SIZE_M"],
                                    global_num_experts, expert_map))

        # if(use_int8_w8a8 and block_shape[0] == 1):

        invoke_fused_moe_kernel(curr_hidden_states,
                                w1,
                                w1, #需要修改 调用链路 只需要传递shuffle权重即可 w8a8_per_token已修改
                                intermediate_cache1,
                                a1_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                top_k_num,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                use_int4_w4a16_base=use_int4_w4a16_base,
                                BM = BM,
                                BN = BN,
                                BK = BK,
                                kloops = kloops,
                                nloops = nloops,
                                is_bottom = False,
                                block_shape=block_shape)

        if activation == "silu":
            moe_c_silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        # elif activation == "gelu":
        #     torch.ops._C.gelu_and_mul(intermediate_cache2,
        #                               intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=topk_ids.numel(),
            group_size=block_shape[1],
            num_experts=w2.shape[0],
            bit=4 if use_int4_w4a16 else 8)

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            config = get_config_func(tokens_in_chunk, is_bottom=True)
        else:
            if use_moe_wna16_cuda and config_dtype in ["int4_w4a16","int8_w8a8","fp8_w8a8","int8_w8a16"] :
                config = get_config_func(M,is_bottom=True,use_moe_wna16_cuda=use_moe_wna16_cuda)
            else:
                config = get_config_func(M, is_bottom=True)

        # intermediate_cache2 = torch.ones_like(intermediate_cache2)

        invoke_fused_moe_kernel(intermediate_cache2,
                                w2,
                                w2, #需要修改 调用链路 只需要传递shuffle权重即可 w8a8_per_token已修改
                                intermediate_cache3,
                                a2_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                True,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                use_int4_w4a16_base=use_int4_w4a16_base,
                                BM = BM,
                                BN = BN2,
                                BK = BK2,
                                kloops = kloops2,
                                nloops = nloops2,
                                is_bottom = True,
                                block_shape=block_shape)

        mode_use_triton_moe_sum = out_hidden_states.dtype == torch.float16 or  \
                                  out_hidden_states.dtype == torch.bfloat16 or \
                                  out_hidden_states.dtype == torch.float32

        mode_use_triton_moe_sum = True

        if mode_use_triton_moe_sum:
            triton_moe_sum_noaiter(intermediate_cache3.view(*intermediate_cache3.shape),
                           out_hidden_states[begin_chunk_idx:end_chunk_idx])
        else:
            moe_c_moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],curr_topk_ids)



    return out_hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    MODE1: int = 1,
    MODE2: int = 1,
    BM: int = 1,
    BN: int = 1,
    BK: int = 1,
    kloops: int = 1,
    nloops: int = 1,
    BN2: int = 1,
    BK2: int = 1,
    kloops2: int = 1,
    nloops2: int = 1,
    inplace: bool = False,
    activation: str = "silu",
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w4a8 : bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_fp4_w4a16: bool = False,
    use_int4_w4a16_base: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use matmul of int8 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
        from the global expert space to the local expert space of the expert
        parallel shard.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:

        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
    return moe_c_fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weights,
                         topk_ids,
                         MODE1,
                         MODE2,
                         BM ,BN,BK,
                         kloops,nloops,
                         BN2,
                         BK2,
                         kloops2,
                         nloops2,
                         inplace=inplace,
                         activation=activation,
                         use_fp8_w8a8=use_fp8_w8a8,
                         use_int8_w8a8=use_int8_w8a8,
                         use_int8_w4a8 = use_int8_w4a8,
                         use_int8_w8a16=use_int8_w8a16,
                         use_int4_w4a16=use_int4_w4a16,
                         use_fp4_w4a16=use_fp4_w4a16,
                         use_int4_w4a16_base=use_int4_w4a16_base,
                         global_num_experts=global_num_experts,
                         expert_map=expert_map,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         w1_zp=w1_zp,
                         w2_zp=w2_zp,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale,
                         block_shape=block_shape)
