# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Common helpers for standalone MOE operator tests."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F


logger = logging.getLogger("moe")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


dtypes = SimpleNamespace(
    fp16=torch.float16,
    bf16=torch.bfloat16,
    fp32=torch.float32,
    i32=torch.int32,
    i8=torch.int8,
    u8=torch.uint8,
)
if hasattr(torch, "float8_e4m3fn"):
    dtypes.fp8 = torch.float8_e4m3fn


def get_gfx() -> str:
    env_arch = os.environ.get("GPU_ARCHS") or os.environ.get("PYTORCH_ROCM_ARCH")
    if env_arch:
        return env_arch.split(";")[0].split(",")[0]
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return "gfx938"


class ActivationType:
    Silu = "silu"


_SCALAR_TYPES_ID_MAP = {}


class NanRepr(Enum):
    NONE = 0
    IEEE_754 = 1
    EXTD_RANGE_MAX_MIN = 2


@dataclass(frozen=True)
class ScalarType:
    exponent: int
    mantissa: int
    signed: bool
    bias: int
    _finite_values_only: bool = False
    nan_repr: NanRepr = NanRepr.IEEE_754

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)

    def is_floating_point(self) -> bool:
        return self.exponent != 0

    def is_integer(self) -> bool:
        return self.exponent == 0

    def is_signed(self) -> bool:
        return self.signed

    def has_bias(self) -> bool:
        return self.bias != 0

    def min(self) -> Union[int, float]:
        if self.is_floating_point():
            raise NotImplementedError("floating ScalarType min is not needed by local tests")
        return (-(1 << (self.size_bits - 1)) if self.is_signed() else 0) - self.bias

    def max(self) -> Union[int, float]:
        if self.is_floating_point():
            raise NotImplementedError("floating ScalarType max is not needed by local tests")
        return (1 << self.mantissa) - 1 - self.bias

    @property
    def id(self) -> int:
        val = 0
        offset = 0
        for member, bit_width in (
            (self.exponent, 8),
            (self.mantissa, 8),
            (self.signed, 1),
            (self.bias, 32),
            (self._finite_values_only, 1),
            (self.nan_repr.value, 8),
        ):
            val |= (int(member) & ((1 << bit_width) - 1)) << offset
            offset += bit_width
        _SCALAR_TYPES_ID_MAP[val] = self
        return val

    @classmethod
    def int_(cls, size_bits: int, bias: Optional[int]) -> "ScalarType":
        ret = cls(0, size_bits - 1, True, bias if bias else 0)
        ret.id
        return ret

    @classmethod
    def uint(cls, size_bits: int, bias: Optional[int]) -> "ScalarType":
        ret = cls(0, size_bits, False, bias if bias else 0)
        ret.id
        return ret


class scalar_types:
    int4 = ScalarType.int_(4, None)
    uint4 = ScalarType.uint(4, None)
    int8 = ScalarType.int_(8, None)
    uint8 = ScalarType.uint(8, None)
    uint4b8 = ScalarType.uint(4, 8)


def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: Optional[int],
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    assert quant_type.is_integer(), "Floating point quantization may work but has not been tested"
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided (-1 group_size is channelwise)"
    )

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape
    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values
    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
        else:
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    if group_size is not None and group_size < size_k:
        def reshape_w(tensor):
            tensor = tensor.reshape((group_size, -1, size_n))
            tensor = tensor.permute(1, 0, 2)
            return tensor.reshape((size_k, size_n)).contiguous()

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )


def torch_compile_guard(
    mutates_args: list[str] | str = "unknown",
    device: str = "cuda",
    calling_func_: Optional[Callable[..., Any]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
):
    del mutates_args, device, calling_func_, gen_fake

    def decorator(func):
        return func

    return decorator


def perftest(num_iters=101, num_warmup=2, testGraph=False, num_rotate_args=0, needTrace=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_warmup):
                func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                data = None
                for _ in range(num_iters):
                    data = func(*args, **kwargs)
                end.record()
                end.synchronize()
                avg_us = start.elapsed_time(end) * 1000.0 / max(num_iters, 1)
            else:
                start = time.perf_counter()
                data = None
                for _ in range(num_iters):
                    data = func(*args, **kwargs)
                avg_us = (time.perf_counter() - start) * 1_000_000.0 / max(num_iters, 1)
            return data, avg_us

        return wrapper

    return decorator


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05, msg="", printNum=8, printLog=True, perfModel=False):
    if perfModel:
        a = a.cpu().to(torch.float32)
        b = b.cpu().to(torch.float32)

    is_close = torch.isclose(a, b, rtol=rtol, atol=atol)
    if is_close.all():
        if printLog:
            logger.info(f"{msg}[checkAllclose atol={atol} rtol={rtol} passed]")
        return 0

    mask = ~is_close
    percent = (mask.sum() / a.numel()).item()
    if printLog:
        a_msked = a[mask]
        b_msked = b[mask]
        delta = (a_msked - b_msked).abs()
        logger.info(
            f"{msg}[checkAllclose atol={atol} rtol={rtol} mismatch] "
            f"error_ratio={percent:.1%}, max_abs_delta={delta.max()}"
        )
        if percent > tol_err_ratio:
            logger.info(f"a: {a_msked[:printNum]}")
            logger.info(f"b: {b_msked[:printNum]}")
            logger.info(f"delta: {delta[:printNum]}")
    return percent


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Print detailed tensor diff statistics and sample elements."""
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量（torch.Tensor）")
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"张量形状不匹配！ tensor1形状: {tensor1.shape}, tensor2形状: {tensor2.shape}"
        )
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
        print(f"警告：张量设备不一致，已将tensor2转移到{tensor1.device}")

    tensor1 = tensor1.to(torch.float32)
    tensor2 = tensor2.to(torch.float32)
    abs_diff = torch.abs(tensor1 - tensor2)
    denom = torch.maximum(torch.abs(tensor1), torch.abs(tensor2))
    rel_diff = abs_diff / (denom + 1e-12)
    match_mask = (abs_diff <= atol) | (rel_diff <= rtol)

    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    abs_diff_flat = abs_diff.flatten()
    match_mask_flat = match_mask.flatten()

    def get_indices_1d(mask: torch.Tensor) -> list[int]:
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

    def print_sample(name: str, indices_1d: list[int], max_samples: int = 3, elem_per_sample: int = 10) -> None:
        if not indices_1d:
            print(f"\n【{name}样本】无数据")
            return

        print(f"\n【{name}样本】（最多展示{max_samples}组，每组{elem_per_sample}个元素）")
        print("-" * 50)
        num_samples = min(max_samples, (len(indices_1d) + elem_per_sample - 1) // elem_per_sample)
        for index in range(num_samples):
            start = index * elem_per_sample
            sample_indices_1d = indices_1d[start:start + elem_per_sample]
            sample_coords = torch.unravel_index(torch.tensor(sample_indices_1d), tensor1.shape)
            sample_coords_list = list(zip(*[coord.tolist() for coord in sample_coords]))

            print(f"\n第{index + 1}组:")
            print(f"  原始多维坐标: {sample_coords_list}")
            print(f"  tensor1: {[round(tensor1_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  tensor2: {[round(tensor2_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  绝对差异: {[round(abs_diff_flat[idx].item(), 6) for idx in sample_indices_1d]}")

    print_sample("匹配", match_indices_1d, max_samples=2)
    print_sample("不匹配", mismatch_indices_1d, max_samples=3)
    print("\n" + "=" * 60)


def per_token_quant_int8(x: torch.Tensor, eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    x_f = x.to(torch.float32)
    scale = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.round(x_f / scale).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.to(torch.float32).contiguous()


def per_token_group_quant_int8(x: torch.Tensor, group_size: int, eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"last dim {x.shape[-1]} must be divisible by group_size {group_size}")
    x_f = x.to(torch.float32)
    grouped = x_f.reshape(*x_f.shape[:-1], -1, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8)
    return q.reshape_as(x).contiguous(), scale.squeeze(-1).contiguous()


def moe_kernel_prepare_input(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w4a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_fp4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    del B
    if use_int8_w8a8 or use_int8_w4a8 or use_fp8_w8a8:
        if B_scale is None:
            raise AssertionError("B_scale is required for quantized activation paths")
        if block_shape is None:
            if not per_channel_quant:
                raise AssertionError("int8 activation quantization requires channel-wise or block-wise mode")
            return per_token_quant_int8(A)
        if len(block_shape) != 2:
            raise AssertionError("block_shape must be [block_n, block_k]")
        return per_token_group_quant_int8(A, block_shape[1])

    if use_int8_w8a16 or use_int4_w4a16 or use_fp4_w4a16:
        if B_scale is None:
            raise AssertionError("B_scale is required for weight-only quantization paths")
        return A, A_scale

    if A_scale is not None or B_scale is not None:
        raise AssertionError("unquantized path should not pass A_scale/B_scale")
    return A, A_scale


def per_token_quant_hip(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    quant_dtype: torch.dtype = torch.int8,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del num_rows, num_rows_factor
    if scale is not None:
        raise ValueError("static per-token quantization scale is not supported")
    if quant_dtype is torch.int8:
        return per_token_quant_int8(x)

    x_f = x.to(torch.float32)
    finfo = torch.finfo(quant_dtype)
    scale = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / finfo.max
    q = (x_f / scale).clamp(finfo.min, finfo.max).to(quant_dtype)
    return q.contiguous(), scale.to(torch.float32).contiguous()


def pertoken_quant(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    scale_dtype: torch.dtype = torch.float32,
    quant_dtype: torch.dtype = torch.int8,
    dtypeMax: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.to(torch.float32)
    hidden_states = x if x_scale is None else x * x_scale
    if dtypeMax is None:
        try:
            dtypeMax = torch.finfo(quant_dtype).max
        except TypeError:
            dtypeMax = torch.iinfo(quant_dtype).max

    if scale is None:
        scale = hidden_states.abs().amax(dim=-1, keepdim=True) / dtypeMax
        scale[scale == 0] = 1

    quantized = hidden_states / scale
    if quant_dtype is torch.int8:
        quantized = quantized.round()
    return quantized.to(quant_dtype).contiguous(), scale.to(scale_dtype).contiguous()


_ACTIVATION_ALIASES = {
    "swiglu": "silu",
    "gelu_fast": "gelu_tanh",
    "gelu_pytorch_tanh": "gelu_tanh",
}
_SUPPORTED_ACTIVATIONS = {
    "silu",
    "silu_no_mul",
    "gelu",
    "gelu_no_mul",
    "gelu_tanh",
    "gelu_tanh_no_mul",
    "relu2",
    "relu2_no_mul",
    "situ",
    "swigluoai",
    "swiglustep",
}
_NO_MUL_ACTIVATIONS = {"silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul", "relu2", "relu2_no_mul"}
_GATED_ONLY_ACTIVATIONS = {"silu", "gelu", "gelu_tanh", "situ", "swigluoai", "swiglustep"}


def _normalize_activation_and_gate(
    activation: str,
    is_gated: Optional[bool],
) -> Tuple[str, bool]:
    activation_name = _ACTIVATION_ALIASES.get(activation.lower(), activation.lower())
    if activation_name not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}")
    if is_gated is None:
        is_gated = activation_name not in _NO_MUL_ACTIVATIONS
    if is_gated and activation_name in _NO_MUL_ACTIVATIONS:
        raise ValueError(f"Activation '{activation}' is non-gated but is_gated=True")
    if not is_gated and activation_name in _GATED_ONLY_ACTIVATIONS:
        raise ValueError(f"Activation '{activation}' requires gated mode")
    return activation_name, is_gated


def _apply_activation(
    activation: str,
    is_gated: bool,
    activated_out: torch.Tensor,
    ffn1_out_2d: torch.Tensor,
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
) -> None:
    x = ffn1_out_2d.to(torch.float32)
    if is_gated:
        gate, up = x.chunk(2, dim=-1)
        if activation == "silu":
            if gemm1_limit is not None:
                gate = gate.clamp(min=-gemm1_limit, max=gemm1_limit)
                up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
            alpha = 1.0 if gemm1_alpha is None else gemm1_alpha
            value = gate * torch.sigmoid(alpha * gate) * up
        elif activation == "gelu":
            value = F.gelu(gate) * up
        elif activation == "gelu_tanh":
            value = F.gelu(gate, approximate="tanh") * up
        elif activation == "swigluoai":
            alpha = 1.702 if gemm1_alpha is None else gemm1_alpha
            limit = 7.0 if gemm1_limit is None else gemm1_limit
            gate = gate.clamp(min=-limit, max=limit)
            up = up.clamp(min=-limit, max=limit)
            value = gate * torch.sigmoid(alpha * gate) * (up + 1)
        elif activation == "situ":
            alpha = 4.0 if gemm1_alpha is None else gemm1_alpha
            limit = 25.0 if gemm1_limit is None else gemm1_limit
            value = alpha * torch.tanh(gate / alpha) * torch.sigmoid(gate)
            value = value * (limit * torch.tanh(up / limit))
        elif activation == "swiglustep":
            limit = 7.0 if gemm1_limit is None else gemm1_limit
            value = F.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)
        else:
            raise ValueError(f"Unsupported gated activation: {activation}")
    elif activation in {"silu", "silu_no_mul"}:
        value = F.silu(x)
    elif activation in {"gelu", "gelu_no_mul"}:
        value = F.gelu(x)
    elif activation in {"gelu_tanh", "gelu_tanh_no_mul"}:
        value = F.gelu(x, approximate="tanh")
    elif activation in {"relu2", "relu2_no_mul"}:
        value = F.relu(x).square()
    else:
        raise ValueError(f"Unsupported non-gated activation: {activation}")
    activated_out.copy_(value.to(activated_out.dtype))


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    is_softmax: bool = True,
):
    from .moe_ops import topk_softmax

    m = hidden_states.shape[0]
    if topk_weights is None:
        topk_weights = torch.empty((m, topk), dtype=torch.float32, device=hidden_states.device)
    if topk_ids is None:
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device=hidden_states.device)
    token_expert_indices = torch.empty((m, topk), dtype=torch.int32, device=hidden_states.device)

    if is_softmax:
        topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output.float(), renormalize)
    else:
        scores = torch.sigmoid(gating_output.float())
        weights, ids = torch.topk(scores, k=topk, dim=-1)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        topk_weights.copy_(weights)
        topk_ids.copy_(ids.to(torch.int32))
    return topk_weights, topk_ids


def torch_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    *_,
    fc1_scale: Optional[torch.Tensor] = None,
    fc2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if w1_scale is None:
        w1_scale = fc1_scale
    if w2_scale is None:
        w2_scale = fc2_scale

    w1_ref = w1.to(torch.float32)
    w2_ref = w2.to(torch.float32)
    if w1_scale is not None:
        w1_ref = w1_ref * w1_scale.to(torch.float32)
    if w2_scale is not None:
        w2_ref = w2_ref * w2_scale.to(torch.float32)

    m, model_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    repeated = hidden_states.to(torch.float32).view(m, 1, model_dim).repeat(1, topk, 1).reshape(-1, model_dim)
    flat_topk_ids = topk_ids.reshape(-1)
    out = torch.zeros((m * topk, w2_ref.shape[1]), device=hidden_states.device, dtype=torch.float32)

    for expert in range(w1_ref.shape[0]):
        mask = flat_topk_ids == expert
        if mask.any():
            intermediate = repeated[mask] @ w1_ref[expert].transpose(0, 1)
            half = intermediate.shape[-1] // 2
            activated = F.silu(intermediate[:, :half]) * intermediate[:, half:]
            out[mask] = activated @ w2_ref[expert].transpose(0, 1)

    return (out.view(m, topk, -1) * topk_weights.view(m, topk, 1)).sum(dim=1)
