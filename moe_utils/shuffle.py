# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Weight and scale layout shuffle helpers for MOE_C operators."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _require_moe_c(moe_config) -> None:
    if getattr(moe_config, "solution_type", None) != "moe_c":
        raise NotImplementedError("Local shuffle currently supports only MOE_C")


def moe_shfl_weight(
    w1: Optional[torch.Tensor],
    w2: Optional[torch.Tensor],
    moe_config,
    block_shape: Optional[list] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    _require_moe_c(moe_config)
    quant_type = getattr(moe_config, "quant_type", None)

    def apply(fn1, fn2):
        s1 = fn1(w1) if w1 is not None else None
        s2 = fn2(w2) if w2 is not None else None
        return s1, s2

    if quant_type == "w4a8":
        return apply(w4a8_moe_repack_shuffle, w4a8_moe_repack_shuffle)

    if quant_type in ("int8_w8a8", "fp8_w8a8"):
        return apply(w8a8_moe_repack_shuffle, w8a8_moe_repack_shuffle)

    if quant_type == "fp4_w4a8":
        shuffle = wfp4a8_moe_repack_shuffle if block_shape else w4a8_moe_repack_shuffle
        return apply(shuffle, shuffle)

    if quant_type == "int8_w8a16":
        return apply(w8a16_marlin_weight_1, w8a16_marlin_weight_2)

    if quant_type == "w4a16":
        return apply(
            lambda x: w4a16_marlin_weight_1(x).view(-1).view(torch.uint8).view(*x.shape),
            lambda x: w4a16_marlin_weight_2(x).view(-1).view(torch.uint8).view(*x.shape),
        )

    if quant_type == "fp4_w4a16":
        return w1, w2

    raise NotImplementedError(f"Unsupported local shuffle quant_type: {quant_type}")


def moe_shfl_scale(
    scale1: Optional[torch.Tensor],
    scale2: Optional[torch.Tensor],
    moe_config,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    _require_moe_c(moe_config)
    quant_type = getattr(moe_config, "quant_type", None)

    def apply(fn1, fn2):
        s1 = fn1(scale1) if scale1 is not None else None
        s2 = fn2(scale2) if scale2 is not None else None
        return s1, s2

    if quant_type == "w4a16":
        return apply(w4a16_marlin_scale, w4a16_marlin_scale)
    if quant_type == "fp4_w4a16":
        return scale1, scale2
    raise NotImplementedError(f"Unsupported local scale shuffle quant_type: {quant_type}")


def w4a8_moe_layout_shuffle(w4a8_w: torch.Tensor, n_tile: Optional[int] = None) -> torch.Tensor:
    full_w4a8_w = w4a8_w.T
    k_tile = 32
    size_k, size_n = full_w4a8_w.shape
    if n_tile is None:
        n_tile = 256 if size_n % 256 == 0 else size_n
    if size_k % k_tile != 0 or size_n % n_tile != 0 or n_tile % 32 != 0:
        return w4a8_w.contiguous()

    full_w4a8_w = full_w4a8_w.reshape(size_k // k_tile, k_tile, size_n // n_tile, n_tile)
    full_w4a8_w = full_w4a8_w.permute((0, 2, 3, 1)).contiguous()
    full_w4a8_w = full_w4a8_w.reshape(
        size_k // k_tile,
        size_n // n_tile,
        n_tile // 32,
        32,
        k_tile // 8,
        8,
    )
    return full_w4a8_w.permute(0, 1, 2, 4, 3, 5).contiguous()


def repack_w4a8_weight_contiguous_to_blocked(packed_k_contiguous: torch.Tensor) -> torch.Tensor:
    if packed_k_contiguous.dim() not in (2, 3):
        raise ValueError(
            "packed_k_contiguous must be [N,K/2] or [E,N,K/2], "
            f"got shape {tuple(packed_k_contiguous.shape)}"
        )
    out_shape = (*packed_k_contiguous.shape[:-1], -1)
    k_half = packed_k_contiguous.shape[-1]
    if k_half % 4 != 0:
        raise ValueError(f"K/2 must be divisible by 4 for W4A8 blocked packing, got {k_half}")

    weight_u8 = packed_k_contiguous.to(torch.uint8)
    weight_0 = (weight_u8 >> 4) & 0x0F
    weight_1 = weight_u8 & 0x0F
    weight_unpacked = torch.stack([weight_0, weight_1], dim=-1).view(*out_shape)
    tile_view = weight_unpacked.view(*weight_unpacked.shape[:-1], -1, 8)
    weight_low = tile_view[..., :4].reshape(*weight_unpacked.shape[:-1], -1)
    weight_high = tile_view[..., 4:].reshape(*weight_unpacked.shape[:-1], -1)
    return ((weight_low << 4) | weight_high).to(packed_k_contiguous.dtype)


def w4a8_moe_repack_shuffle(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError(f"weight must be [E,N,K/2], got shape {tuple(weight.shape)}")
    if weight.element_size() != 1:
        raise ValueError("W4A8 packed weight must be an 8-bit tensor")
    if not weight.is_contiguous():
        weight = weight.contiguous()
    for expert_id in range(weight.shape[0]):
        blocked = repack_w4a8_weight_contiguous_to_blocked(weight[expert_id])
        shuffled = w4a8_moe_layout_shuffle(blocked).contiguous().view_as(weight[expert_id])
        weight[expert_id].copy_(shuffled)
    return weight


def _w8a8_marlin_weights_gemm2(q_w: torch.Tensor, k_tile: int = 64, n_tile: int = 16) -> torch.Tensor:
    e, size_k, size_n = q_w.shape
    q_w = q_w.reshape(e, size_k // k_tile, k_tile, size_n // n_tile, n_tile)
    q_w = q_w.permute((0, 1, 3, 4, 2)).contiguous()
    q_w = q_w.reshape(e, size_k // k_tile, size_n // n_tile, n_tile // 16, 16, k_tile // 16, 16)
    return q_w.permute(0, 1, 2, 3, 5, 4, 6).contiguous()


def w8a8_moe_layout_shuffle_gemm2(weight: torch.Tensor) -> torch.Tensor:
    return _w8a8_marlin_weights_gemm2(weight.permute(0, 2, 1).contiguous())


def w8a8_moe_repack_shuffle(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError(f"weight must be [E,N,K], got shape {tuple(weight.shape)}")
    return w8a8_moe_layout_shuffle_gemm2(weight).contiguous().view_as(weight)


def w4a16_marlin_weight_1(weight: torch.Tensor) -> torch.Tensor:
    e, n, k_half = weight.shape
    k = k_half * 2
    weight_u32 = weight.contiguous().view(-1).view(torch.uint32)
    weight_u32 = weight_u32.view(e, n // 16, 16, k // 32, 4)
    weight_u32 = weight_u32.transpose(2, 3).contiguous()
    weight_u32 = weight_u32.view(e, n // 16, k // 128, 4, 16, 4)
    return weight_u32.transpose(1, 2).contiguous()


def w4a16_marlin_weight_2(weight: torch.Tensor) -> torch.Tensor:
    e, k, n_half = weight.shape
    n = n_half * 2
    weight_u32 = weight.contiguous().view(-1).view(torch.uint32)
    weight_u32 = weight_u32.view(e, k // 16, 16, n // 32, 4)
    weight_u32 = weight_u32.transpose(2, 3).contiguous()
    weight_u32 = weight_u32.view(e, k // 16, n // 128, 4, 16, 4)
    return weight_u32.transpose(1, 2).contiguous()


def w4a16_marlin_scale(scale: torch.Tensor) -> torch.Tensor:
    e, n, k = scale.shape
    scale = scale.reshape(e, n, k // 4, 4).transpose(2, 3).contiguous()
    return scale.reshape(e, n, k)


def w8a16_marlin_weight_1(weight: torch.Tensor) -> torch.Tensor:
    e, n, k = weight.shape
    weight_u32 = weight.contiguous().view(-1).view(torch.uint32)
    weight_u32 = weight_u32.view(e, n // 16, 16, k // 32, 8)
    weight_u32 = weight_u32.transpose(2, 3).contiguous()
    weight_u32 = weight_u32.view(e, n // 16, k // 128, 4, 16, 8)
    shuffled = weight_u32.transpose(1, 2).contiguous()
    return shuffled.view(-1).view(torch.uint8).view_as(weight)


def w8a16_marlin_weight_2(weight: torch.Tensor) -> torch.Tensor:
    e, k, n = weight.shape
    weight_u32 = weight.contiguous().view(-1).view(torch.uint32)
    weight_u32 = weight_u32.view(e, k // 16, 16, n // 32, 8)
    weight_u32 = weight_u32.transpose(2, 3).contiguous()
    weight_u32 = weight_u32.view(e, k // 16, n // 128, 4, 16, 8)
    shuffled = weight_u32.transpose(1, 2).contiguous()
    return shuffled.view(-1).view(torch.uint8).view_as(weight)


def wfp4a8_moe_layout_shuffle(weight: torch.Tensor, n_tile: Optional[int] = None) -> torch.Tensor:
    transposed = weight.T
    k_tile = 32
    size_k, size_n = transposed.shape
    n_tile = 32 if n_tile is None else n_tile
    if size_k % k_tile != 0 or size_n % n_tile != 0 or n_tile % 32 != 0:
        return weight.contiguous()
    transposed = transposed.reshape(size_k // k_tile, k_tile, size_n // n_tile, n_tile)
    transposed = transposed.permute((0, 2, 3, 1)).contiguous()
    transposed = transposed.reshape(size_k // k_tile, size_n // n_tile, n_tile // 32, 32, k_tile // 4, 4)
    return transposed.permute(0, 1, 2, 4, 3, 5).contiguous()


def wfp4a8_moe_repack_shuffle(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError(f"weight must be [E,N,K/2], got shape {tuple(weight.shape)}")
    shuffled = [wfp4a8_moe_layout_shuffle(weight[i]) for i in range(weight.shape[0])]
    return torch.stack(shuffled).contiguous().view_as(weight)
