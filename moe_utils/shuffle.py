# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import torch
import numpy as np

# Moe_c Shuffle Function
#=================================================================================================================
def moe_layout_shuffle_gemm1(weight):

    return _w8a8_marlin_weight_1(weight)

def moe_layout_shuffle_gemm2(weight):

    return _w8a8_marlin_weight_2(weight)

def w4a8_moe_layout_shuffle_gemm1(weight):

    return _w4a8_gemm1_weight_shuffle(weight)

def w4a8_moe_layout_shuffle_gemm2(weight):

    return _w4a8_gemm2_weight_shuffle(weight)

#w4a16
def w4a16_marlin_weight_1(weight_input # [size_n, size_k// 2 ]
                        ):
    w1_qweight = weight_input
    e,n,k=w1_qweight.shape
    k = k * 2
    w1_qweight_uint32 = w1_qweight.view(-1).view(torch.uint32)
    new_shape = (e, n // 16, 16, k // 32, 4)  # uint32张量的形状
    w1_qweight_uint32_reshaped = w1_qweight_uint32.view(new_shape)
    w1_qweight_uint32_transposed = w1_qweight_uint32_reshaped.transpose(2, 3).contiguous()
    new_shape = (e, n // 16, k // 128, 4, 16, 4)
    w1_new_trans = w1_qweight_uint32_transposed.view(new_shape)
    w1_qweight_shuffle = w1_new_trans.transpose(1, 2).contiguous()

    return w1_qweight_shuffle

def w4a16_marlin_weight_2(weight_input # [size_n, size_k// 2 ]
                        ):
    w2_qweight = weight_input
    e,k,n=w2_qweight.shape
    n = n * 2
    w2_qweight_uint32 = w2_qweight.view(-1).view(torch.uint32)
    new_shape = (e, k // 16, 16, n // 32, 4)  # uint32张量的形状
    w2_qweight_uint32_reshaped = w2_qweight_uint32.view(new_shape)
    w2_qweight_uint32_transposed = w2_qweight_uint32_reshaped.transpose(2, 3).contiguous()
    new_shape = (e, k // 16, n // 128, 4, 16, 4)
    w2_new_trans = w2_qweight_uint32_transposed.view(new_shape)
    w2_qweight_shuffle = w2_new_trans.transpose(1, 2).contiguous()

    return w2_qweight_shuffle

def w4a16_marlin_scale(scale :torch.Tensor):
    pmt_factor = 4
    e, n, k =  scale.shape
    scale_out = scale.reshape(e, n, (k // pmt_factor), pmt_factor).transpose(2,3).contiguous()
    scale_out = scale_out.reshape(e, n, k)

    return scale_out

def wfp4a16_e8m0_scale(scale: torch.Tensor):
    return scale.contiguous()

#w8a8
def _w8a8_marlin_weight_1(weight_input # [size_n, size_k// 2 ]
                        ):
    weight = weight_input
    weight = weight.permute(0,2,1)
    marlin_q_w = _marlin_weights(weight,  k_tile=64, n_tile=16, pack_factor=8)
    return marlin_q_w

def _w8a8_marlin_weight_2(weight_input # [size_n, size_k// 2 ]
                        ):
    weight = weight_input
    weight = weight.permute(0,2,1)
    marlin_q_w = _marlin_weights_2(weight, k_tile=64, n_tile=16, pack_factor=8)
    return marlin_q_w

#w8a16
def w8a16_marlin_weight_1(weight_input # [size_n, size_k]
                        ):
    w1_qweight = weight_input
    e,n,k=w1_qweight.shape
    # k = k * 2
    w1_qweight_uint32 = w1_qweight.view(-1).view(torch.uint32)
    new_shape = (e, n // 16, 16, k // 32, 8)  # uint32张量的形状
    w1_qweight_uint32_reshaped = w1_qweight_uint32.view(new_shape)
    w1_qweight_uint32_transposed = w1_qweight_uint32_reshaped.transpose(2, 3).contiguous()
    new_shape = (e, n // 16, k // 128, 4, 16, 8)
    w1_new_trans = w1_qweight_uint32_transposed.view(new_shape)
    w1_qweight_shuffle = w1_new_trans.transpose(1, 2).contiguous()
    w1_new = w1_qweight_shuffle.view(-1).view(torch.uint8).view(*w1_qweight.shape)


    return w1_new

def w8a16_marlin_weight_2(weight_input # [size_n, size_k]
                        ):
    w2_qweight = weight_input
    e,k,n=w2_qweight.shape
    # n = n * 2
    w2_qweight_uint32 = w2_qweight.view(-1).view(torch.uint32)
    new_shape = (e, k // 16, 16, n // 32, 8)  # uint32张量的形状
    w2_qweight_uint32_reshaped = w2_qweight_uint32.view(new_shape)
    w2_qweight_uint32_transposed = w2_qweight_uint32_reshaped.transpose(2, 3).contiguous()
    new_shape = (e, k // 16, n // 128, 4, 16, 8)
    w2_new_trans = w2_qweight_uint32_transposed.view(new_shape)
    w2_qweight_shuffle = w2_new_trans.transpose(1, 2).contiguous()
    w2_new = w2_qweight_shuffle.view(-1).view(torch.uint8).view(*w2_qweight.shape)

    return w2_new


def _w16a16_marlin_weight_perm(device: torch.device) -> torch.Tensor:
    perm = []
    for i in range(64):
        for col in range(2):
            cur_col = (i % 16) * 2 + col
            for row in range(4):
                cur_row = (i // 16) * 4 + row
                perm.append(cur_row * 32 + cur_col)
    return torch.tensor(perm, dtype=torch.long, device=device)


def _w16a16_marlin_weights_npack2(
    weight: torch.Tensor,
    weight_perm: torch.Tensor,
    k_tile: int = 16,
    n_tile: int = 32,
) -> torch.Tensor:
    size_k, size_n = weight.shape
    weight = weight.reshape((size_k // k_tile, k_tile, size_n // n_tile, n_tile))
    weight = weight.permute((0, 2, 1, 3))
    weight = weight.reshape((size_k // k_tile, size_n * k_tile))
    weight = weight.reshape((-1, weight_perm.numel()))[:, weight_perm].reshape(weight.shape)
    return weight.contiguous()


def w16a16_marlin_weight(weight: torch.Tensor) -> torch.Tensor:
    weight_perm = _w16a16_marlin_weight_perm(weight.device)
    return _w16a16_marlin_weights_npack2(weight.T.contiguous(), weight_perm)


def w16a16_marlin_weight_k_n(weight: torch.Tensor) -> torch.Tensor:
    weight_perm = _w16a16_marlin_weight_perm(weight.device)
    return _w16a16_marlin_weights_npack2(weight.contiguous(), weight_perm)


def _marlin_weights(
                    q_w,
                    k_tile=64,
                    n_tile=16,
                    pack_factor=8):
    # 7168, 256
    e,size_k, size_n = q_w.shape
    q_w = q_w.reshape(e,size_k // k_tile, k_tile, size_n  )


    q_w = q_w.permute(0,1,3,2).contiguous()
    q_w = q_w.reshape(e,size_k // k_tile, size_n * k_tile)



    return q_w


def _marlin_weights_2(
                    q_w,
                    k_tile=64,
                    n_tile=16,
                    pack_factor=8):
    # 128 7168
    e, size_k, size_n = q_w.shape
    q_w = q_w.reshape(e,size_k // k_tile, k_tile, size_n //n_tile , n_tile )
    q_w = q_w.permute((0,1, 3, 4, 2)).contiguous()
    q_w = q_w.reshape(e, size_k // k_tile , size_n //n_tile , n_tile // 16 , 16, k_tile // 16 , 16 )
    q_w = q_w.permute(0,1,2,3,5,4,6).contiguous()

    return q_w

# w4a8


def w4a8_moe_layout_shuffle(w4a8_w, n_tile=None):
    full_w4a8_w = w4a8_w
    full_w4a8_w = full_w4a8_w.T
    k_tile=32
    size_k, size_n = full_w4a8_w.shape
    if n_tile is None:
        n_tile = 256 if size_n % 256 == 0 else size_n
    if size_k % k_tile != 0 or size_n % n_tile != 0 or n_tile % 32 != 0:
        return w4a8_w.contiguous()
    full_w4a8_w = full_w4a8_w.reshape(size_k // k_tile, k_tile, size_n //n_tile , n_tile )
    full_w4a8_w = full_w4a8_w.permute((0, 2, 3, 1)).contiguous()
    full_w4a8_w = full_w4a8_w.reshape(size_k // k_tile , size_n //n_tile , n_tile // 32 , 32, k_tile // 8 , 8 )
    full_w4a8_w = full_w4a8_w.permute(0,1,2,4,3,5).contiguous()

    return full_w4a8_w


def repack_w4a8_weight_contiguous_to_blocked(packed_k_contiguous: torch.Tensor) -> torch.Tensor:
    if packed_k_contiguous.dim() not in (2, 3):
        raise ValueError(
            "packed_k_contiguous must be [N,K/2] or [E,N,K/2], "
            f"got shape {tuple(packed_k_contiguous.shape)}"
        )
    if packed_k_contiguous.dim() == 2:
        n, k_half = packed_k_contiguous.shape
        out_shape = (n, -1)
    else:
        experts, n, k_half = packed_k_contiguous.shape
        out_shape = (experts, n, -1)
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


def wfp4a8_moe_layout_shuffle(wfp4a8_w, n_tile=None):
    full_wfp4a8_w = wfp4a8_w
    full_wfp4a8_w = full_wfp4a8_w.T
    k_tile = 32
    size_k, size_n = full_wfp4a8_w.shape
    if n_tile is None:
        n_tile = 32
    if size_k % k_tile != 0 or size_n % n_tile != 0 or n_tile % 32 != 0:
        return wfp4a8_w.contiguous()
    full_wfp4a8_w = full_wfp4a8_w.reshape(
        size_k // k_tile, k_tile, size_n // n_tile, n_tile
    )
    full_wfp4a8_w = full_wfp4a8_w.permute((0, 2, 3, 1)).contiguous()
    full_wfp4a8_w = full_wfp4a8_w.reshape(
        size_k // k_tile, size_n // n_tile, n_tile // 32, 32, k_tile // 4, 4
    )
    full_wfp4a8_w = full_wfp4a8_w.permute(0, 1, 2, 4, 3, 5).contiguous()

    return full_wfp4a8_w

def _w4a8_gemm1_weight_shuffle(w4a8_w):

    return w4a8_moe_layout_shuffle(w4a8_w)

def _w4a8_gemm2_weight_shuffle(w4a8_w):
    return w4a8_moe_layout_shuffle(w4a8_w)

def _wfp4a8_gemm1_weight_shuffle(wfp4a8_w):
    return wfp4a8_moe_layout_shuffle(wfp4a8_w)

def _wfp4a8_gemm2_weight_shuffle(wfp4a8_w):
    return wfp4a8_moe_layout_shuffle(wfp4a8_w)

#=======================================================Moe_c Shuffle Function================================================================

def asm_shuffle_weight_b8(x: torch.Tensor, stage: torch.int32 = 1) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    assert x.dtype in [
        torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.float8_e4m3fn
    ]

    if x.dtype == torch.int8 or x.dtype == torch.float8_e4m3fn:
        N = 16
        K = 16
        IK = 64
        IN = 64
        BK = 256
        BN = 128
        if stage == 1:
            if x.shape[-2] % 128 != 0 and x.shape[-2] % 64 == 0:
                BN = 64
        if stage == 2:
            if x.shape[-1] % 128 == 0:
                BK = 128
            elif x.shape[-1] % 128 == 64:
                BN = 64
                BK = 64
            elif x.shape[-1] % 128 == 96:
                BN = 64
                BK = 64
                assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
                x_ = x
                multiple = x.shape[-1] // BK * BK
                part1 = x[:, :, :multiple]
                ### part1 shuffle
                #             0,          1,           2,       3,     4,        5,              6,       7,    8
                part1 = part1.view(-1, part1.shape[-2] // BN, BN // IN, IN // N, N, part1.shape[-1] // BK, BK // IK, IK // K, K)
                part1 = part1.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
                part1 = part1.flatten(start_dim=1)
                ### part2 shuffle
                part2 = x[:, :, multiple:]
                IK = 32
                BK = 32
                #             0,          1,           2,       3,     4,        5,              6,       7,    8
                part2 = part2.view(-1, part2.shape[-2] // BN, BN // IN, IN // N, N, part2.shape[-1] // BK, BK // IK, IK // K, K)
                part2 = part2.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
                part2 = part2.flatten(start_dim=1)
                ### combine
                x_ = torch.cat((part1, part2), dim=1)
                x_ = x_.view(*x.shape)
                return x_

    elif x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        N = 16
        K = 8
        IK = 32
        IN = 64
        BK = 128
        BN = 64
        if stage == 2:
            BK = 32
    else:
        assert False, f"not support {x.dtype}"

    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    #             0,          1,           2,       3,     4,        5,              6,       7,    8
    x_ = x_.view(-1, x.shape[-2] // BN, BN // IN, IN // N, N, x.shape[-1] // BK, BK // IK, IK // K, K)
    x_ = x_.permute(0, 1, 5, 2, 6, 3, 7, 4, 8)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_

def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_

# TN Layout in -> CK Tiling Layout out
# layout(NWaves, NRepeat, NLane, NInterleave, NVec,  KWaves, KRepeat, KLane,  KVec)
def ck_shuffle_weight(x:torch.Tensor, layout=(4, 1, 16, 2, 1, 1, 4, 4, 8)) -> torch.Tensor:
    NWaves, NRepeat, NLane, NInterleave, NVec,  KWaves, KRepeat, KLane,  KVec = layout
    Block_N = NWaves * NRepeat * NLane * NInterleave * NVec
    Block_K = KWaves * KRepeat * KLane * KVec
    assert x.shape[-2] % Block_N == 0, f"{x.shape[-2]} % {Block_N} == {x.shape[-2] % Block_N }"
    assert x.shape[-1] % Block_K == 0, f"{x.shape[-1]} % {Block_K} == {x.shape[-1] % Block_K }"

    x_ = x
    #           (0,  1                      , 2     , 3     , 4    , 5          , 6   , 7                     , 8     , 9      , 10   , 11)
    x_ = x_.view(-1, x.shape[-2] // Block_N, NWaves, NRepeat, NLane, NInterleave, NVec, x.shape[-1] // Block_K, KWaves, KRepeat, KLane, KVec)
    # x_ = x_.permute(0, 1, 7, 3, 5, 9, 8, 2, 10, 4, 6, 11)
    x_ = x_.permute(0, 1, 7, 2, 8, 3, 5, 9, 10, 4, 6, 11)
    x_ = x_.contiguous()
    x_ = x_.view(-1, x.shape[-2] // Block_N, x.shape[-1] // Block_K, Block_N * Block_K)
    return x_

# layout(NWaves, NRepeat, NLane, NInterleave, NVec,  KWaves, KRepeat, KLane,  KVec)
def ck_shuffle_weight_down(x:torch.Tensor, layout=(4, 2, 16, 1, 1, 1, 4, 4, 8)) -> torch.Tensor:
    NWaves, NRepeat, NLane, NInterleave, NVec,  KWaves, KRepeat, KLane,  KVec = layout
    Block_N = NWaves * NRepeat * NLane * NInterleave * NVec
    Block_K = KWaves * KRepeat * KLane * KVec
    assert x.shape[-2] % Block_N == 0, f"{x.shape[-2]} % {Block_N} == {x.shape[-2] % Block_N }"
    assert x.shape[-1] % Block_K == 0, f"{x.shape[-1]} % {Block_K} == {x.shape[-1] % Block_K }"

    x_ = x
    #           (0,  1                      , 2     , 3     , 4    , 5          , 6   , 7                     , 8     , 9      , 10   , 11)
    x_ = x_.view(-1, x.shape[-2] // Block_N, NWaves, NRepeat, NLane, NInterleave, NVec, x.shape[-1] // Block_K, KWaves, KRepeat, KLane, KVec)
    x_ = x_.permute(0, 7, 1, 2, 8, 3, 5, 9, 10, 4, 6, 11)       #down weight loop in N dim
    x_ = x_.contiguous()
    x_ = x_.view(-1, x.shape[-1] // Block_K, x.shape[-2] // Block_N, Block_N * Block_K)
    return x_

def reverse_awq_order(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse the AWQ order of the given tensor.

    Args:
        tensor: Input tensor to reorder

    Returns:
        Reordered tensor with bits masked to 4 bits
    """
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        tensor.shape[-1],
        dtype=torch.int32,
        device=tensor.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    tensor = tensor[:, reverse_order_tensor] & 0xF
    return tensor

def awq_reorder_and_repack(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder and pack weights and zeros using AWQ order.

    This function unpacks the 4-bit quantized weights and zeros from int32,
    applies reverse_awq_order to reorder them, and then packs them.
    For weight, repack to [N, K//2]
    For zeros, repack to [K//G, N//2]
    Args:
        qweight: Quantized weight tensor of shape [K, N // 8] with dtype int32
        qzeros: Quantized zero points tensor of shape [K // G, N // 8] with dtype int32

    Returns:
        Tuple of (reordered_qweight, reordered_qzeros) both with dtype int8
    """
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    K = qweight.shape[0]
    N = qweight.shape[1] * 8
    G = K // qzeros.shape[0]

    # Unpack weights: [K, N//8] -> [K, N//8, 8] -> [K, N]
    iweights = torch.bitwise_right_shift(
        qweight[:, :, None],
        shifts[None, None, :],
    ).to(torch.int8)
    iweights = iweights.view(K, -1)

    # Unpack zeros: [K//G, N//8] -> [K//G, N//8, 8] -> [K//G, N]
    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None],
        shifts[None, None, :],
    ).to(torch.int8)
    zeros = zeros.view(K//G, -1)

    # Apply reverse AWQ order to both tensors
    iweights = reverse_awq_order(iweights)
    zeros = reverse_awq_order(zeros)

    # Mask to 4 bits
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    # Repack weight to int32 and pack along the K direction
    # [K, N] -> [N, K]
    iweights = iweights.transpose(1, 0).contiguous()
    # Reshape to [N, K//2, 2] for weights
    iweights_packed = iweights.view(N, -1, 2)

    # Repack zeros to int8 and pack along the N direction
    # Reshape to [K//G, N//2, 2] for zeros
    zeros_packed = zeros.view(K//G, -1, 2)

    # Pack 2 int4 values into int8 using bit shifts
    # Direct packing: pack in the order they appear after reordering
    packed_weights = torch.zeros([N, K//2], dtype=torch.int8, device=qweight.device)
    packed_zeros = torch.zeros([K//G, N//2], dtype=torch.int8, device=zeros.device)

    for i in range(2):
        packed_weights |= (iweights_packed[:, :, i].to(torch.int8) << (i * bits))
        packed_zeros |= (zeros_packed[:, :, i].to(torch.int8) << (i * bits))

    return packed_weights, packed_zeros




# Compatibility aliases used by the legacy standalone dispatcher.
def w8a8_moe_layout_shuffle_gemm2(weight):
    return moe_layout_shuffle_gemm2(weight)


def w8a8_moe_repack_shuffle(weight):
    return moe_layout_shuffle_gemm2(weight).contiguous().view_as(weight)


def wfp4a8_moe_repack_shuffle(weight):
    if weight.dim() != 3:
        raise ValueError(f"weight must be [E,N,K/2], got shape {tuple(weight.shape)}")
    return torch.stack([wfp4a8_moe_layout_shuffle(weight[i]) for i in range(weight.shape[0])]).contiguous().view_as(weight)

def _require_moe_c(moe_config) -> None:
    if getattr(moe_config, "solution_type", None) != "moe_c":
        raise NotImplementedError("Local shuffle currently supports only MOE_C")


def moe_shfl_weight(w1, w2, moe_config, block_shape=None):
    _require_moe_c(moe_config)
    quant_type = getattr(moe_config, "quant_type", None)
    def apply(fn1, fn2):
        return (fn1(w1) if w1 is not None else None,
                fn2(w2) if w2 is not None else None)
    if quant_type == "w4a8":
        return apply(w4a8_moe_repack_shuffle, w4a8_moe_repack_shuffle)
    if quant_type in ("int8_w8a8", "fp8_w8a8"):
        if quant_type == "fp8_w8a8":
            padded_k = (getattr(moe_config, "config", None) or {}).get("PADDED_K")
            if padded_k is not None:
                def pad(x, dim):
                    if x.shape[dim] >= int(padded_k):
                        return x
                    shape=list(x.shape); shape[dim]=int(padded_k)
                    y=x.new_zeros(shape); sl=[slice(None)]*x.dim(); sl[dim]=slice(0,x.shape[dim]); y[tuple(sl)]=x
                    return y
                return apply(lambda x: moe_layout_shuffle_gemm2(pad(x, 2)).contiguous().view(x.shape[0], x.shape[1], int(padded_k)),
                             lambda x: moe_layout_shuffle_gemm2(pad(x, 1)).contiguous().view(x.shape[0], int(padded_k), x.shape[2]))
        return apply(lambda x: moe_layout_shuffle_gemm2(x).contiguous().view(*x.shape),
                     lambda x: moe_layout_shuffle_gemm2(x).contiguous().view(*x.shape))
    if quant_type == "fp4_w4a8":
        fn = wfp4a8_moe_repack_shuffle if block_shape else w4a8_moe_repack_shuffle
        return apply(fn, fn)
    if quant_type == "int8_w8a16":
        return apply(w8a16_marlin_weight_1, w8a16_marlin_weight_2)
    if quant_type == "w4a16":
        return apply(lambda x: w4a16_marlin_weight_1(x).view(-1).view(torch.uint8).view(*x.shape),
                     lambda x: w4a16_marlin_weight_2(x).view(-1).view(torch.uint8).view(*x.shape))
    if quant_type == "fp4_w4a16":
        return w1, w2
    if quant_type == "w16a16":
        return apply(w16a16_marlin_weight, w16a16_marlin_weight)
    raise NotImplementedError(f"Unsupported local shuffle quant_type: {quant_type}")


def moe_shfl_scale(scale1, scale2, moe_config):
    _require_moe_c(moe_config)
    quant_type = getattr(moe_config, "quant_type", None)
    def apply(fn):
        return (fn(scale1) if scale1 is not None else None, fn(scale2) if scale2 is not None else None)
    if quant_type == "w4a16":
        return apply(w4a16_marlin_scale)
    if quant_type == "fp4_w4a16":
        return scale1, scale2
    raise NotImplementedError(f"Unsupported local scale shuffle quant_type: {quant_type}")
