# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""INT4 groupwise inputs, reference and direct kernel calls for the HIPC tuner."""

from dataclasses import dataclass
from typing import Optional

from tune.hipc_runtime import fused_moe, ops


W4A16_TOLERANCES = dict(atol=0.02, rtol=0.02, max_mismatch=0.01)


@dataclass
class W4A16BenchRow:
    m: int
    stage: str
    mode: int
    bm: int
    status: str  # ok | mismatch | failed | crashed | unsupported
    latency_us: Optional[float] = None
    # GEMM1 modes without fused silu pay a separate silu_and_mul pass; it is part of the objective.
    epilogue_us: float = 0.0
    max_abs: Optional[float] = None
    mismatch_ratio: Optional[float] = None
    error: str = ""


def quantize_symmetric_w4(w, group_size: int):
    """[E, N, K] float -> (packed uint8 [E, N, K/2], scales [E, N, K/G], dequantized ref).

    Matches the kernel's symmetric dequant (q - 8) * scale with scales stored
    in the activation dtype.
    """
    import torch

    e, n, k = w.shape
    packed = torch.empty((e, n, k // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((e, n, k // group_size), dtype=w.dtype, device=w.device)
    ref = torch.empty_like(w)
    # Work in expert chunks: the fp32 intermediates of a whole [E, N, K] weight (24 GB for
    # E=384, 2N=4096, K=4096) do not fit next to the weight and its reference.
    step = max(1, (1 << 28) // (n * k))
    for lo in range(0, e, step):
        grouped = w[lo:lo + step].float().reshape(-1, n, k // group_size, group_size)
        sc = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8).to(w.dtype)
        q = torch.clamp(torch.round(grouped / sc.float()), -8, 7) + 8
        ref[lo:lo + step] = ((q - 8) * sc.float()).reshape(-1, n, k).to(w.dtype)
        q = q.reshape(-1, n, k).to(torch.uint8)
        packed[lo:lo + step] = q[..., 1::2] * 16 + q[..., ::2]
        scales[lo:lo + step] = sc.reshape(-1, n, k // group_size)
    return packed, scales, ref


def _w4a16_per_expert_matmul(rows_by_expert, a, w_ref, out_rows):
    """out[r] = a[src(r)] @ w_ref[e].T in fp32 for each expert's routed rows."""
    for e, (dst, src) in rows_by_expert.items():
        out_rows[dst] = a[src].float() @ w_ref[e].float().T


def routed_rows(topk_ids):
    flat = topk_ids.reshape(-1)
    order = flat.argsort()
    experts, counts = flat[order].unique_consecutive(return_counts=True)
    rows = {}
    start = 0
    for e, c in zip(experts.tolist(), counts.tolist()):
        rows[e] = order[start:start + c]
        start += c
    return rows


class W4A16StageData:
    """Inputs, kernel operands and fp32 reference for one stage at one M."""

    def __init__(self, a, qweight, scale, zeros, topk_weights, out_shape, reference, top_k):
        self.a, self.qweight, self.scale, self.zeros = a, qweight, scale, zeros
        self.topk_weights, self.out_shape, self.reference, self.top_k = topk_weights, out_shape, reference, top_k
        self.fused_reference = None  # GEMM1 only: silu(gate) * up of ``reference``


def make_inputs(shape, dtype, seed: int):
    import torch
    from moe_utils.shuffle import w4a16_marlin_scale

    e, n, k, topk, m = shape.e, shape.n, shape.k, shape.topk, shape.m
    g = shape.group_size
    gen = torch.Generator(device="cuda").manual_seed(seed + m)

    def randn(*shape):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=gen) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype, generator=gen)
    hidden = randn(m, k)
    topk_weights, topk_ids = fused_moe().fused_topk(hidden, score, topk, renormalize=True)
    topk_weights = topk_weights.float().contiguous()
    rows = routed_rows(topk_ids)  # routed row r = token * topk + slot
    token_of_row = torch.arange(m * topk, device="cuda") // topk

    def stage(w_fp, a, src_of_row, scale_rows, out_shape, top_k):
        packed, scales, ref_w = quantize_symmetric_w4(w_fp, g)
        zeros = torch.full((e, w_fp.shape[1] // 2, w_fp.shape[2] // g), 0x88,
                           device="cuda", dtype=torch.uint8)
        ref = torch.empty((m * topk, w_fp.shape[1]), device="cuda", dtype=torch.float32)
        _w4a16_per_expert_matmul({x: (r, src_of_row[r]) for x, r in rows.items()}, a, ref_w, ref)
        if scale_rows is not None:
            ref *= scale_rows.reshape(-1, 1)
        ref = ref.to(dtype).reshape(out_shape)
        del ref_w
        if g == 32:
            scales = w4a16_marlin_scale(scales)
        return W4A16StageData(a, packed.view(torch.uint32), scales, zeros,
                               None if scale_rows is None else topk_weights, out_shape, ref, top_k)

    # GEMM1: hidden [M, K] x W1 [E, 2N, K] -> [M, topk, 2N]  (gate = first N columns, up = last N)
    gemm1 = stage(randn(e, 2 * n, k), hidden, token_of_row, None, (m, topk, 2 * n), topk)
    # fused-silu modes write silu(gate) * up as [M, topk, N]
    act_ref = gemm1.reference.float()
    gemm1.fused_reference = (torch.nn.functional.silu(act_ref[..., :n]) * act_ref[..., n:]).to(dtype)
    del act_ref
    # GEMM2: an independent [M*topk, N] activation x W2 [E, K, N] -> [M, topk, K],
    # scaled by the routed topk weight.
    a2 = randn(m * topk, n)
    gemm2 = stage(randn(e, k, n), a2, torch.arange(m * topk, device="cuda"),
                  topk_weights, (m, topk, k), 1)
    return topk_ids, gemm1, gemm2


_W4A16_SILU_COST = {}


def _silu_cost_us(data, warmup, iters, event_time_us) -> float:
    """Time of the separate silu_and_mul pass an unfused GEMM1 output still needs (cached per shape)."""
    import torch
    moe_c_silu_and_mul = ops().moe_c_silu_and_mul

    key = (tuple(data.out_shape), data.reference.dtype, data.a.device, warmup, iters)
    dims = data.out_shape
    if key not in _W4A16_SILU_COST:
        x = torch.randn(dims[0] * dims[1], dims[2], device="cuda", dtype=data.reference.dtype)
        y = torch.empty(x.shape[0], dims[2] // 2, device="cuda", dtype=x.dtype)
        _W4A16_SILU_COST[key] = event_time_us(lambda: moe_c_silu_and_mul(y, x), warmup, iters)
    return _W4A16_SILU_COST[key]


def bench_candidate(m, desc, data, routing, warmup, iters, event_time_us, *,
                    atol=W4A16_TOLERANCES["atol"], rtol=W4A16_TOLERANCES["rtol"],
                    max_mismatch=W4A16_TOLERANCES["max_mismatch"]) -> W4A16BenchRow:
    import torch
    moe_c_moe_gemm_marlin_w4a16 = ops().moe_c_moe_gemm_marlin_w4a16

    fused = desc.unfused_mode is not None
    row = W4A16BenchRow(m, f"gemm{desc.stage}", desc.mode, desc.bm, status="failed")
    sorted_ids, expert_ids, num_post = routing
    reference, out_shape = data.reference, data.out_shape
    if fused:
        reference, out_shape = data.fused_reference, data.fused_reference.shape
    out = torch.full(out_shape, float("nan"), device="cuda", dtype=reference.dtype)

    def run():
        moe_c_moe_gemm_marlin_w4a16(data.a, data.qweight, out, data.scale, data.zeros,
                                    data.topk_weights, sorted_ids, expert_ids, num_post,
                                    data.top_k, desc.mode, 1)

    try:
        run()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        text = str(exc)
        if any(x in text.lower() for x in ("illegal memory", "memory access fault", "device-side assert")):
            raise  # poisoned device: abort the whole sweep loudly
        row.error = text.splitlines()[0][:300]
        return row
    ref = reference.float()
    got = out.float()
    diff = (got - ref).abs()
    finite = torch.isfinite(got).all().item()
    row.max_abs = float(diff.nan_to_num(float("inf")).max().item())
    bad = ~(diff <= atol + rtol * ref.abs())  # NaN counts as bad
    row.mismatch_ratio = float(bad.float().mean().item())
    if not finite or row.mismatch_ratio > max_mismatch:
        row.status = "mismatch"
        row.error = "non-finite output" if not finite else f"mismatch_ratio={row.mismatch_ratio:.4g}"
        return row
    row.latency_us = event_time_us(run, warmup, iters)
    if desc.stage == 1 and not fused:
        row.epilogue_us = _silu_cost_us(data, warmup, iters, event_time_us)
    row.status = "ok"
    return row
