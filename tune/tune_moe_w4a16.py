#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""moe_c W4A16 groupwise GEMM tuner (also reachable as ``tune_moe_hipc.py --quant int4_w4a16``).

Follows README_hipc.md.  GEMM1 and GEMM2 get independent inputs.  Candidate modes are the ones the
built library registered (moe_c_w4a16_modes).  Every candidate is checked against a PyTorch
reference built from the dequantized weights; outputs are pre-filled with NaN so a kernel that
never ran fails the check.  Routing, quantization and references are prepared outside the timed
region, and only the GEMM launch is timed.

For each M the tuner takes the fastest passing GEMM1 and GEMM2 independently (their BMs may
differ; the runtime then aligns the routing once per distinct BM).  The objective is GEMM-only:
gemm1 (+ a separate silu_and_mul when the mode does not fuse it) + gemm2.  A GPU fault kills the
process, so a supervising parent records the running candidate as ``crashed`` and resumes.

Results are merged into a ``tuned_moe_c_w4a16_group.csv`` (default under --out-dir).  The runtime
config directory is only written when --config-csv points at it.  See README_w4a16.md.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Same token grid as tune_moe_w4a8_perchannel.DEFAULT_M_VALUES.
DEFAULT_M_VALUES = [
    1, 2, 3, 4, 5, 6, 7, 8,
    16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 6144, 8192, 16384,
]
DTYPES = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}


@dataclass
class CandidateRow:
    m: int
    stage: str
    mode: int
    bm: int
    status: str  # ok | mismatch | failed | crashed
    latency_us: Optional[float] = None
    # GEMM1 modes without fused silu pay a separate silu_and_mul pass; it is part of the objective.
    epilogue_us: float = 0.0
    max_abs: Optional[float] = None
    mismatch_ratio: Optional[float] = None
    error: str = ""


from tune.tune_moe_hipc import _event_time_us  # shared with the other quantizers


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


def _per_expert_matmul(rows_by_expert, a, w_ref, out_rows):
    """out[r] = a[src(r)] @ w_ref[e].T in fp32 for each expert's routed rows."""
    for e, (dst, src) in rows_by_expert.items():
        out_rows[dst] = a[src].float() @ w_ref[e].float().T


def _route_rows(topk_ids):
    flat = topk_ids.reshape(-1)
    order = flat.argsort()
    experts, counts = flat[order].unique_consecutive(return_counts=True)
    rows = {}
    start = 0
    for e, c in zip(experts.tolist(), counts.tolist()):
        rows[e] = order[start:start + c]
        start += c
    return rows


class StageData:
    """Inputs, kernel operands and fp32 reference for one stage at one M."""

    def __init__(self, a, qweight, scale, zeros, topk_weights, out_shape, reference, top_k):
        self.a, self.qweight, self.scale, self.zeros = a, qweight, scale, zeros
        self.topk_weights, self.out_shape, self.reference, self.top_k = topk_weights, out_shape, reference, top_k
        self.fused_reference = None  # GEMM1 only: silu(gate) * up of ``reference``


def make_inputs(m: int, args, dtype):
    import torch
    from moe_utils.shuffle import w4a16_marlin_scale
    from moe_utils.utils import fused_topk

    e, n, k, topk, g = args.e, args.n, args.k, args.topk, args.group_size
    gen = torch.Generator(device="cuda").manual_seed(args.seed + m)

    def randn(*shape):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=gen) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype, generator=gen)
    hidden = randn(m, k)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)
    topk_weights = topk_weights.float().contiguous()
    rows = _route_rows(topk_ids)  # routed row r = token * topk + slot
    token_of_row = torch.arange(m * topk, device="cuda") // topk

    def stage(w_fp, a, src_of_row, scale_rows, out_shape, top_k):
        packed, scales, ref_w = quantize_symmetric_w4(w_fp, g)
        zeros = torch.full((e, w_fp.shape[1] // 2, w_fp.shape[2] // g), 0x88,
                           device="cuda", dtype=torch.uint8)
        ref = torch.empty((m * topk, w_fp.shape[1]), device="cuda", dtype=torch.float32)
        _per_expert_matmul({x: (r, src_of_row[r]) for x, r in rows.items()}, a, ref_w, ref)
        if scale_rows is not None:
            ref *= scale_rows.reshape(-1, 1)
        ref = ref.to(dtype).reshape(out_shape)
        del ref_w
        return StageData(a, packed.view(torch.uint32), w4a16_marlin_scale(scales), zeros,
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


_SILU_COST = {}


def silu_cost_us(data, args) -> float:
    """Time of the separate silu_and_mul pass an unfused GEMM1 output still needs (cached per shape)."""
    import torch
    from moe_utils.moe_ops import moe_c_silu_and_mul

    key = tuple(data.out_shape)
    if key not in _SILU_COST:
        x = torch.randn(key[0] * key[1], key[2], device="cuda", dtype=data.reference.dtype)
        y = torch.empty(x.shape[0], key[2] // 2, device="cuda", dtype=x.dtype)
        _SILU_COST[key] = _event_time_us(lambda: moe_c_silu_and_mul(y, x), args.warmup, args.iters)
    return _SILU_COST[key]


def bench_candidate(m, stage_name, mode, data, routing, args) -> CandidateRow:
    import torch
    from moe_utils.fused_moe_c import w4a16_fuse_silu
    from moe_utils.moe_ops import moe_c_moe_gemm_marlin_w4a16

    fused = w4a16_fuse_silu(mode)
    row = CandidateRow(m, stage_name, mode, mode // 1000 % 100 * 16, status="failed")
    sorted_ids, expert_ids, num_post = routing
    reference, out_shape = data.reference, data.out_shape
    if fused:
        reference, out_shape = data.fused_reference, data.fused_reference.shape
    out = torch.full(out_shape, float("nan"), device="cuda", dtype=reference.dtype)

    def run():
        moe_c_moe_gemm_marlin_w4a16(data.a, data.qweight, out, data.scale, data.zeros,
                                    data.topk_weights, sorted_ids, expert_ids, num_post,
                                    data.top_k, mode, 1)

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
    bad = ~(diff <= args.atol + args.rtol * ref.abs())  # NaN counts as bad
    row.mismatch_ratio = float(bad.float().mean().item())
    if not finite or row.mismatch_ratio > args.max_mismatch:
        row.status = "mismatch"
        row.error = "non-finite output" if not finite else f"mismatch_ratio={row.mismatch_ratio:.4g}"
        return row
    row.latency_us = _event_time_us(run, args.warmup, args.iters)
    if stage_name == "gemm1" and not fused:
        row.epilogue_us = silu_cost_us(data, args)
    row.status = "ok"
    return row


def best_pair(rows: List[CandidateRow], m: int) -> Optional[Dict[str, Any]]:
    """Fastest passing GEMM1 and GEMM2, chosen independently: their BMs may differ because the
    runtime aligns the routing once per distinct BM (dual align)."""
    passed = [r for r in rows if r.m == m and r.status == "ok"]
    up = [r for r in passed if r.stage == "gemm1"]
    down = [r for r in passed if r.stage == "gemm2"]
    if not up or not down:
        return None
    g1 = min(up, key=lambda r: r.latency_us + r.epilogue_us)
    g2 = min(down, key=lambda r: r.latency_us)
    return {"gemm1": asdict(g1), "gemm2": asdict(g2), "dual_routing": g1.bm != g2.bm,
            "latency_us": g1.latency_us + g1.epilogue_us + g2.latency_us}


def candidate_modes(stage: int, selected: Optional[List[int]], families=("decode", "prefill")):
    """Modes to try: every mode the build registered a kernel for (moe_c_w4a16_modes), or the
    requested subset.  Shape limits (K, N) are enforced by the launchers; a candidate that cannot
    run this shape fails there and is never written to the config."""
    from moe_utils.moe_ops import moe_c_w4a16_modes

    registered = moe_c_w4a16_modes(stage)
    if selected is not None:
        unknown = sorted(set(selected) - set(registered))
        if unknown:
            raise SystemExit(f"GEMM{stage} modes {unknown} have no kernel; available: {registered}")
        return sorted(set(selected))
    return [m for m in registered if ("prefill" if m // 100000 % 10 else "decode") in families]


def tune(args) -> int:
    import torch
    from moe_utils.fused_moe_c import CSV_NAMES, moe_align_block_size, rows_from_pair, write_config_csv

    torch.manual_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    arch = args.arch or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    # The runtime keys lookups on the device CU count (_resolve_num_cu); match it.
    cu = args.cu if args.cu is not None else torch.cuda.get_device_properties(0).multi_processor_count
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"w4a16_{arch}_E{args.e}_N{args.n}_K{args.k}_topk{args.topk}_cu{cu}_{args.dtype}"
    rows_path = out_dir / f"{tag}_candidates.csv"
    log_path, inflight_path = out_dir / f"{tag}_candidates.jsonl", out_dir / f"{tag}.inflight"
    stage_modes = {1: candidate_modes(1, args.gemm1_modes, args.families),
                   2: candidate_modes(2, args.gemm2_modes, args.families)}

    # Finished candidates are appended to a JSONL log, so a restart after a GPU fault resumes.
    rows: List[CandidateRow] = []
    if log_path.exists():
        rows = [CandidateRow(**json.loads(line)) for line in log_path.read_text().splitlines() if line]
    done = {(r.m, r.stage, r.mode) for r in rows}
    for m in args.m:
        if all((m, name, mode) in done for stage, name in ((1, "gemm1"), (2, "gemm2")) for mode in stage_modes[stage]):
            continue
        topk_ids, gemm1, gemm2 = make_inputs(m, args, dtype)
        routings = {}
        for stage, name, data in ((1, "gemm1", gemm1), (2, "gemm2", gemm2)):
            for mode in stage_modes[stage]:
                if (m, name, mode) in done:
                    continue
                bm = mode // 1000 % 100 * 16
                if bm not in routings:
                    routings[bm] = moe_align_block_size(topk_ids, bm, args.e)
                inflight_path.write_text(f"{m} {name} {mode}\n")  # names the culprit if the GPU faults
                row = bench_candidate(m, name, mode, data, routings[bm], args)
                rows.append(row)
                with log_path.open("a") as log:
                    log.write(json.dumps(asdict(row)) + "\n")
                inflight_path.unlink()
                latency = "" if row.latency_us is None else f"{row.latency_us:.2f}us"
                print(f"M={m} {name} mode={mode} BM={bm} {row.status} {latency} {row.error}", flush=True)
        del gemm1, gemm2, routings
        torch.cuda.empty_cache()
        with rows_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(CandidateRow.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(r) for r in rows)

    summary = {"shape": {"arch": arch, "cu": cu, "e": args.e, "n": args.n, "k": args.k,
                         "topk": args.topk, "dtype": args.dtype, "group_size": args.group_size},
               "objective": "gemm1_us (+ silu_and_mul_us if unfused) + gemm2_us; BMs may differ (dual align)", "m_results": {}}
    configs = {"gemm1": {}, "gemm2": {}}
    missing = []
    for m in args.m:
        best = best_pair(rows, m)
        summary["m_results"][str(m)] = {"best": best}
        if best is None:
            missing.append(m)
            print(f"M={m}: no passing GEMM1/GEMM2 candidate pair", flush=True)
            continue
        for stage in ("gemm1", "gemm2"):
            configs[stage][str(m)] = {"MODE": best[stage]["mode"], "BLOCK_SIZE_M": best[stage]["bm"]}
        print(f"M={m} modes={best['gemm1']['mode']}+{best['gemm2']['mode']} "
              f"BM={best['gemm1']['bm']}/{best['gemm2']['bm']} sum_us={best['latency_us']:.2f}", flush=True)
    (out_dir / f"{tag}_best_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if configs["gemm1"]:
        times = {m: r["best"]["latency_us"] for m, r in summary["m_results"].items() if r["best"]}
        csv_rows = rows_from_pair(configs["gemm1"], configs["gemm2"], arch, "int4_w4a16", args.e,
                                  args.n, args.k, args.topk, cu, (DTYPES[args.dtype],), times)
        config_csv = args.config_csv or out_dir / CSV_NAMES["int4_w4a16"]
        write_config_csv(str(config_csv), csv_rows)
        print(f"tuned rows merged into {config_csv}", flush=True)
    print(f"candidates: {rows_path}", flush=True)
    return 2 if missing else 0


def supervise(args, argv) -> int:
    """Run the sweep in a child process.  A VMFault kills the process (it cannot be caught), so the
    candidate that was running is recorded as ``crashed`` and the sweep resumes after it."""
    import subprocess

    args.out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv, "--worker"])
        inflight = list(args.out_dir.glob("*.inflight"))
        if rc in (0, 2) or not inflight:
            return rc
        for path in inflight:
            m, name, mode = path.read_text().split()
            stem = path.name[:-len(".inflight")]
            row = CandidateRow(int(m), name, int(mode), int(mode) // 1000 % 100 * 16, status="crashed",
                               error=f"worker exited with {rc}")
            with (args.out_dir / f"{stem}_candidates.jsonl").open("a") as log:
                log.write(json.dumps(asdict(row)) + "\n")
            path.unlink()
            print(f"M={m} {name} mode={mode} crashed (worker exit {rc}); resuming", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e", type=int, required=True, help="experts on this rank")
    parser.add_argument("--n", type=int, required=True, help="intermediate size per gate/up branch")
    parser.add_argument("--k", type=int, required=True, help="model dim")
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--m", nargs="+", type=int, default=DEFAULT_M_VALUES)
    parser.add_argument("--arch", default=None, help="default: detected from the device")
    parser.add_argument("--cu", "--num-cu", dest="cu", type=int, default=None,
                        help="default: the device CU count, as used by the runtime lookup")
    parser.add_argument("--families", nargs="+", choices=("decode", "prefill"), default=["decode", "prefill"],
                        help="kernel families to sweep when --gemm{1,2}-modes are not given")
    parser.add_argument("--gemm1-modes", nargs="+", type=int, default=None)
    parser.add_argument("--gemm2-modes", nargs="+", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--max-mismatch", type=float, default=0.01,
                        help="max fraction of elements outside atol/rtol")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="sets HIP_VISIBLE_DEVICES")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "tune/tune_results/w4a16")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fresh", action="store_true", help="ignore candidates logged by an earlier run")
    parser.add_argument("--config-csv", type=Path, default=None,
                        help="CSV to merge tuned rows into (default: <out-dir>/tuned_moe_c_w4a16_group.csv)")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    if args.group_size != 32:
        parser.error("moe_c W4A16 kernels only support --group-size 32 (fixed zero point 8)")
    if args.k % (args.group_size * 4) or args.n % (args.group_size * 4):
        parser.error("K and N must be multiples of 4 * group_size (marlin scale layout)")
    if args.device is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    if args.worker:
        return tune(args)
    if args.fresh:
        for stale in args.out_dir.glob("w4a16_*_candidates.jsonl"):
            stale.unlink()
    return supervise(args, [a for a in argv if a != "--fresh"])


if __name__ == "__main__":
    sys.exit(main())
