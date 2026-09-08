#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Tune WFP4A8 groupwise moe_c split kernels.

The script calls 3rdparty/moe_c split shared libraries directly. It measures
the C++ wrapper time files and writes per-M best configs for Kimi-K3 style
MXFP4 groupwise WFP4A8.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


MOE_C_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
AITER_ROOT = MOE_C_ROOT.parents[1]

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


@dataclass(frozen=True)
class Candidate:
    stage: str
    mode: int
    bm: int
    bn: int
    bk: int
    note: str

    def config(self) -> dict[str, int]:
        return {"BLOCK_SIZE_M": int(self.bm), "MODE": int(self.mode)}


@dataclass
class Result:
    stage: str
    m: int
    mode: int
    bm: int
    status: str
    mean_ms: float | None
    median_ms: float | None
    tflops: float | None
    note: str
    error: str = ""


GROUPWISE_DECODE_GEMM1 = [
    Candidate("gemm1", 201, 16, 128, 64, "decode-gemm1-bn128"),
    Candidate("gemm1", 221, 16, 32, 64, "decode-gemm1-bn32"),
]

GROUPWISE_DECODE_GEMM1_LEGACY = [
    Candidate("gemm1", 9524, 16, 128, 64, "legacy-9524"),
    Candidate("gemm1", 9121, 16, 32, 64, "legacy-9121"),
]

GROUPWISE_DECODE_GEMM2 = [
    Candidate("gemm2", 202, 16, 128, 64, "decode-gemm2-bn128"),
]

GROUPWISE_DECODE_GEMM2_LEGACY = [
    Candidate("gemm2", 9525, 16, 128, 64, "legacy-9525"),
]

GROUPWISE_PREFILL_GEMM1 = [
    Candidate("gemm1", 601, 64, 128, 64, "prefill-gemm1-bm64"),
]

GROUPWISE_PREFILL_GEMM1_LEGACY = [
    Candidate("gemm1", 9537, 64, 128, 64, "legacy-9537"),
]

GROUPWISE_PREFILL_GEMM2 = [
    Candidate("gemm2", 602, 64, 128, 64, "prefill-gemm2-bm64-k192"),
]

GROUPWISE_PREFILL_GEMM2_LEGACY = [
    Candidate("gemm2", 9523, 64, 128, 64, "legacy-9523"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--m", type=int, nargs="+", default=M_VALUES)
    common.add_argument("--e", type=int, default=896)
    common.add_argument("--n", type=int, default=192, help="GEMM2 input/intermediate N. GEMM1 output is 2*N.")
    common.add_argument("--k", type=int, default=3584, help="hidden size")
    common.add_argument("--topk", type=int, default=16)
    common.add_argument("--warmup", type=int, default=3)
    common.add_argument("--iters", type=int, default=20)
    common.add_argument("--device", type=int, default=0)
    common.add_argument("--build-dir", default=str(MOE_C_ROOT / "build"))
    common.add_argument("--out-dir", default=str(TEST_ROOT / "wfp4a8_groupwise_tune_results"))
    common.add_argument("--no-shuffle", action="store_true", help="Use contiguous random packed weights directly.")

    run = sub.add_parser("run", parents=[common], help="tune groupwise configs")
    run.add_argument("--prefill-threshold", type=int, default=1024)
    run.add_argument("--include-legacy", action="store_true", help="Also test legacy experiment mode aliases.")

    bench = sub.add_parser("bench-config", parents=[common], help="benchmark explicit groupwise config")
    bench.add_argument("--gemm1-mode", type=int, required=True)
    bench.add_argument("--gemm2-mode", type=int, required=True)
    bench.add_argument("--block-size-m", type=int, required=True)

    return parser.parse_args()


def add_import_paths(build_dir: str) -> None:
    os.environ["MOE_C_BUILD_DIR"] = build_dir
    os.environ["MOE_C_PREFER_SPLIT"] = "1"
    os.environ["WHICH_TO_TEST"] = "1"
    sys.path.insert(0, str(Path(build_dir).resolve()))
    sys.path.insert(0, str(TEST_ROOT))
    sys.path.insert(0, str(AITER_ROOT))


def fp8_bytes(torch_mod, rows: int, cols: int):
    return torch_mod.randint(1, 127, (rows, cols), device="cuda", dtype=torch_mod.uint8)


def make_align(ops, torch_mod, m: int, topk: int, e: int, bm: int):
    topk_ids = torch_mod.randint(0, e, (m, topk), device="cuda", dtype=torch_mod.int32)
    max_padded = m * topk + e * (bm - 1)
    sorted_ids = torch_mod.empty((max_padded,), device="cuda", dtype=torch_mod.int32)
    sorted_ids.fill_(m * topk)
    expert_ids = torch_mod.empty(((max_padded + bm - 1) // bm,), device="cuda", dtype=torch_mod.int32)
    num_post = torch_mod.empty((1,), device="cuda", dtype=torch_mod.int32)
    ops.moe_align_block_size(topk_ids, e, bm, sorted_ids, expert_ids, num_post)
    return sorted_ids, expert_ids, num_post


def maybe_shuffle_pair(w1, w2, no_shuffle: bool):
    if no_shuffle:
        return w1.contiguous(), w2.contiguous()
    try:
        from aiter.moe import AiterMoeConfig, MoeQuantType, MoeSolutionType, aiter_moe_shfl_weight
    except Exception:
        return w1.contiguous(), w2.contiguous()
    cfg = AiterMoeConfig(
        quant_type=MoeQuantType.WFP4A8,
        solution_type=MoeSolutionType.MOE_C,
        config={},
        need_shuffle=True,
        need_shuffle_scale=False,
    )
    try:
        return aiter_moe_shfl_weight(w1, w2, cfg, block_shape=[0, 32])
    except Exception:
        return w1.contiguous(), w2.contiguous()


def make_weights(torch_mod, e: int, n: int, k: int, no_shuffle: bool):
    w1 = torch_mod.randint(0, 256, (e, 2 * n, k // 2), device="cuda", dtype=torch_mod.uint8)
    w2 = torch_mod.randint(0, 256, (e, k, n // 2), device="cuda", dtype=torch_mod.uint8)
    w1, w2 = maybe_shuffle_pair(w1, w2, no_shuffle)
    s1 = torch_mod.randint(121, 126, (e, 2 * n, k // 32), device="cuda", dtype=torch_mod.uint8)
    s2 = torch_mod.randint(121, 126, (e, k, n // 32), device="cuda", dtype=torch_mod.uint8)
    return {
        "w1": w1.contiguous(),
        "w2": w2.contiguous(),
        "s1": s1.contiguous(),
        "s2": s2.contiguous(),
    }


def candidates_for(stage: str, m: int, prefill_threshold: int, include_legacy: bool) -> list[Candidate]:
    prefill = m >= prefill_threshold
    if stage == "gemm1":
        base = GROUPWISE_PREFILL_GEMM1 if prefill else GROUPWISE_DECODE_GEMM1
        legacy = GROUPWISE_PREFILL_GEMM1_LEGACY if prefill else GROUPWISE_DECODE_GEMM1_LEGACY
    else:
        base = GROUPWISE_PREFILL_GEMM2 if prefill else GROUPWISE_DECODE_GEMM2
        legacy = GROUPWISE_PREFILL_GEMM2_LEGACY if prefill else GROUPWISE_DECODE_GEMM2_LEGACY
    return base + legacy if include_legacy else base


def time_file(stage: str) -> Path:
    return Path(f"wfp4a8_kernel_{1 if stage == 'gemm1' else 2}_timecost")


def read_times(path: Path, iters: int) -> list[float]:
    if not path.exists():
        return []
    vals: list[float] = []
    for line in path.read_text().splitlines():
        try:
            vals.append(float(line.strip()))
        except ValueError:
            continue
    return vals[-iters:]


def flops(stage: str, m: int, n: int, k: int, topk: int) -> int:
    if stage == "gemm1":
        return 2 * m * topk * (2 * n) * k
    return 2 * m * topk * k * n


def run_candidate(
    ops,
    torch_mod,
    op: Callable,
    cand: Candidate,
    m: int,
    e: int,
    n: int,
    k: int,
    topk: int,
    tensors: dict[str, object],
    warmup: int,
    iters: int,
) -> Result:
    sorted_ids, expert_ids, num_post = make_align(ops, torch_mod, m, topk, e, cand.bm)
    tfile = time_file(cand.stage)
    try:
        tfile.unlink()
    except FileNotFoundError:
        pass

    if cand.stage == "gemm1":
        qinput = fp8_bytes(torch_mod, m, k)
        a_scale = torch_mod.ones((m, 1), device="cuda", dtype=torch_mod.float32)
        output = torch_mod.empty((m, topk, 2 * n), device="cuda", dtype=torch_mod.bfloat16)
        call_args = (
            qinput, tensors["w1"], output, a_scale, tensors["s1"], None,
            sorted_ids, expert_ids, num_post, topk, cand.mode, topk, m,
        )
    else:
        qinput = fp8_bytes(torch_mod, m * topk, n)
        a_scale = torch_mod.ones((m * topk, 1), device="cuda", dtype=torch_mod.float32)
        output = torch_mod.empty((m, topk, k), device="cuda", dtype=torch_mod.bfloat16)
        topk_weights = torch_mod.rand((m, topk), device="cuda", dtype=torch_mod.float32)
        call_args = (
            qinput, tensors["w2"], output, a_scale, tensors["s2"], topk_weights,
            sorted_ids, expert_ids, num_post, 1, cand.mode, topk, m,
        )

    try:
        for _ in range(warmup + iters):
            op(*call_args)
        torch_mod.cuda.synchronize()
        times = read_times(tfile, iters)
        if not times:
            raise RuntimeError(f"no kernel times written to {tfile}")
        mean_ms = statistics.fmean(times)
        median_ms = statistics.median(times)
        perf = flops(cand.stage, m, n, k, topk) / (mean_ms * 1e-3) / 1e12
        return Result(cand.stage, m, cand.mode, cand.bm, "ok", mean_ms, median_ms, perf, cand.note)
    except Exception as exc:
        return Result(cand.stage, m, cand.mode, cand.bm, "fail", None, None, None, cand.note, str(exc).splitlines()[0][:240])


def best_result(results: Iterable[Result]) -> Result | None:
    ok = [r for r in results if r.status == "ok" and r.tflops is not None]
    if not ok:
        return None
    max_tflops = max(r.tflops or 0.0 for r in ok)
    close = [r for r in ok if (r.tflops or 0.0) >= max_tflops * 0.995]
    return min(close, key=lambda r: r.mode)


def print_result(result: Result) -> None:
    if result.status == "ok":
        print(
            f"M={result.m:5d} {result.stage} mode={result.mode:5d} bm={result.bm:2d} "
            f"{result.mean_ms:.6f} ms {result.tflops:.2f} TFLOPS {result.note}"
        )
    else:
        print(f"M={result.m:5d} {result.stage} mode={result.mode:5d} bm={result.bm:2d} FAIL {result.error}")


def run_tune(args: argparse.Namespace) -> None:
    add_import_paths(args.build_dir)

    import torch
    import _custom_ops as ops

    torch.cuda.set_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    op = ops.moe_c_moe_gemm_marlin_wfp4a8_groupwise
    tensors = make_weights(torch, args.e, args.n, args.k, args.no_shuffle)

    all_results: list[Result] = []
    best_gemm1: dict[str, dict[str, int]] = {}
    best_gemm2: dict[str, dict[str, int]] = {}
    best_rows: list[dict[str, object]] = []

    for m in args.m:
        for stage in ("gemm1", "gemm2"):
            local: list[Result] = []
            for cand in candidates_for(stage, m, args.prefill_threshold, args.include_legacy):
                res = run_candidate(
                    ops, torch, op, cand, m, args.e, args.n, args.k, args.topk,
                    tensors, args.warmup, args.iters,
                )
                print_result(res)
                local.append(res)
                all_results.append(res)
            best = best_result(local)
            if best is not None:
                best_rows.append(asdict(best))
                target = best_gemm1 if stage == "gemm1" else best_gemm2
                target[str(m)] = {"BLOCK_SIZE_M": int(best.bm), "MODE": int(best.mode)}
                print(
                    f"BEST M={m:5d} {stage} mode={best.mode} "
                    f"{best.mean_ms:.6f} ms {best.tflops:.2f} TFLOPS"
                )

    write_outputs(out_dir, args, all_results, best_rows, best_gemm1, best_gemm2)


def bench_config(args: argparse.Namespace) -> None:
    add_import_paths(args.build_dir)

    import torch
    import _custom_ops as ops

    torch.cuda.set_device(args.device)
    op = ops.moe_c_moe_gemm_marlin_wfp4a8_groupwise
    tensors = make_weights(torch, args.e, args.n, args.k, args.no_shuffle)
    all_results: list[Result] = []
    for m in args.m:
        for cand in (
            Candidate("gemm1", args.gemm1_mode, args.block_size_m, 128, 64, "explicit"),
            Candidate("gemm2", args.gemm2_mode, args.block_size_m, 128, 64, "explicit"),
        ):
            res = run_candidate(
                ops, torch, op, cand, m, args.e, args.n, args.k, args.topk,
                tensors, args.warmup, args.iters,
            )
            print_result(res)
            all_results.append(res)
    write_outputs(Path(args.out_dir), args, all_results, [asdict(r) for r in all_results], {}, {})


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    all_results: list[Result],
    best_rows: list[dict[str, object]],
    best_gemm1: dict[str, dict[str, int]],
    best_gemm2: dict[str, dict[str, int]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_results:
        csv_path = out_dir / "wfp4a8_groupwise_tune_results.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for row in all_results:
                writer.writerow(asdict(row))
        print(f"Wrote {csv_path}")

    (out_dir / "wfp4a8_groupwise_tune_best.json").write_text(
        json.dumps(best_rows, indent=2), encoding="utf-8"
    )
    if best_gemm1:
        (out_dir / f"E={args.e},N={args.n},K={args.k},dtype=fp4_w4a8.json").write_text(
            json.dumps(best_gemm1, indent=2), encoding="utf-8"
        )
    if best_gemm2:
        (out_dir / f"E={args.e},N={args.n},K={args.k},dtype=fp4_w4a8,is_bottom=True.json").write_text(
            json.dumps(best_gemm2, indent=2), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    if args.cmd == "run":
        run_tune(args)
    elif args.cmd == "bench-config":
        bench_config(args)


if __name__ == "__main__":
    main()
