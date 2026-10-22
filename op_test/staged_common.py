# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared helpers for the ``*_staged.py`` stage-level MoE benchmarks.

All staged benchmarks share one measurement model, so their numbers are
comparable with each other and with the endpoint tests:

* ``e2e_us``: endpoint device time from the shared ``perftest`` helper.
* ``<stage>_us``: device time of the kernels belonging to a stage, classified
  by kernel name from a profiler run of the same callable.
* ``gap_ratio``: ``(stage_sum - e2e) / e2e``.  A large value usually means a
  kernel is landing in ``other`` and needs a rule in ``_KERNEL_RULES``.

A benchmark script only supplies a ``run`` callable (one full MoE call), an
optional ``check`` callable and its own CLI extras; everything else lives here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.profiler as tpf

OP_TEST_DIR = Path(__file__).resolve().parent
MOE_C_ROOT = OP_TEST_DIR.parent
for _path in (MOE_C_ROOT, OP_TEST_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from moe_utils.moe_ops import load_moe_c_library  # noqa: E402
from moe_utils.utils import dtypes, perftest  # noqa: E402

# Not a pytest module.
__test__ = False

STAGE_NAMES = (
    "routing",
    "quant1",
    "gemm1",
    "activation",
    "quant2",
    "gemm2",
    "sum",
    "other",
)

DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1024, 2048, 4096, 6144, 8192, 16384]

# (stage, kernel-name substrings), first match wins, matched case-insensitively.
# "quant" and "gemm" are resolved to stage 1 or 2 by order within an iteration.
_KERNEL_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("routing", ("moe_align", "token_sort", "moe_sorting", "moesorting",
                 "fillfunctor")),
    ("quant", ("per_token_quant", "per_token_scaled_quant")),
    ("gemm", ("moe_w4a8", "moe_w8a8", "gemm_nt_", "marlin")),
    ("activation", ("glu_kernel", "silu")),
    ("sum", ("moe_sum",)),
)


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    for stage, needles in _KERNEL_RULES:
        if any(needle in lowered for needle in needles):
            return stage
    return "other"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def profile_stages(
    fn: Callable[[], Any], warmup: int, iters: int
) -> Tuple[Dict[str, float], List[str]]:
    """Mean per-iteration device time (us) per stage, plus unclassified names."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with tpf.profile(activities=[tpf.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    kernels = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )

    totals = {name: 0.0 for name in STAGE_NAMES}
    unknown: List[str] = []
    quant_seen = gemm_seen = 0
    for event in kernels:
        stage = classify_kernel(event.name)
        if stage == "routing" and gemm_seen:
            quant_seen = gemm_seen = 0  # routing after the GEMMs starts a new call
        if stage == "quant":
            quant_seen += 1
            stage = "quant1" if quant_seen == 1 else "quant2"
        elif stage == "gemm":
            gemm_seen += 1
            stage = "gemm1" if gemm_seen == 1 else "gemm2"
        elif stage == "other" and event.name not in unknown:
            unknown.append(event.name)
        totals[stage] += float(event.device_time)

    return {k: v / iters for k, v in totals.items()}, unknown


def profile_e2e(
    fn: Callable[[], Any], warmup: int, iters: int, test_graph: bool = True
) -> float:
    """Endpoint device time using the same helper as the endpoint tests."""

    @perftest(num_warmup=warmup, num_iters=iters, num_rotate_args=1,
              testGraph=test_graph)
    def worker() -> Any:
        return fn()

    _, average_us = worker()
    return float(average_us)


def measure(
    run: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    check: Optional[Callable[[torch.Tensor], None]] = None,
    test_graph: bool = True,
) -> Dict[str, Any]:
    """Run the optional correctness check, then measure e2e and every stage."""
    if warmup < 0 or iters <= 1:
        raise ValueError("warmup must be >= 0 and iters must be > 1")

    with torch.inference_mode():
        if check is not None:
            output = run()
            torch.cuda.synchronize()
            check(output)
        e2e_us = profile_e2e(run, warmup, iters, test_graph)
        stages, unknown = profile_stages(run, warmup, iters)

    if unknown:
        print(f"[staged] unclassified kernels (add a rule?): {unknown}",
              file=sys.stderr, flush=True)
    stage_sum_us = sum(stages.values())
    gap_us = stage_sum_us - e2e_us
    row: Dict[str, Any] = {f"{name}_us": value for name, value in stages.items()}
    row.update(
        correctness="passed" if check is not None else "skipped",
        stage_sum_us=stage_sum_us,
        e2e_us=e2e_us,
        gap_us=gap_us,
        gap_ratio=gap_us / e2e_us if e2e_us else 0.0,
    )
    return row


# ---------------------------------------------------------------------------
# Environment and reporting
# ---------------------------------------------------------------------------

def select_build_dir(build_dir: Optional[str], libraries: Sequence[str]) -> Path:
    """Validate the split-library directory and load it."""
    selected = Path(build_dir).resolve() if build_dir else (MOE_C_ROOT / "build")
    missing = [n for n in ("libmoe_c_common.so", *libraries)
               if not (selected / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing split libraries in {selected}: {', '.join(missing)}"
        )
    os.environ["MOE_C_BUILD_DIR"] = str(selected)
    load_moe_c_library()
    return selected


def parse_dtype(name: str) -> torch.dtype:
    return {"bf16": dtypes.bf16, "fp16": dtypes.fp16}[name]


def base_parser(description: str, *, e: int, n: int, k: int,
                topk: int = 8, iters: int = 100) -> argparse.ArgumentParser:
    """CLI flags shared by every staged benchmark."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--tokens", "--m", nargs="+", type=int, default=DEFAULT_TOKENS,
                   help="M values to benchmark (space-separated)")
    p.add_argument("--e", type=int, default=e)
    p.add_argument("--n", type=int, default=n)
    p.add_argument("--k", type=int, default=k)
    p.add_argument("--topk", type=int, default=topk)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=iters)
    p.add_argument("--build-dir", default=None,
                   help="Directory containing split moe_c .so files")
    p.add_argument("--output", default=None, help="Optional CSV output path")
    p.add_argument("--no-check", action="store_true",
                   help="Skip the reference correctness check")
    p.add_argument("--json", action="store_true", help="Print rows as JSON")
    return p


_TABLE_COLUMNS = (
    ["m", "MODE", "BLOCK_SIZE_M"]
    + [f"{name}_us" for name in STAGE_NAMES]
    + ["stage_sum_us", "e2e_us", "gap_ratio", "correctness"]
)


def print_table(rows: List[Dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    cols = [c for c in _TABLE_COLUMNS if c in frame.columns]
    print("\n=== Staged benchmark (device time, us) ===")
    with pd.option_context("display.max_columns", None, "display.width", 2000):
        print(frame[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def print_breakdown(row: Dict[str, Any]) -> None:
    total = row["stage_sum_us"] or 1.0
    print(f"\n>>> Breakdown for M={row['m']}")
    print(f"{'stage':>12} | {'us':>10} | {'% of sum':>9}")
    for name in STAGE_NAMES:
        value = row[f"{name}_us"]
        print(f"{name:>12} | {value:>10.3f} | {value / total * 100:>8.2f}%")
    print(f"{'stage_sum':>12} | {row['stage_sum_us']:>10.3f}")
    print(f"{'e2e':>12} | {row['e2e_us']:>10.3f}")
    print(f"{'gap':>12} | {row['gap_us']:>+10.3f} ({row['gap_ratio']:+.2%})")


def run_sweep(
    args: argparse.Namespace,
    case_fn: Callable[[int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run ``case_fn(m)`` for each token count, report, and exit non-zero on failure."""
    tokens = sorted({m for m in args.tokens if m > 0})
    if not tokens:
        sys.exit("--tokens must contain at least one positive integer")

    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for m in tokens:
        try:
            rows.append(case_fn(m))
        except Exception as exc:  # noqa: BLE001 - keep sweeping the other M values
            failures.append(f"M={m}: {exc}")
            print(f"[FAILED] M={m}: {exc}", file=sys.stderr, flush=True)

    if not rows:
        sys.exit("No case completed successfully.")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        print_table(rows)
        if len(rows) == 1:
            print_breakdown(rows[0])
        if args.output:
            print(f"[info] saved results to {args.output}")
    if failures:
        print("\n" + "\n".join(failures), file=sys.stderr)
        sys.exit(1)
    return rows
