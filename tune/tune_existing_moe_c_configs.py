#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Retune every existing HIPC moe_c config into a separate output tree.

The script deliberately never writes back to ``moe_c_configs``.  It discovers
the currently shipped config files, uses their M keys as the tuning workload,
and writes newly tuned configs under ``--out-dir/configs`` with shape, topk,
and CU fields in their names. Raw per-shape tuning logs stay under ``--out-dir/runs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
AITer_ROOT = ROOT.parents[1]
CONFIG_ROOT = ROOT / "moe_c_configs"
TUNE_HIPC = ROOT / "tune" / "tune_moe_hipc.py"
TUNE_W4A8 = ROOT / "tune" / "tune_moe_w4a8_perchannel.py"


@dataclass(frozen=True)
class ConfigShape:
    quant: str
    arch: str
    e: int
    n: int
    k: int
    topk: int
    cu: Optional[int]
    filename: str
    bottom_filename: str
    m_values: Tuple[int, ...]

    @property
    def label(self) -> str:
        cu = "none" if self.cu is None else str(self.cu)
        return f"E{self.e}_N{self.n}_K{self.k}_topk{self.topk}_cu{cu}"

    @property
    def runtime_filename(self) -> str:
        cu = self.cu if self.cu is not None else default_cu_for_arch(self.arch)
        cu_field = f",cu={cu}" if cu is not None else ""
        return f"E={self.e},N={self.n},K={self.k},topk={self.topk}{cu_field},dtype={self.quant}.json"


def default_cu_for_arch(arch: str) -> Optional[int]:
    if arch == "gfx938":
        return 72
    if arch == "gfx936":
        return 80
    return None


def parse_config_name(path: Path, quant: str) -> Optional[Dict[str, int]]:
    if "is_bottom=True" in path.name:
        return None
    match = re.search(
        rf"E=(\d+),N=(\d+)(?:,K=(\d+))?(?:,topk=(\d+))?(?:,cu=(\d+))?,dtype={re.escape(quant)}\.json$",
        path.name,
    )
    if not match:
        return None
    return {
        "e": int(match.group(1)),
        "n": int(match.group(2)),
        "k": int(match.group(3) or 7168),
        "topk": int(match.group(4) or 8),
        "cu": int(match.group(5)) if match.group(5) else None,
    }


def bottom_name(filename: str) -> str:
    return filename[:-5] + ",is_bottom=True.json"


def load_m_values(top_path: Path, bottom_path: Path) -> Tuple[int, ...]:
    keys = set()
    for path in (top_path, bottom_path):
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        keys.update(int(k) for k in data.keys())
    return tuple(sorted(keys))


def discover_shapes(arch: str, quant: str, config_root: Path = CONFIG_ROOT) -> List[ConfigShape]:
    base = config_root / arch / quant
    shapes: List[ConfigShape] = []
    for path in sorted(base.glob("*.json")):
        fields = parse_config_name(path, quant)
        if fields is None:
            continue
        bname = bottom_name(path.name)
        m_values = load_m_values(path, base / bname)
        if not m_values:
            continue
        shapes.append(ConfigShape(
            quant=quant,
            arch=arch,
            e=fields["e"],
            n=fields["n"],
            k=fields["k"],
            topk=fields["topk"],
            cu=fields["cu"],
            filename=path.name,
            bottom_filename=bname,
            m_values=m_values,
        ))
    return shapes


def trim_shape_m_values(shape: ConfigShape, max_m_per_shape: int) -> ConfigShape:
    if max_m_per_shape <= 0 or len(shape.m_values) <= max_m_per_shape:
        return shape
    return ConfigShape(
        quant=shape.quant,
        arch=shape.arch,
        e=shape.e,
        n=shape.n,
        k=shape.k,
        topk=shape.topk,
        cu=shape.cu,
        filename=shape.filename,
        bottom_filename=shape.bottom_filename,
        m_values=shape.m_values[:max_m_per_shape],
    )


def run_cmd(cmd: List[str], env: Dict[str, str], cwd: Path, log_path: Path, timeout: Optional[int] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("RUN: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout}s; killing process\n")
            log.flush()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return 124


def best_rows_from_rows(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    best_top: Dict[str, Dict[str, int]] = {}
    best_bottom: Dict[str, Dict[str, int]] = {}
    for m in sorted({int(r["m"]) for r in rows if r.get("ok") == "True"}):
        passed = [r for r in rows if int(r["m"]) == m and r.get("ok") == "True" and r.get("latency_us")]
        pairs = []
        for bm, sorting in {(int(r["bm"]), r["sorting"]) for r in passed}:
            up = [r for r in passed if r["stage"] == "gemm1" and (int(r["bm"]), r["sorting"]) == (bm, sorting)]
            down = [r for r in passed if r["stage"] == "gemm2" and (int(r["bm"]), r["sorting"]) == (bm, sorting)]
            if up and down:
                g1 = min(up, key=lambda r: float(r["latency_us"]))
                g2 = min(down, key=lambda r: float(r["latency_us"]))
                pairs.append((float(g1["latency_us"]) + float(g2["latency_us"]), g1, g2))
        if pairs:
            _, g1, g2 = min(pairs, key=lambda p: p[0])
            best_top[str(m)] = {"MODE": int(g1["mode"]), "BLOCK_SIZE_M": int(g1["bm"])}
            best_bottom[str(m)] = {"MODE": int(g2["mode"]), "BLOCK_SIZE_M": int(g2["bm"])}
    return best_top, best_bottom


def rows_from_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def tune_w8a8(shape: ConfigShape, args: argparse.Namespace, env: Dict[str, str], manifest: List[Dict]) -> None:
    shape_run = args.out_dir / "runs" / shape.quant / shape.label
    cu = shape.cu if shape.cu is not None else default_cu_for_arch(shape.arch)
    print(f"[tune {shape.quant}] {shape.label} M={list(shape.m_values)}", flush=True)

    all_rows: List[Dict[str, str]] = []
    failed_m: List[Dict[str, object]] = []
    for m in shape.m_values:
        m_run = shape_run / f"M{m}"
        base_cmd = [
            sys.executable,
            str(TUNE_HIPC),
            "--quant", shape.quant,
            "--arch", shape.arch,
            "--e", str(shape.e),
            "--n", str(shape.n),
            "--k", str(shape.k),
            "--topk", str(shape.topk),
            "--m", str(m),
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--candidate-timeout", str(args.candidate_timeout),
            "--bench-direct",
            "--sweep-direct",
        ]
        if cu is not None:
            base_cmd.extend(["--cu", str(cu)])
        if args.device is not None:
            base_cmd.extend(["--device", str(args.device)])

        use_inprocess = args.direct_inprocess
        cmd = [*base_cmd]
        if use_inprocess:
            cmd.append("--direct-inprocess")
        cmd.extend(["--out-dir", str(m_run)])
        rc = run_cmd(cmd, env, ROOT, m_run / "run.log", timeout=args.m_timeout)
        csvs = sorted(m_run.glob("*_direct_bench.csv"))
        if rc != 0 or not csvs:
            fallback_run = shape_run / f"M{m}_isolated"
            cmd = [*base_cmd, "--out-dir", str(fallback_run)]
            rc = run_cmd(cmd, env, ROOT, fallback_run / "run.log", timeout=args.m_timeout)
            csvs = sorted(fallback_run.glob("*_direct_bench.csv"))
        if rc == 0 and csvs:
            all_rows.extend(rows_from_csv(csvs[-1]))
        else:
            failed_m.append({"m": m, "returncode": rc, "csv": str(csvs[-1]) if csvs else ""})

    combined_csv = shape_run / f"{shape.quant}_{shape.arch}_{shape.label}_combined_direct_bench.csv"
    write_csv(combined_csv, all_rows)
    best_top, best_bottom = best_rows_from_rows(all_rows)
    config_dir = args.out_dir / "configs" / shape.arch / shape.quant
    write_json(config_dir / shape.runtime_filename, best_top)
    write_json(config_dir / bottom_name(shape.runtime_filename), best_bottom)
    total_expected = len(shape.m_values) * 2
    total_selected = len(best_top) + len(best_bottom)
    status = "passed" if total_selected == total_expected and not failed_m else "partial"
    detail = {
        "returncode": 0 if not failed_m else 1,
        "csv": str(combined_csv),
        "selected": total_selected,
        "expected": total_expected,
        "candidate_rows": len(all_rows),
        "failed_m": failed_m,
        "top_config": str(config_dir / shape.runtime_filename),
        "bottom_config": str(config_dir / bottom_name(shape.runtime_filename)),
    }
    manifest.append({"shape": asdict(shape), "status": status, **detail})


def tune_w4a8(shape: ConfigShape, args: argparse.Namespace, env: Dict[str, str], manifest: List[Dict]) -> None:
    shape_run = args.out_dir / "runs" / shape.quant / shape.label
    cmd = [
        sys.executable,
        str(TUNE_W4A8),
        "run",
        "--e", str(shape.e),
        "--n", str(shape.n),
        "--k", str(shape.k),
        "--arch", shape.arch,
        "--topk", str(shape.topk),
        "--m", *[str(m) for m in shape.m_values],
        "--build-dir", str(args.build_dir),
        "--out", str(shape_run),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--timeout", str(args.candidate_timeout),
    ]
    cu = shape.cu if shape.cu is not None else default_cu_for_arch(shape.arch)
    if cu is not None:
        cmd.extend(["--cu", str(cu)])
    print(f"[tune {shape.quant}] {shape.label} M={list(shape.m_values)}", flush=True)
    rc = run_cmd(cmd, env, ROOT, shape_run / "run.log")
    status = "failed"
    detail = {"returncode": rc}
    top_src = shape_run / shape.filename
    bottom_src = shape_run / shape.bottom_filename
    if not top_src.exists() or not bottom_src.exists():
        top_matches = sorted(p for p in shape_run.glob(f"E={shape.e},N={shape.n},K={shape.k},*dtype={shape.quant}.json") if "is_bottom=True" not in p.name)
        bottom_matches = sorted(shape_run.glob(f"E={shape.e},N={shape.n},K={shape.k},*dtype={shape.quant},is_bottom=True.json"))
        if top_matches:
            top_src = top_matches[-1]
        if bottom_matches:
            bottom_src = bottom_matches[-1]
    if rc == 0 and top_src.exists() and bottom_src.exists():
        config_dir = args.out_dir / "configs" / shape.arch / shape.quant
        top = json.loads(top_src.read_text())
        bottom = json.loads(bottom_src.read_text())
        write_json(config_dir / shape.runtime_filename, top)
        write_json(config_dir / bottom_name(shape.runtime_filename), bottom)
        total_expected = len(shape.m_values) * 2
        total_selected = len(top) + len(bottom)
        status = "passed" if total_selected == total_expected else "partial"
        detail.update({
            "selected": total_selected,
            "expected": total_expected,
            "top_config": str(config_dir / shape.runtime_filename),
            "bottom_config": str(config_dir / bottom_name(shape.runtime_filename)),
        })
    manifest.append({"shape": asdict(shape), "status": status, **detail})


def tune_inprocess(shape: ConfigShape, args: argparse.Namespace,
                   env: Dict[str, str], manifest: List[Dict]) -> None:
    """Reuse imports and kernel modules for all M values of one shape."""
    run = args.out_dir / "runs" / shape.quant / shape.label
    cu = shape.cu if shape.cu is not None else default_cu_for_arch(shape.arch)
    cmd = [sys.executable, str(TUNE_HIPC), "--quant", shape.quant,
           "--arch", shape.arch, "--dtype", "bf16", "--e", str(shape.e),
           "--n", str(shape.n), "--k", str(shape.k), "--topk", str(shape.topk),
           "--m", *map(str, shape.m_values), "--warmup", str(args.warmup),
           "--iters", str(args.iters), "--bench-direct", "--sweep-direct",
           "--direct-inprocess", "--out-dir", str(run)]
    if cu is not None:
        cmd.extend(["--cu", str(cu)])
    print(f"[tune {shape.quant}] {shape.label} M={list(shape.m_values)}", flush=True)
    rc = run_cmd(cmd, env, ROOT, run / "run.log")
    csvs = sorted(run.glob("*_direct_bench.csv"))
    rows = rows_from_csv(csvs[-1]) if csvs else []
    top, bottom = best_rows_from_rows(rows)
    dest = args.out_dir / "configs" / shape.arch / shape.quant
    write_json(dest / shape.runtime_filename, top)
    write_json(dest / bottom_name(shape.runtime_filename), bottom)
    missing = sorted(set(shape.m_values) - {int(m) for m in top})
    manifest.append({"shape": asdict(shape), "status": "passed" if not missing and rc == 0 else "partial",
                     "returncode": rc, "candidate_rows": len(rows), "missing_m": missing,
                     "selected": len(top) + len(bottom), "expected": 2 * len(shape.m_values),
                     "csv": str(csvs[-1]) if csvs else "",
                     "top_config": str(dest / shape.runtime_filename),
                     "bottom_config": str(dest / bottom_name(shape.runtime_filename))})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx938")
    parser.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--quant", nargs="+", default=["fp8_w8a8", "int8_w8a8", "int8_w4a8"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path(os.environ.get("MOE_C_BUILD_DIR", ROOT / "build")))
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--candidate-timeout", type=int, default=120)
    parser.add_argument("--m-timeout", type=int, default=240, help="outer timeout for one W8A8 M in direct-inprocess mode")
    parser.add_argument("--limit", type=int, default=0, help="debug only: limit shapes per quant")
    parser.add_argument("--max-m-per-shape", type=int, default=0, help="debug only: limit M keys per shape")
    parser.add_argument("--shard-index", type=int, default=0, help="process shapes whose index %% shard-count equals this value")
    parser.add_argument("--shard-count", type=int, default=1, help="number of shape shards")
    parser.add_argument("--direct-inprocess", action="store_true", help="run all channelwise candidates and M values in one process per shape")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.build_dir = args.build_dir.resolve()

    env = os.environ.copy()
    env["MOE_C_BUILD_DIR"] = str(args.build_dir)
    env["PYTHONPATH"] = f"{AITer_ROOT / 'aiter'}:{ROOT / 'test'}:{ROOT}:{env.get('PYTHONPATH', '')}"
    if args.device is not None:
        env["HIP_VISIBLE_DEVICES"] = str(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict] = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for quant in args.quant:
        shapes = discover_shapes(args.arch, quant, args.config_root)
        if args.shard_count > 1:
            shapes = [shape for idx, shape in enumerate(shapes) if idx % args.shard_count == args.shard_index]
        if args.limit:
            shapes = shapes[:args.limit]
        print(f"[discover] quant={quant} shapes={len(shapes)} shard={args.shard_index}/{args.shard_count}", flush=True)
        for shape in shapes:
            shape = trim_shape_m_values(shape, args.max_m_per_shape)
            if args.direct_inprocess:
                tune_inprocess(shape, args, env, manifest)
            elif quant == "int8_w4a8":
                tune_w4a8(shape, args, env, manifest)
            elif quant in ("fp8_w8a8", "int8_w8a8"):
                tune_w8a8(shape, args, env, manifest)
            else:
                manifest.append({"shape": asdict(shape), "status": "skipped", "error": f"unsupported quant={quant}"})
            write_json(args.out_dir / "manifest.json", {
                "started_at": started,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "arch": args.arch,
                "build_dir": str(args.build_dir),
                "device": args.device,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "results": manifest,
            })

    failed = [row for row in manifest if row.get("status") not in ("passed",)]
    print(f"[done] total={len(manifest)} passed={len(manifest)-len(failed)} non_passed={len(failed)}", flush=True)
    if failed:
        for row in failed:
            shape = row["shape"]
            print(f"[non-pass] {shape['quant']} {shape['filename']} status={row.get('status')} rc={row.get('returncode')}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
