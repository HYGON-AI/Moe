#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Local moe_c W4A8 per-channel tuner.

This script tunes against 3rdparty/moe_c local shared libraries instead of the
top-level AITER JIT module. Edit the DEFAULT_* block below for normal use.
"""

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MOE_C_ROOT = Path(__file__).resolve().parents[1]
AITER_ROOT = MOE_C_ROOT.parents[1]
TEST_ROOT = Path(__file__).resolve().parent

DEFAULT_E = 192
DEFAULT_N = 384
DEFAULT_K = 4096
DEFAULT_TOPK = 8
DEFAULT_DTYPE = "bf16"
DEFAULT_BUILD_DIR = os.environ.get("MOE_C_BUILD_DIR", "/tmp/moe_c_w4a8_tune_split_build_zxl")
DEFAULT_OUT = None
DEFAULT_MAX_NLOOP = 4
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT = 600
DEFAULT_ATOL = 0.2
DEFAULT_RTOL = 0.01
DEFAULT_ROUTE_CHUNK = 16
DEFAULT_GOLDEN = "torch"

DEFAULT_M_VALUES = [
    1, 2, 3, 4, 5, 6, 7, 8,
    16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 6144, 8192, 16384,
]


@dataclass(frozen=True)
class Shape:
    e: int
    n: int
    k: int
    topk: int
    dtype: str
    m_values: List[int]
    cu: Optional[int] = None


def default_cu_for_arch(arch: Optional[str]) -> Optional[int]:
    if arch == "gfx938":
        return 72
    if arch == "gfx936":
        return 80
    return None


def resolve_cu(arch: Optional[str], cu: Optional[int]) -> Optional[int]:
    return cu if cu is not None else default_cu_for_arch(arch)


def config_file_name(shape: Shape, is_bottom: bool = False) -> str:
    suffix = ",is_bottom=True" if is_bottom else ""
    if shape.cu is not None:
        return (f"E={shape.e},N={shape.n},K={shape.k},topk={shape.topk},"
                f"cu={shape.cu},dtype=int8_w4a8{suffix}.json")
    return f"E={shape.e},N={shape.n},K={shape.k},dtype=int8_w4a8{suffix}.json"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    gemm: str
    mode: int
    block_size_m: int
    block_size_n: int
    block_size_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    n_loop: int
    note: str
    stages: int = 2
    k_step: int = 128
    k_tail: Optional[int] = None
    sorting: int = 1

    def config(self) -> Dict[str, int]:
        return {"BLOCK_SIZE_M": int(self.block_size_m), "MODE": int(self.mode)}


def generate_candidates(shape: Shape, max_nloop: int, arch: str = "gfx936") -> List[Candidate]:
    if __package__:
        from .tune_moe_hipc import PROFILES, Shape as TuneShape
    else:
        from tune_moe_hipc import PROFILES, Shape as TuneShape

    profile = PROFILES["int8_w4a8"]
    tune_shape = TuneShape(shape.e, shape.n, shape.k, shape.topk, 0,
                          shape.dtype, arch, shape.cu)
    candidates = []
    for base_desc in profile.descriptors.values():
        desc = base_desc.for_shape(tune_shape)
        if desc.family in (2, 3) and desc.nloop > max_nloop:
            continue
        if desc.validate(tune_shape) is not None:
            continue
        candidates.append(Candidate(
            candidate_id=f"g{desc.stage}_f{desc.family}_bm{desc.bm}_bn{desc.bn}_bk{desc.bk}_nl{desc.nloop}",
            gemm="gemm1" if desc.stage == 1 else "gemm2",
            mode=desc.mode, block_size_m=desc.bm, block_size_n=desc.bn,
            block_size_k=desc.bk, warp_m=desc.wm, warp_n=desc.wn,
            warp_k=desc.wk, n_loop=desc.nloop, note=desc.kernel,
            stages=desc.stages, k_step=desc.k_step, k_tail=desc.k_tail,
            sorting=desc.sorting,
        ))
    return candidates


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def dtype_from_name(name: str):
    import torch
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype={name}")


def cache_file(out_dir: Path, shape: Shape, m: int, golden: str) -> Path:
    out_dir = out_dir.resolve()
    return out_dir / "golden_cache" / (
        f"E={shape.e},N={shape.n},K={shape.k},topk={shape.topk},"
        f"dtype={shape.dtype},golden={golden},M={m}.pt"
    )


def default_out_dir(shape: Shape) -> Path:
    return TEST_ROOT / f"moe_c_w4a8_tune_E{shape.e}_N{shape.n}"


def load_golden_cache_cpu(cache_path: Path) -> Dict[str, Any]:
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            cached = torch.load(cache_path, map_location="cpu")
    return cached


def load_golden_cache(cache_path: Path) -> Dict[str, Any]:
    import torch

    cached = load_golden_cache_cpu(cache_path)
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in cached.items()}


def unlink_cache(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"WARN failed to remove cache {path}: {exc}", flush=True)


def _torch_per_token_quant_int8(x, eps: float = 1e-10):
    import torch

    x_f = x.to(torch.float32)
    scale = torch.amax(torch.abs(x_f), dim=-1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.round(x_f / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def _pack_and_shuffle_one_expert(unsigned_weight):
    import torch
    from aiter.ops.shuffle import w4a8_moe_layout_shuffle

    n, k = unsigned_weight.shape
    w_unsigned = unsigned_weight.to(torch.uint8) & 0x0F
    temp_view = w_unsigned.view(n, k // 8, 8)
    w_low = temp_view[..., :4].reshape(n, k // 2)
    w_high = temp_view[..., 4:].reshape(n, k // 2)
    packed = (w_low << 4) | w_high
    packed = packed.to(torch.int8)
    shuffled = w4a8_moe_layout_shuffle(packed).contiguous().view_as(packed)
    del w_unsigned, temp_view, w_low, w_high, packed
    return shuffled


def run_low_peak_torch_golden(
    hidden_states,
    w1_ref,
    w2_ref,
    topk_weights,
    topk_ids,
    out_dtype,
    w1_scales,
    w2_scales,
):
    import torch

    with torch.no_grad():
        hidden_states = hidden_states.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()
        if w1_scales.shape[-1] == 1:
            w1_scales = w1_scales.squeeze(-1)
        if w2_scales.shape[-1] == 1:
            w2_scales = w2_scales.squeeze(-1)

        m = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        inter = w1_ref.shape[1] // 2
        hidden_out = w2_ref.shape[1]
        qinput1, a1_scale = _torch_per_token_quant_int8(hidden_states)
        input1 = qinput1.to(torch.float32) * a1_scale
        out_acc = torch.zeros((m, hidden_out), device=hidden_states.device, dtype=torch.float32)

        for token_idx in range(m):
            a1 = input1[token_idx]
            for topk_idx in range(topk):
                expert_id = int(topk_ids[token_idx, topk_idx].item())
                w1 = w1_ref[expert_id].to(torch.float32)
                s1 = w1_scales[expert_id].to(torch.float32).unsqueeze(-1)
                stage1 = torch.matmul(w1 * s1, a1).to(out_dtype)
                del w1, s1

                activated = (
                    torch.nn.functional.silu(stage1[:inter].to(torch.float32))
                    * stage1[inter:].to(torch.float32)
                ).to(out_dtype)
                qinput2, a2_scale = _torch_per_token_quant_int8(activated.unsqueeze(0))
                a2 = (qinput2[0].to(torch.float32) * a2_scale[0])
                w2 = w2_ref[expert_id].to(torch.float32)
                s2 = w2_scales[expert_id].to(torch.float32).unsqueeze(-1)
                route_out = torch.matmul(w2 * s2, a2)
                out_acc[token_idx] += route_out * topk_weights[token_idx, topk_idx].to(torch.float32)
                del stage1, activated, qinput2, a2_scale, a2, w2, s2, route_out

        return out_acc.to(out_dtype)


def prepare_low_peak_inputs(shape: Shape, m: int, need_ref: bool, golden: str = "torch") -> Dict[str, Any]:
    import torch

    if golden == "triton" and need_ref:
        raise ValueError("triton golden is not supported in low-peak correctness mode")

    torch.manual_seed(0)
    torch.set_default_device("cuda")
    dtype = dtype_from_name(shape.dtype)
    device = "cuda"
    n1 = 2 * shape.n
    w1_shape = (shape.e, n1, shape.k // 2)
    w2_shape = (shape.e, shape.k, shape.n // 2)
    w1_run = torch.empty(w1_shape, device=device, dtype=torch.int8)
    w2_run = torch.empty(w2_shape, device=device, dtype=torch.int8)
    w1_ref = torch.empty((shape.e, n1, shape.k), device=device, dtype=torch.int8) if need_ref else None
    w2_ref = torch.empty((shape.e, shape.k, shape.n), device=device, dtype=torch.int8) if need_ref else None

    for expert in range(shape.e):
        w1_unsigned = torch.randint(0, 15, (n1, shape.k), device=device, dtype=torch.int8)
        w1_run[expert].copy_(_pack_and_shuffle_one_expert(w1_unsigned))
        if need_ref:
            w1_ref[expert].copy_(torch.where(w1_unsigned > 7, w1_unsigned - 16, w1_unsigned))
        del w1_unsigned

        w2_unsigned = torch.randint(0, 15, (shape.k, shape.n), device=device, dtype=torch.int8)
        w2_run[expert].copy_(_pack_and_shuffle_one_expert(w2_unsigned))
        if need_ref:
            w2_ref[expert].copy_(torch.where(w2_unsigned > 7, w2_unsigned - 16, w2_unsigned))
        del w2_unsigned

    tensor = torch.rand((m, shape.topk), device=device)
    data = {
        "input": (torch.rand((m, shape.k), device=device).to(dtype=dtype)) / 10000,
        "w1_run": w1_run,
        "w2_run": w2_run,
        "w1_scales": torch.randn((shape.e, n1, 1), dtype=torch.float32, device=device),
        "w2_scales": torch.randn((shape.e, shape.k, 1), dtype=torch.float32, device=device),
        "topk_weights": torch.nn.functional.softmax(tensor, dim=1),
        "topk_ids": torch.randint(0, shape.e, (m, shape.topk), device=device, dtype=torch.int32),
    }
    if need_ref:
        if golden == "torch":
            data["ref"] = run_low_peak_torch_golden(
                data["input"],
                w1_ref,
                w2_ref,
                data["topk_weights"],
                data["topk_ids"],
                dtype,
                data["w1_scales"],
                data["w2_scales"],
            )
        data["golden"] = golden
        del w1_ref, w2_ref
    return data


def prepare_golden_cache(shape: Shape, m: int, cache_path: Path, route_chunk: int, golden: str) -> None:
    import torch

    if cache_path.exists():
        return

    data = prepare_low_peak_inputs(shape, m, need_ref=True, golden=golden)
    cached = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in data.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(cached, tmp_path)
        tmp_path.replace(cache_path)
    finally:
        unlink_cache(tmp_path)
    del data, cached


def ensure_golden_cache(shape: Shape, m: int, cache_path: Path, route_chunk: int, golden: str) -> None:
    prepare_golden_cache(shape, m, cache_path, route_chunk, golden)
    try:
        cached = load_golden_cache_cpu(cache_path)
        del cached
    except Exception as exc:
        print(f"WARN invalid golden cache {cache_path}: {exc}; rebuilding", flush=True)
        unlink_cache(cache_path)
        prepare_golden_cache(shape, m, cache_path, route_chunk, golden)
        cached = load_golden_cache_cpu(cache_path)
        del cached


def load_or_prepare_golden_cache(
    shape: Shape,
    m: int,
    cache_path: Path,
    route_chunk: int,
    golden: str,
) -> Dict[str, Any]:
    ensure_golden_cache(shape, m, cache_path, route_chunk, golden)
    return load_golden_cache(cache_path)


@contextlib.contextmanager
def quiet_stdio():
    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    old_disable_level = logging.root.manager.disable
    with open(os.devnull, "w") as devnull:
        try:
            logging.disable(logging.CRITICAL)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            logging.disable(old_disable_level)


def run_local_child(payload_path: Path) -> int:
    payload = read_json(payload_path)
    shape = Shape(**payload["shape"])
    m = int(payload["m"])
    gemm1 = {k: int(v) for k, v in payload["gemm1_config"].items()}
    gemm2 = {k: int(v) for k, v in payload["gemm2_config"].items()}
    build_dir = payload["build_dir"]

    os.environ["MOE_C_BUILD_DIR"] = build_dir
    os.environ["WHICH_TO_TEST"] = "1"
    sys.path.insert(0, str(MOE_C_ROOT / "test"))
    sys.path.insert(0, str(AITER_ROOT))

    result: Dict[str, Any] = {
        "ok": False,
        "m": m,
        "gemm1_config": gemm1,
        "gemm2_config": gemm2,
        "latency_us": None,
        "latency_event_us": None,
        "correctness": "not_run",
        "failure": None,
    }

    try:
        import torch
        from aiter.test_common import perftest
        from moe_utils.fused_moe_c import moe_c_fused_experts as fused_experts

        torch.set_default_device("cuda")
        data = load_golden_cache(Path(payload["cache_path"]))
        ref = data["ref"]

        def invoke():
            return fused_experts(
                data["input"],
                data["w1_run"],
                data["w2_run"],
                data["topk_weights"],
                data["topk_ids"],
                int(gemm1["MODE"]),
                int(gemm2["MODE"]),
                int(gemm1["BLOCK_SIZE_M"]),
                inplace=False,
                activation="silu",
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w4a8=True,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                use_int4_w4a16_base=False,
                global_num_experts=shape.e,
                expert_map=None,
                w1_scale=data["w1_scales"] / 16,
                w2_scale=data["w2_scales"] / 16,
                w1_zp=None,
                w2_zp=None,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
            )

        out = invoke()
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs()
        denom = torch.maximum(out.float().abs(), ref.float().abs()).clamp_min(1e-12)
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
        passed = bool(torch.allclose(out, ref, atol=float(payload["atol"]), rtol=float(payload["rtol"])))
        result["max_abs"] = max_abs
        result["max_rel"] = max_rel
        result["correctness"] = "passed" if passed else "failed"
        if not passed:
            result["failure"] = f"correctness failed max_abs={max_abs} max_rel={max_rel}"
            print("TUNE_RESULT_JSON=" + json.dumps(result, sort_keys=True))
            return 2

        for _ in range(int(payload["warmup"])):
            invoke()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(payload["iters"])):
            out = invoke()
        end.record()
        end.synchronize()
        result["latency_event_us"] = start.elapsed_time(end) * 1000.0 / max(1, int(payload["iters"]))

        @perftest(num_warmup=10, num_iters=100, num_rotate_args=1, testGraph=True)
        def invoke_profiler():
            return invoke()

        _, latency_us = invoke_profiler()
        result["latency_us"] = float(latency_us)
        result["ok"] = True
    except BaseException as exc:
        result["failure"] = repr(exc)
        print("TUNE_RESULT_JSON=" + json.dumps(result, sort_keys=True))
        return 1

    print("TUNE_RESULT_JSON=" + json.dumps(result, sort_keys=True))
    return 0


def run_local_batch_child(payload_path: Path) -> int:
    payload = read_json(payload_path)
    shape = Shape(**payload["shape"])
    m = int(payload["m"])
    build_dir = payload["build_dir"]

    os.environ["MOE_C_BUILD_DIR"] = build_dir
    os.environ["WHICH_TO_TEST"] = "1"
    sys.path.insert(0, str(MOE_C_ROOT / "test"))
    sys.path.insert(0, str(AITER_ROOT))

    try:
        import torch
        from aiter.test_common import perftest
        from moe_utils.fused_moe_c import moe_c_fused_experts as fused_experts

        torch.set_default_device("cuda")
        data = load_golden_cache(Path(payload["cache_path"]))
        ref = data["ref"]

        for trial in payload["trials"]:
            gemm1 = {k: int(v) for k, v in trial["gemm1_config"].items()}
            gemm2 = {k: int(v) for k, v in trial["gemm2_config"].items()}
            result: Dict[str, Any] = {
                "ok": False,
                "m": m,
                "label": trial["label"],
                "gemm1_config": gemm1,
                "gemm2_config": gemm2,
                "latency_us": None,
                "latency_event_us": None,
                "correctness": "not_run",
                "failure": None,
            }

            def invoke():
                return fused_experts(
                    data["input"],
                    data["w1_run"],
                    data["w2_run"],
                    data["topk_weights"],
                    data["topk_ids"],
                    int(gemm1["MODE"]),
                    int(gemm2["MODE"]),
                    int(gemm1["BLOCK_SIZE_M"]),
                    inplace=False,
                    activation="silu",
                    use_fp8_w8a8=False,
                    use_int8_w8a8=False,
                    use_int8_w4a8=True,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    use_int4_w4a16_base=False,
                    global_num_experts=shape.e,
                    expert_map=None,
                    w1_scale=data["w1_scales"] / 16,
                    w2_scale=data["w2_scales"] / 16,
                    w1_zp=None,
                    w2_zp=None,
                    a1_scale=None,
                    a2_scale=None,
                    block_shape=None,
                )

            try:
                out = invoke()
                torch.cuda.synchronize()
                diff = (out.float() - ref.float()).abs()
                denom = torch.maximum(out.float().abs(), ref.float().abs()).clamp_min(1e-12)
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
                max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
                passed = bool(torch.allclose(out, ref, atol=float(payload["atol"]), rtol=float(payload["rtol"])))
                result["max_abs"] = max_abs
                result["max_rel"] = max_rel
                result["correctness"] = "passed" if passed else "failed"
                if not passed:
                    result["failure"] = f"correctness failed max_abs={max_abs} max_rel={max_rel}"
                    print("TUNE_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
                    continue

                for _ in range(int(payload["warmup"])):
                    invoke()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(int(payload["iters"])):
                    out = invoke()
                end.record()
                end.synchronize()
                result["latency_event_us"] = start.elapsed_time(end) * 1000.0 / max(1, int(payload["iters"]))

                @perftest(num_warmup=10, num_iters=100, num_rotate_args=1, testGraph=True)
                def invoke_profiler():
                    return invoke()

                _, latency_us = invoke_profiler()
                result["latency_us"] = float(latency_us)
                result["ok"] = True
            except BaseException as exc:
                result["failure"] = repr(exc)
            print("TUNE_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    except BaseException as exc:
        print("TUNE_BATCH_FATAL_JSON=" + json.dumps({"failure": repr(exc)}, sort_keys=True), flush=True)
        return 1
    return 0


def parse_child_result(stdout: str, stderr: str, returncode: int, timeout: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "returncode": returncode,
        "timeout": timeout,
        "core_dump": returncode < 0,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "failure": None,
    }
    for line in reversed(stdout.splitlines()):
        if line.startswith("TUNE_RESULT_JSON="):
            try:
                result.update(json.loads(line.split("=", 1)[1]))
            except json.JSONDecodeError:
                result["failure"] = "failed to parse child JSON"
            break
    if timeout:
        result["failure"] = "timeout"
    elif returncode < 0:
        result["failure"] = f"terminated by signal {-returncode}"
    elif returncode != 0 and not result.get("failure"):
        result["failure"] = f"child exited with {returncode}"
    return result


def parse_child_results(stdout: str, stderr: str, returncode: int, timeout: bool) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        if line.startswith("TUNE_RESULT_JSON="):
            try:
                results.append(json.loads(line.split("=", 1)[1]))
            except json.JSONDecodeError:
                results.append({"ok": False, "failure": "failed to parse child JSON"})
    if timeout or returncode < 0 or returncode != 0:
        failure = "timeout" if timeout else (f"terminated by signal {-returncode}" if returncode < 0 else f"child exited with {returncode}")
        if results:
            for result in results:
                result.setdefault("failure", failure)
        else:
            results.append({
                "ok": False,
                "returncode": returncode,
                "timeout": timeout,
                "core_dump": returncode < 0,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
                "failure": failure,
            })
    for result in results:
        result.setdefault("returncode", returncode)
        result.setdefault("timeout", timeout)
        result.setdefault("core_dump", returncode < 0)
        result.setdefault("stdout_tail", stdout[-4000:])
        result.setdefault("stderr_tail", stderr[-4000:])
    return results


def run_one(shape: Shape, m: int, gemm1: Dict[str, int], gemm2: Dict[str, int], args, label: str) -> Dict[str, Any]:
    out_dir = Path(args.out) if args.out else default_out_dir(shape)
    payload = {
        "shape": asdict(shape),
        "m": m,
        "gemm1_config": gemm1,
        "gemm2_config": gemm2,
        "build_dir": args.build_dir,
        "cache_path": str(cache_file(out_dir, shape, m, args.golden)),
        "warmup": args.warmup,
        "iters": args.iters,
        "timeout": args.timeout,
        "atol": args.atol,
        "rtol": args.rtol,
        "route_chunk": args.route_chunk,
        "label": label,
    }
    with tempfile.TemporaryDirectory(prefix="moe_c_local_w4a8_") as tmp:
        payload_path = Path(tmp) / "payload.json"
        write_json(payload_path, payload)
        cmd = [sys.executable, str(Path(__file__).resolve()), "child-run", "--payload", str(payload_path)]
        env = os.environ.copy()
        env["MOE_C_BUILD_DIR"] = args.build_dir
        env["WHICH_TO_TEST"] = "1"
        env["PYTHONPATH"] = f"{MOE_C_ROOT / 'test'}:{AITER_ROOT}:{env.get('PYTHONPATH', '')}"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(MOE_C_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
            )
            result = parse_child_result(proc.stdout, proc.stderr, proc.returncode, False)
        except subprocess.TimeoutExpired as exc:
            result = parse_child_result(exc.stdout or "", exc.stderr or "", -999, True)
    result.update({"label": label, "m": m, "gemm1_config": gemm1, "gemm2_config": gemm2})
    return result


def run_many(shape: Shape, m: int, trials: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    out_dir = Path(args.out) if args.out else default_out_dir(shape)
    payload = {
        "shape": asdict(shape),
        "m": m,
        "trials": trials,
        "build_dir": args.build_dir,
        "cache_path": str(cache_file(out_dir, shape, m, args.golden)),
        "warmup": args.warmup,
        "iters": args.iters,
        "timeout": args.timeout,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    with tempfile.TemporaryDirectory(prefix="moe_c_local_w4a8_batch_") as tmp:
        payload_path = Path(tmp) / "payload.json"
        write_json(payload_path, payload)
        cmd = [sys.executable, str(Path(__file__).resolve()), "batch-child-run", "--payload", str(payload_path)]
        env = os.environ.copy()
        env["MOE_C_BUILD_DIR"] = args.build_dir
        env["WHICH_TO_TEST"] = "1"
        env["PYTHONPATH"] = f"{MOE_C_ROOT / 'test'}:{AITER_ROOT}:{env.get('PYTHONPATH', '')}"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(MOE_C_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
            )
            results = parse_child_results(proc.stdout, proc.stderr, proc.returncode, False)
        except subprocess.TimeoutExpired as exc:
            results = parse_child_results(exc.stdout or "", exc.stderr or "", -999, True)

    by_label = {r.get("label"): r for r in results}
    ordered = []
    for trial in trials:
        result = by_label.get(trial["label"], {"ok": False, "failure": "missing child result"})
        result.update({"label": trial["label"], "m": m, "gemm1_config": trial["gemm1_config"], "gemm2_config": trial["gemm2_config"]})
        ordered.append(result)
    return ordered


def best_success(results: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    passed = [r for r in results if r.get("ok") and r.get("latency_us") is not None]
    return min(passed, key=lambda r: float(r["latency_us"])) if passed else None


def make_invoke(fused_experts, data: Dict[str, Any], shape: Shape, gemm1: Dict[str, int], gemm2: Dict[str, int]):
    def invoke():
        return fused_experts(
            data["input"],
            data["w1_run"],
            data["w2_run"],
            data["topk_weights"],
            data["topk_ids"],
            int(gemm1["MODE"]),
            int(gemm2["MODE"]),
            int(gemm1["BLOCK_SIZE_M"]),
            inplace=False,
            activation="silu",
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w4a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            use_int4_w4a16_base=False,
            global_num_experts=shape.e,
            expert_map=None,
            w1_scale=data["w1_scales"] / 16,
            w2_scale=data["w2_scales"] / 16,
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
        )
    return invoke


def run_trial_inprocess(
    shape: Shape,
    m: int,
    data: Dict[str, Any],
    fused_experts,
    gemm1: Dict[str, int],
    gemm2: Dict[str, int],
    args,
    label: str,
) -> Dict[str, Any]:
    import torch
    from aiter.test_common import perftest

    result: Dict[str, Any] = {
        "ok": False,
        "m": m,
        "label": label,
        "gemm1_config": gemm1,
        "gemm2_config": gemm2,
        "latency_us": None,
        "latency_event_us": None,
        "correctness": "not_run",
        "failure": None,
    }

    invoke = make_invoke(fused_experts, data, shape, gemm1, gemm2)
    out = invoke()
    torch.cuda.synchronize()
    if args.perf_only:
        result["correctness"] = "skipped"
    else:
        ref = data["ref"]
        diff = (out.float() - ref.float()).abs()
        denom = torch.maximum(out.float().abs(), ref.float().abs()).clamp_min(1e-12)
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
        passed = bool(torch.allclose(out, ref, atol=float(args.atol), rtol=float(args.rtol)))
        result["max_abs"] = max_abs
        result["max_rel"] = max_rel
        result["correctness"] = "passed" if passed else "failed"
        if not passed:
            result["failure"] = f"correctness failed max_abs={max_abs} max_rel={max_rel}"
            return result

    for _ in range(int(args.warmup)):
        invoke()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(args.iters)):
        out = invoke()
    end.record()
    end.synchronize()
    result["latency_event_us"] = start.elapsed_time(end) * 1000.0 / max(1, int(args.iters))

    @perftest(num_warmup=10, num_iters=100, num_rotate_args=1, testGraph=False)
    def invoke_profiler():
        return invoke()

    with quiet_stdio():
        _, latency_us = invoke_profiler()
    result["latency_us"] = float(latency_us)
    result["ok"] = True
    return result


def print_trial(result: Dict[str, Any]) -> None:
    candidate = result.get("candidate", {})
    name = candidate.get("candidate_id") or result.get("label")
    mode = candidate.get("mode")
    if mode is None:
        mode = result.get("gemm1_config", {}).get("MODE")
    latency = result.get("latency_us")
    latency_text = "None" if latency is None else f"{float(latency):.3f}"
    status = "pass" if result.get("ok") else "fail"
    color = "\033[32m" if result.get("ok") else "\033[31m"
    reset = "\033[0m"
    print(
        f"{name:<36} {str(mode):>8} {latency_text:>12}  {color}{status}{reset}",
        flush=True,
    )


def print_trial_start(candidate_id: str, mode: Any) -> None:
    print(f"{candidate_id:<36} {str(mode):>8} ", end="", flush=True)


def print_trial_finish(result: Dict[str, Any]) -> None:
    latency = result.get("latency_us")
    latency_text = "None" if latency is None else f"{float(latency):.3f}"
    status = "pass" if result.get("ok") else "fail"
    color = "\033[32m" if result.get("ok") else "\033[31m"
    reset = "\033[0m"
    print(f"{latency_text:>12}  {color}{status}{reset}", flush=True)


def stage_fixed_config(candidates: List[Candidate], bm: int) -> Optional[Dict[str, int]]:
    same_bm = [c for c in candidates if c.block_size_m == bm]
    for family, nloop in ((3, 4), (2, 4), (1, None), (3, None), (2, None)):
        for candidate in same_bm:
            if (candidate.mode // 100_000) % 10 == family and (nloop is None or candidate.n_loop == nloop):
                return candidate.config()
    return same_bm[0].config() if same_bm else None


def run_tune(args) -> None:
    """Tune independent GEMMs through the shared descriptor-based runner."""
    if args.golden != "torch":
        raise ValueError("independent GEMM tuning requires the torch matmul golden")
    if args.perf_only:
        raise ValueError("independent GEMM tuning requires correctness before timing")
    os.environ["MOE_C_BUILD_DIR"] = args.build_dir
    shape = Shape(args.e, args.n, args.k, args.topk, args.dtype, args.m, resolve_cu(args.arch, args.cu))
    out_dir = Path(args.out) if args.out else default_out_dir(shape)
    cmd = [sys.executable, str(Path(__file__).with_name("tune_moe_hipc.py")),
           "--quant", "int8_w4a8", "--arch", args.arch, "--dtype", args.dtype,
           "--e", str(args.e), "--n", str(args.n), "--k", str(args.k),
           "--topk", str(args.topk), "--m", *map(str, args.m),
           "--warmup", str(args.warmup), "--iters", str(args.iters),
           "--max-nloop", str(args.max_nloop), "--candidate-timeout", str(args.timeout),
           "--out-dir", str(out_dir), "--bench-direct", "--sweep-direct", "--direct-inprocess"]
    if shape.cu is not None:
        cmd.extend(["--cu", str(shape.cu)])
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[1]))


def bench_config(args) -> None:
    os.environ["MOE_C_BUILD_DIR"] = args.build_dir
    os.environ["WHICH_TO_TEST"] = "1"
    shape = Shape(args.e, args.n, args.k, args.topk, args.dtype, [args.m], resolve_cu(args.arch, args.cu))
    out_dir = Path(args.out) if args.out else default_out_dir(shape)
    m_cache = cache_file(out_dir, shape, args.m, args.golden)
    print(f"TUNE_CACHE m={args.m} path={m_cache}", flush=True)
    ensure_golden_cache(shape, args.m, m_cache, args.route_chunk, args.golden)
    gemm1 = {"BLOCK_SIZE_M": int(args.block_size_m), "MODE": int(args.gemm1_mode)}
    gemm2 = {"BLOCK_SIZE_M": int(args.block_size_m), "MODE": int(args.gemm2_mode)}
    try:
        args.out = str(out_dir)
        result = run_one(shape, args.m, gemm1, gemm2, args, f"m{args.m}_bench_config")
        print("BENCH_CONFIG_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        unlink_cache(m_cache)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run local moe_c W4A8 tune")
    run.add_argument("--e", type=int, default=DEFAULT_E)
    run.add_argument("--n", type=int, default=DEFAULT_N)
    run.add_argument("--k", type=int, default=DEFAULT_K)
    run.add_argument("--arch", default="gfx936")
    run.add_argument("--cu", "--num-cu", dest="cu", type=int, default=None)
    run.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    run.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("bf16", "fp16"))
    run.add_argument("--m", nargs="*", type=int, default=list(DEFAULT_M_VALUES))
    run.add_argument("--max-nloop", type=int, default=DEFAULT_MAX_NLOOP)
    run.add_argument("--build-dir", default=DEFAULT_BUILD_DIR)
    run.add_argument("--out", default=DEFAULT_OUT)
    run.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    run.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    run.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    run.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    run.add_argument("--route-chunk", type=int, default=DEFAULT_ROUTE_CHUNK)
    run.add_argument("--golden", default=DEFAULT_GOLDEN, choices=("torch", "triton"))
    run.add_argument("--perf-only", action="store_true", help="skip golden/cache and run performance only")
    run.set_defaults(func=run_tune)

    bench = sub.add_parser("bench-config", help="benchmark one explicit W4A8 moe_c config")
    bench.add_argument("--e", type=int, default=DEFAULT_E)
    bench.add_argument("--n", type=int, default=DEFAULT_N)
    bench.add_argument("--k", type=int, default=DEFAULT_K)
    bench.add_argument("--arch", default="gfx936")
    bench.add_argument("--cu", "--num-cu", dest="cu", type=int, default=None)
    bench.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    bench.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("bf16", "fp16"))
    bench.add_argument("--m", type=int, default=1)
    bench.add_argument("--gemm1-mode", type=int, required=True)
    bench.add_argument("--gemm2-mode", type=int, required=True)
    bench.add_argument("--block-size-m", type=int, required=True)
    bench.add_argument("--build-dir", default=DEFAULT_BUILD_DIR)
    bench.add_argument("--out", default=DEFAULT_OUT)
    bench.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    bench.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    bench.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    bench.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    bench.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    bench.add_argument("--route-chunk", type=int, default=DEFAULT_ROUTE_CHUNK)
    bench.add_argument("--golden", default=DEFAULT_GOLDEN, choices=("torch", "triton"))
    bench.set_defaults(func=bench_config)

    child = sub.add_parser("child-run", help=argparse.SUPPRESS)
    child.add_argument("--payload", required=True)
    child.set_defaults(func=lambda args: sys.exit(run_local_child(Path(args.payload))))

    batch_child = sub.add_parser("batch-child-run", help=argparse.SUPPRESS)
    batch_child.add_argument("--payload", required=True)
    batch_child.set_defaults(func=lambda args: sys.exit(run_local_batch_child(Path(args.payload))))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
