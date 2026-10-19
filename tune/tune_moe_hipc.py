#!/usr/bin/env python3
"""Unified HIPC moe_c config auditor and tune entry point.

The HIPC mode namespace is shared by the channelwise quantizers and has the
fixed-width form ``S F K MM VVV``::

    S   stage (1=GEMM1, 2=GEMM2)
    F   metadata layout (0=CK sorting, 1=moe_align)
    K   kernel family
    MM  BLOCK_SIZE_M / 16
    VVV profile (the profile owns N_LOOP and other compile-time constants)

This file deliberately keeps quantizer-specific legality in descriptors.  The
common flow only decodes modes, validates config shape, writes reports, and
delegates correctness/performance execution to the existing per-quant tests.
It therefore cannot accidentally make an INT8 or W4A8 mode look like FP8.
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "moe_c_configs"
sys.path.insert(0, str(ROOT))
from moe_utils.fused_moe_c import CSV_NAMES, find_csv_config, rows_from_pair, write_config_csv


@dataclass(frozen=True)
class Shape:
    e: int
    n: int
    k: int
    topk: int
    m: int
    dtype: str
    arch: str
    cu: Optional[int] = None


@dataclass(frozen=True)
class Candidate:
    stage: int
    mode: int
    bm: int
    sorting: int
    family: int
    profile: int
    nloop: Optional[int]
    tag: str


@dataclass(frozen=True)
class ModeDescriptor:
    quant: str
    mode: int
    stage: int
    sorting: int
    family: int
    kernel: str
    bm: int
    bn: int
    bk: int
    wm: int
    wn: int
    wk: int
    stages: int
    nloop: int
    k_step: int
    k_tail: Optional[int] = None
    kloop: Optional[int] = None  # No independent K_LOOP template in these launchers.
    fixed_reduce_k: Optional[int] = None
    out_n_variants: Tuple[Tuple[int, int, str], ...] = ()
    supported_reduce_k: Tuple[int, ...] = ()
    exact_out_n: Optional[int] = None
    min_reduce_k: int = 1
    supports_n_tail: bool = False
    architectures: Tuple[str, ...] = ("gfx936", "gfx938", "gfx92a", "gfx928", "native")
    input_dtypes: Tuple[str, ...] = ("bf16", "fp16")
    recommended_m_min: int = 1
    recommended_m_max: Optional[int] = None
    source: str = ""

    @property
    def tile_n(self) -> int:
        return self.bn * self.nloop

    @property
    def sorting_name(self) -> str:
        return "ck_sorting" if self.sorting == 0 else "moe_align"

    @property
    def token_encoding(self) -> str:
        return "(topk_slot << 24) | token" if self.sorting == 0 else "token * topk + topk_slot"

    def for_shape(self, shape: Shape) -> "ModeDescriptor":
        out_n = shape.k if self.stage == 2 else 2 * shape.n
        for n, nloop, kernel in self.out_n_variants:
            if out_n == n:
                return replace(self, nloop=nloop, kernel=kernel, out_n_variants=())
        return self

    def recommended(self, m: int) -> bool:
        return m >= self.recommended_m_min and (
            self.recommended_m_max is None or m <= self.recommended_m_max)

    def validate(self, shape: Shape) -> Optional[str]:
        if self.sorting not in (0, 1):
            return f"unsupported metadata layout F={self.sorting}"
        arch = shape.arch.split(":", 1)[0]
        if arch not in self.architectures:
            return f"arch={shape.arch} is not supported by {self.kernel}"
        # Older direct callers store the quant category in Shape.dtype.
        if shape.dtype != self.quant and shape.dtype not in self.input_dtypes:
            return f"dtype={shape.dtype} unsupported; expected {self.input_dtypes}"
        if shape.e and shape.topk and not 0 < shape.topk <= shape.e:
            return f"topk={shape.topk} must be in [1, E={shape.e}]"
        if self.sorting == 0 and (shape.m >= 1 << 24 or shape.topk >= 1 << 8):
            return "shape exceeds CK sorting's 24-bit token / 8-bit topk encoding"
        if self.sorting == 1 and shape.m * shape.topk + shape.e * (self.bm - 1) >= 1 << 31:
            return "moe_align padded route indices exceed signed int32"
        out_n = shape.k if self.stage == 2 else 2 * shape.n
        reduce_k = shape.n if self.stage == 2 else shape.k
        # Zero dimensions are unknown in config filenames, not a tuning shape.
        if out_n:
            if self.exact_out_n is not None and out_n != self.exact_out_n:
                return f"out_n={out_n}, expected {self.exact_out_n}"
            tile_n = self.for_shape(shape).tile_n
            if not self.supports_n_tail and out_n % tile_n:
                return f"out_n={out_n} is not divisible by tile_n={tile_n}"
        if reduce_k:
            if self.supported_reduce_k and reduce_k not in self.supported_reduce_k:
                return f"reduce_k={reduce_k} is not in {self.supported_reduce_k}"
            if reduce_k < self.min_reduce_k:
                return f"reduce_k={reduce_k} is below minimum {self.min_reduce_k}"
            remainder = reduce_k % self.k_step
            if remainder and remainder != self.k_tail:
                return f"reduce_k={reduce_k} needs step={self.k_step}, tail={self.k_tail}"
        return None


@dataclass(frozen=True)
class BenchResult:
    file: str
    topk: int
    cu: Optional[int]
    token: int
    stage: int
    mode: int
    bm: int
    sorting: str
    family: int
    profile: int
    nloop: Optional[int]
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class DirectBenchRow:
    m: int
    stage: str
    mode: int
    bm: int
    sorting: str
    latency_us: Optional[float]
    ok: bool
    error: str = ""
    max_delta: Optional[float] = None


def decode_mode(mode: int) -> Candidate:
    mode = int(mode)
    if not 10_000_000 <= mode < 30_000_000:
        raise ValueError("legacy or malformed mode; expected S F K MM VVV")
    return Candidate(
        stage=mode // 10_000_000,
        mode=mode,
        bm=((mode // 1_000) % 100) * 16,
        sorting=(mode // 1_000_000) % 10,
        family=(mode // 100_000) % 10,
        profile=mode % 1_000,
        nloop=None,
        tag="descriptive",
    )


def mode(stage: int, sorting: int, family: int, bm: int, profile: int) -> int:
    if stage not in (1, 2) or sorting not in (0, 1):
        raise ValueError("invalid mode prefix")
    if bm <= 0 or bm % 16 or bm // 16 >= 100:
        raise ValueError(f"BM is not representable: {bm}")
    if not 0 < profile < 1000:
        raise ValueError(f"profile is not representable: {profile}")
    return stage * 10_000_000 + sorting * 1_000_000 + family * 100_000 + (bm // 16) * 1_000 + profile


def default_cu_for_arch(arch: Optional[str]) -> Optional[int]:
    if arch == "gfx938":
        return 72
    if arch == "gfx936":
        return 80
    return None


def resolve_cu(arch: Optional[str], cu: Optional[int]) -> Optional[int]:
    return cu if cu is not None else default_cu_for_arch(arch)


class QuantProfile:
    name: str
    categories: Tuple[str, ...]
    allowed_sorting: Tuple[int, ...]
    allowed_families: Tuple[int, ...]
    ordinary_profiles: Tuple[int, ...]
    modes_by_stage: Dict[int, Tuple[int, ...]]

    def __init__(self, name: str, categories: Tuple[str, ...], sorting: Tuple[int, ...], families: Tuple[int, ...], ordinary_profiles: Tuple[int, ...], modes_by_stage: Dict[int, Tuple[int, ...]]):
        self.name = name
        self.categories = categories
        self.allowed_sorting = sorting
        self.allowed_families = families
        self.ordinary_profiles = ordinary_profiles
        self.modes_by_stage = {stage: tuple(sorted(set(values))) for stage, values in modes_by_stage.items()}
        self.descriptors = {
            value: _describe_mode(name, decode_mode(value))
            for values in self.modes_by_stage.values() for value in values
        }

    def descriptor(self, mode_i: int, shape: Optional[Shape] = None) -> ModeDescriptor:
        try:
            desc = self.descriptors[int(mode_i)]
            return desc.for_shape(shape) if shape is not None else desc
        except KeyError:
            raise ValueError(f"mode={mode_i} is not declared by {self.name} profile") from None

    def candidate(self, mode_i: int, shape: Optional[Shape] = None) -> Candidate:
        desc = self.descriptor(mode_i, shape)
        return Candidate(**{**asdict(decode_mode(mode_i)), "nloop": desc.nloop, "tag": desc.kernel})

    def candidate_modes(self, stage: int) -> Tuple[int, ...]:
        return self.modes_by_stage.get(stage, ())

    def owns_mode(self, candidate: Candidate) -> bool:
        return candidate.mode in self.candidate_modes(candidate.stage)

    def validate(self, candidate: Candidate, shape: Shape, is_bottom: bool) -> Optional[str]:
        expected_stage = 2 if is_bottom else 1
        if candidate.stage != expected_stage:
            return f"stage={candidate.stage}, expected={expected_stage}"
        if not self.owns_mode(candidate):
            return f"mode={candidate.mode} is not declared by {self.name} profile"
        if candidate.sorting not in self.allowed_sorting:
            return f"sorting={candidate.sorting} unsupported by {self.name}"
        if candidate.family not in self.allowed_families:
            return f"family={candidate.family} unsupported by {self.name}"
        if candidate.bm <= 0:
            return "BM is zero"
        return self.descriptor(candidate.mode).validate(shape)


def _modes(items: Iterable[Tuple[int, int, int, int, int]]) -> Tuple[int, ...]:
    return tuple(mode(stage, sorting, family, bm, profile)
                 for stage, sorting, family, bm, profile in items)


def _decode_modes(stage: int, sorting: int, profiles_by_bm: Dict[int, Iterable[int]]) -> List[Tuple[int, int, int, int, int]]:
    return [(stage, sorting, 0, bm, profile)
            for bm, profiles in profiles_by_bm.items()
            for profile in profiles]


def _nloop_modes(stage: int, sorting: int, family: int,
                 bms: Iterable[int], nloops: Iterable[int]) -> List[Tuple[int, int, int, int, int]]:
    return [(stage, sorting, family, bm, 100 + nloop)
            for bm in bms
            for nloop in nloops]


def _ordinary_modes(stage: int, sorting: int, family: int,
                    bm_profile: Iterable[Tuple[int, int]]) -> List[Tuple[int, int, int, int, int]]:
    return [(stage, sorting, family, bm, profile) for bm, profile in bm_profile]


def _w8a8_int8_modes() -> Dict[int, Tuple[int, ...]]:
    gemm1 = []
    gemm1 += _decode_modes(1, 0, {16: (101, 102, 103)})
    gemm1 += _ordinary_modes(1, 0, 1, ((32, 201), (32, 202), (32, 203), (32, 204)))
    gemm1 += _nloop_modes(1, 0, 2, (32, 48, 64), (1, 2))

    gemm2 = []
    gemm2 += _decode_modes(2, 0, {16: (101, 102, 103, 104)})
    gemm2 += _ordinary_modes(2, 0, 1, ((32, 201), (64, 202)))
    gemm2 += _nloop_modes(2, 0, 2, (32, 48, 64), (4,))
    gemm2 += _nloop_modes(2, 0, 2, (32, 48, 64), (1, 2))
    return {1: _modes(gemm1), 2: _modes(gemm2)}


def _w8a8_fp8_modes() -> Dict[int, Tuple[int, ...]]:
    gemm1 = []
    gemm1 += _decode_modes(1, 0, {16: (101, 102, 103), 32: (101,)})
    gemm1 += _nloop_modes(1, 1, 1, (32, 48, 64, 80, 96, 128), (1, 2, 5))
    gemm1 += _nloop_modes(1, 1, 1, (80, 96), (4,))
    gemm1 += _nloop_modes(1, 0, 2, (32, 48, 64), (1, 2, 5))

    gemm2 = []
    gemm2 += _decode_modes(2, 0, {16: (101, 102, 103, 104), 32: (101,)})
    gemm2 += _nloop_modes(2, 1, 1, (32, 48, 64, 80, 96, 128), (1, 2))
    gemm2 += _nloop_modes(2, 1, 1, (32, 48, 64, 80, 96, 128), (4,))
    gemm2 += _nloop_modes(2, 0, 2, (32, 48, 64), (1, 2, 4))
    return {1: _modes(gemm1), 2: _modes(gemm2)}


def _w4a8_modes() -> Dict[int, Tuple[int, ...]]:
    bms = (16, 32, 48, 64)
    gemm1 = []
    gemm1 += _decode_modes(1, 1, {16: (104,)})
    gemm1 += _ordinary_modes(1, 1, 1, ((16, 201), (32, 202), (48, 203), (64, 204)))
    gemm1 += _nloop_modes(1, 1, 2, bms, (1, 2, 3, 4, 5))

    gemm2 = []
    gemm2 += _decode_modes(2, 1, {16: (101,), 32: (102,), 48: (103,), 64: (104,)})
    gemm2 += _ordinary_modes(2, 1, 1, ((16, 201), (32, 202), (48, 203), (64, 204)))
    gemm2 += _nloop_modes(2, 1, 2, bms, (1, 2, 3, 4))
    gemm2 += _nloop_modes(2, 1, 3, bms, (1, 2, 3, 4))
    return {1: _modes(gemm1), 2: _modes(gemm2)}


def _describe_mode(quant: str, cand: Candidate) -> ModeDescriptor:
    up_k = (2048, 3072, 4096, 6144, 7168)
    fp8_up_k = (1280, *up_k)
    w8_down_k = (128, 256, 320, 384, 512, 640, 768, 1024, 1280, 2048)
    fp8_down_k = (128, 192, *w8_down_k[1:])
    w4_down_k = (128, 192, 256, 384, 512, 640, 768, 1024, 1536, 2048, 3072)
    params: Dict[str, Any] = {}
    up = cand.stage == 1
    is_fp8 = quant == "fp8_w8a8"
    source_file = "moe_w8a8_opt.h"
    if quant == "int8_w4a8":
        source_file = "moe_w4a8_opt.h"
        if cand.family == 0 and up:
            tiles = (32, 512, 16, 32, 128, 2)
            kernel = "launch_moe_w4a8_first_stage_decode"
            params.update(nloop=1, k_step=512, min_reduce_k=512)
        elif cand.family == 0:
            tiles = (128, 64, cand.bm, 32, 64, 2)
            kernel = "launch_moe_w4a8_second_stage_decode"
            params.update(nloop=4, k_step=128, k_tail=64,
                          supported_reduce_k=(128, 192, 256, 384, 512, 768, 1024, 1536, 2048))
        elif up:
            tiles = (128, 64, cand.bm, 32, 64, 2)
            kernel = ("launch_moe_w4a8_first_stage_prefill" if cand.family == 1
                      else "launch_moe_w4a8_first_stage_prefill_GEMM1Nloop")
            params.update(nloop=4 if cand.family == 1 else cand.profile - 100,
                          k_step=128, k_tail=64, min_reduce_k=128)
            if cand.family == 1:
                params["out_n_variants"] = ((256, 2, "launch_moe_w4a8_first_stage_prefill_GEMM1N256"),)
        else:
            tiles = (128, 64, cand.bm, 32, 64, 2)
            kernel = ("launch_moe_w4a8_second_stage_prefill" if cand.family == 1
                      else "launch_moe_w4a8_second_stage_prefill_fixed_k")
            params.update(nloop=4 if cand.family == 1 else cand.profile - 100,
                          k_step=128, k_tail=64,
                          supported_reduce_k=(192,) if cand.family == 3 else w4_down_k)
            if cand.family == 3:
                params["fixed_reduce_k"] = 192
    elif cand.family == 0:
        if up:
            variants = ({(16, 101): (64, 1024, 16, 64, 128, 1),
                         (16, 102): (16, 512, 16, 16, 128, 4),
                         (16, 103): (64, 256, 16, 64, 128, 2),
                         (32, 101): (128, 64, 32, 32, 64, 2)} if is_fp8 else
                        {(16, 101): (64, 1024, 16, 64, 128, 1),
                         (16, 102): (16, 512, 16, 16, 128, 4),
                         (16, 103): (16, 1024, 16, 16, 256, 4)})
            tiles = variants[(cand.bm, cand.profile)]
            params.update(nloop=1, k_step=tiles[1],
                          min_reduce_k=tiles[1],
                          supported_reduce_k=fp8_up_k if is_fp8 else up_k)
        else:
            variants = {(16, 101): (128, 64, 16, 32, 64, 2),
                        (16, 102): (256, 64, 16, 64, 64, 2),
                        (16, 103): (512, 64, 16, 64, 64, 2),
                        (16, 104): (512, 64, 16, 128, 64, 2),
                        (32, 101): (128, 64, 32, 32, 64, 2)}
            tiles = variants[(cand.bm, cand.profile)]
            params.update(nloop=1, k_step=128, k_tail=64 if is_fp8 else None,
                          min_reduce_k=128)
        kernel = "launch_moe_w8a8_" + ("first" if up else "second") + "_stage_decode" + ("_fp8" if is_fp8 else "")
    elif quant == "int8_w8a8" and cand.family == 1:
        if up:
            tiles = {201: (128, 64, 32, 128, 64, 1),
                     202: (128, 64, 32, 128, 64, 2),
                     203: (256, 64, 32, 128, 64, 2),
                     204: (64, 128, 32, 64, 64, 1)}[cand.profile]
        else:
            tiles = (128, 64, cand.bm, 32, 64, 2)
        kernel = "launch_moe_w8a8_" + ("first" if up else "second") + "_stage_prefill"
        params.update(nloop=1 if up else 4, k_step=tiles[1] if up else 128,
                      supported_reduce_k=up_k if up else tuple(k for k in w8_down_k if k != 1280))
        if not up:
            params["k_tail"] = 64
    else:
        tiles = (128, 64, cand.bm, 32, 64, 2)
        params.update(nloop=cand.profile - 100, k_step=128, k_tail=64, min_reduce_k=128)
        if is_fp8 and cand.sorting == 1:
            kernel = "launch_moe_w8a8_" + ("first" if up else "second") + "_stage_prefill_fp8_nloop"
            params["supported_reduce_k"] = up_k if up else fp8_down_k
        else:
            precision = "fp8" if is_fp8 else "int8"
            kernel = f"launch_moe_w8a8_prefill_{'up' if up else 'down'}_{precision}_bk64_nloop"
            source_file = f"moe_w8a8_{'bk128' if is_fp8 else 'int8_bk128'}_prefill.h"
            params["supported_reduce_k"] = (fp8_up_k if is_fp8 else up_k) if up else (
                (*fp8_down_k, 1536) if is_fp8 else w8_down_k)
    bn, bk, wm, wn, wk, stages = tiles
    return ModeDescriptor(
        quant=quant, mode=cand.mode, stage=cand.stage, sorting=cand.sorting,
        family=cand.family, kernel=kernel, bm=cand.bm,
        bn=bn, bk=bk, wm=wm, wn=wn, wk=wk, stages=stages,
        recommended_m_min=1 if cand.family == 0 else 513,
        recommended_m_max=512 if cand.family == 0 else None,
        source=f"csrc_for_aiter/936_938/{source_file}", **params,
    )


PROFILES: Dict[str, QuantProfile] = {
    "fp8_w8a8": QuantProfile("fp8_w8a8", ("fp8_w8a8",), (0, 1), (0, 1, 2), (201, 202, 203, 204), _w8a8_fp8_modes()),
    "int8_w8a8": QuantProfile("int8_w8a8", ("int8_w8a8",), (0,), (0, 1, 2), (201, 202, 203, 204), _w8a8_int8_modes()),
    "int8_w4a8": QuantProfile("int8_w4a8", ("int8_w4a8",), (1,), (0, 1, 2, 3), (201, 202, 203, 204), _w4a8_modes()),
}


def _is_bottom(path: Path) -> bool:
    return ",is_bottom=True" in path.name


def _arch_files(arch: str, profile: QuantProfile) -> Iterable[Path]:
    for category in profile.categories:
        yield from sorted((CONFIG_ROOT / arch / category).glob("*.json"))


def _shape_from_config_name(path: Path, arch: str, dtype: str) -> Shape:
    fields = {}
    for item in path.name[:-5].split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return Shape(
        e=int(fields.get("E", 0)),
        n=int(fields.get("N", 0)),
        k=int(fields.get("K", 0)),
        topk=int(fields.get("topk", 0)),
        m=0,
        dtype=dtype,
        arch=arch,
        cu=int(fields["cu"]) if fields.get("cu") else None,
    )


def audit_configs(profile: QuantProfile, arch: str, out_dir: Path,
                  shape_filter: Optional[Tuple[int, int, int]] = None) -> List[BenchResult]:
    results: List[BenchResult] = []
    records = []
    csv_path = CONFIG_ROOT / CSV_NAMES[profile.name]
    if csv_path.is_file():
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["arch"] != arch or row["quant_type"] != profile.name + "_channel":
                    continue
                for stage, mode_i in enumerate(row["sol_id"].split("+"), 1):
                    file_shape = Shape(int(row["expert"]), int(row["inter_dim"]), int(row["model_dim"]),
                                       int(row["topk"] or 0), int(row["token"]),
                                       "fp16" if row["indtype"] == "torch.float16" else "bf16", arch,
                                       int(row["cu_num"]) if row["cu_num"] else None)
                    records.append((csv_path, {row["token"]: {"MODE": int(mode_i)}}, stage == 2, file_shape))
    else:
        for path in _arch_files(arch, profile):
            try:
                data = json.loads(path.read_text())
            except Exception as exc:
                results.append(BenchResult(str(path), 0, None, 0, 0, 0, 0, "", 0, 0, None, False, f"invalid JSON: {exc}"))
                continue
            records.append((path, data, _is_bottom(path), _shape_from_config_name(path, arch, profile.name)))
    for path, data, bottom, file_shape in records:
        if shape_filter is not None:
            filter_e, filter_n, filter_k = shape_filter
            if filter_e and file_shape.e != filter_e:
                continue
            if filter_n and file_shape.n != filter_n:
                continue
            if filter_k and file_shape.k and file_shape.k != filter_k:
                continue
        for token, value in data.items():
            try:
                token_i = int(token)
                mode_i = int(value["MODE"])
                cand = profile.candidate(mode_i, file_shape)
                error = profile.validate(cand, file_shape, bottom)
            except Exception as exc:
                cand = None
                error = str(exc)
                token_i = int(token) if str(token).isdigit() else 0
            results.append(BenchResult(
                str(path.relative_to(ROOT)), file_shape.topk, file_shape.cu,
                token_i, cand.stage if cand else 0, mode_i if cand else 0,
                cand.bm if cand else 0,
                "moe_align" if cand and cand.sorting else "ck_sorting" if cand else "",
                cand.family if cand else 0, cand.profile if cand else 0,
                cand.nloop if cand else None, error is None, error or ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{profile.name}_{arch}_config_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else list(BenchResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    summary = {
        "arch": arch,
        "quant": profile.name,
        "files": len({r.file for r in results}),
        "entries": len(results),
        "passed": sum(r.ok for r in results),
        "failed": sum(not r.ok for r in results),
        "failed_examples": [asdict(r) for r in results if not r.ok][:20],
    }
    (out_dir / f"{profile.name}_{arch}_config_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results


def test_command(profile: QuantProfile, args: argparse.Namespace) -> List[str]:
    tokens = [str(x) for x in args.m]
    if profile.name == "int8_w4a8":
        script = ROOT / "op_test/test_aiter_moe_with_config_w4a8_perchannel.py"
        return [
            sys.executable,
            str(script),
            "--e", str(args.e),
            "--n", str(args.n),
            "--k", str(args.k),
            "--topk", str(args.topk),
            "--tokens", *tokens,
        ]
    script = ROOT / "op_test/test_aiter_moe_with_config_w8a8_channelwise.py"
    quant = "fp8" if profile.name == "fp8_w8a8" else "int8"
    return [
        sys.executable,
        str(script),
        "--quant", quant,
        "--e", str(args.e),
        "--n", str(args.n),
        "--k", str(args.k),
        "--topk", str(args.topk),
        "--tokens", *tokens,
    ]


def _event_time_us(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / max(1, iters))


def _select_config(configs: Dict[int, Any], m: int) -> Dict[str, Any]:
    key = min(configs.keys(), key=lambda x: abs(int(x) - m))
    config = dict(configs[key])
    config["key_selected"] = int(key)
    return config


def _config_file_names(e: int, n: int, k: int, dtype: str, is_bottom: bool,
                       topk: Optional[int], cu: Optional[int]) -> List[str]:
    suffix = ",is_bottom=True" if is_bottom else ""
    names: List[str] = []
    if k:
        if topk is not None and cu is not None:
            names.append(f"E={e},N={n},K={k},topk={topk},cu={cu},dtype={dtype}{suffix}.json")
        names.append(f"E={e},N={n},K={k},dtype={dtype}{suffix}.json")
    if topk is not None and cu is not None:
        names.append(f"E={e},N={n},topk={topk},cu={cu},dtype={dtype}{suffix}.json")
    names.append(f"E={e},N={n},dtype={dtype}{suffix}.json")
    return list(dict.fromkeys(names))


def _load_shape_config(profile: QuantProfile, arch: str, e: int, n: int, k: int,
                       is_bottom: bool, topk: Optional[int] = None,
                       cu: Optional[int] = None) -> Dict[int, Any]:
    if (CONFIG_ROOT / CSV_NAMES[profile.name]).is_file():
        result = find_csv_config(str(CONFIG_ROOT), arch, profile.name, e, n, k,
                                 topk, cu, is_bottom)
        if result is None:
            raise FileNotFoundError(f"No CSV config for {profile.name}/{arch}, N={n}, K={k}")
        return result
    candidates = []
    for category in profile.categories:
        base = CONFIG_ROOT / arch / category
        candidates.extend(base / name for name in _config_file_names(e, n, k, profile.name, is_bottom, topk, cu))
    for path in candidates:
        if path.exists():
            with path.open() as f:
                return {int(key): value for key, value in json.load(f).items()}
    raise FileNotFoundError("no config found; tried " + ", ".join(str(p) for p in candidates))


def _config_declared_modes(profile: QuantProfile, arch: str, stage: int) -> List[int]:
    modes = set()
    for category in profile.categories:
        base = CONFIG_ROOT / arch / category
        if not base.exists():
            continue
        want_bottom = stage == 2
        for path in base.glob("*.json"):
            if _is_bottom(path) != want_bottom:
                continue
            try:
                with path.open() as f:
                    data = json.load(f)
                for item in data.values():
                    modes.add(int(item["MODE"]))
            except Exception:
                continue
    return sorted(mode_i for mode_i in modes if mode_i in profile.candidate_modes(stage))


def _candidate_modes(profile: QuantProfile, stage: int, arch: Optional[str] = None,
                     config_declared_only: bool = False) -> List[int]:
    if config_declared_only and arch:
        modes = _config_declared_modes(profile, arch, stage)
        if modes:
            return modes
    return list(profile.candidate_modes(stage))


def _make_w8a8_inputs(m: int, shape: Shape, quant: str) -> Dict[str, Any]:
    import torch

    from moe_utils.fused_moe_c import fused_topk
    from moe_utils.shuffle import w8a8_moe_repack_shuffle
    from moe_utils.utils import dtypes, pertoken_quant

    torch.manual_seed(0)
    if quant == "int8_w4a8":
        from tune.tune_moe_w4a8_perchannel import _pack_and_shuffle_one_expert

        dtype = torch.float16 if shape.dtype == "fp16" else torch.bfloat16
        data = {"input": torch.randn((m, shape.k), device="cuda", dtype=dtype) / 100,
                "down_input": torch.randn((m * shape.topk, shape.n), device="cuda", dtype=dtype) / 100}
        for stage, width, depth in ((1, 2 * shape.n, shape.k), (2, shape.k, shape.n)):
            raw = torch.randint(-8, 8, (shape.e, width, depth), device="cuda", dtype=torch.int8)
            packed = torch.empty((shape.e, width, depth // 2), device="cuda", dtype=torch.int8)
            for expert in range(shape.e):
                packed[expert].copy_(_pack_and_shuffle_one_expert(raw[expert] & 15))
            scale = torch.rand((shape.e, width, 1), device="cuda", dtype=torch.float32) / 8
            data[f"w{stage}"] = packed
            data[f"w{stage}_scale"] = scale / 16
            data[f"reference_w{stage}"] = raw
            data[f"reference_scale{stage}"] = scale
        data["a1_scale"] = None
        data["topk_ids"] = torch.rand((m, shape.e), device="cuda").topk(shape.topk).indices.int()
        data["topk_weights"] = torch.softmax(torch.randn((m, shape.topk), device="cuda"), dim=1)
        return data
    is_fp8 = quant == "fp8_w8a8"
    input_dtype = torch.float32 if is_fp8 else (torch.float16 if shape.dtype == "fp16" else dtypes.bf16)
    input_fp = torch.randn((m, shape.k), dtype=input_dtype, device="cuda") / 100
    w1_fp = torch.randn((shape.e, 2 * shape.n, shape.k), dtype=input_dtype, device="cuda")
    w2_fp = torch.randn((shape.e, shape.k, shape.n), dtype=input_dtype, device="cuda")
    if is_fp8:
        qinput, a1_scale = pertoken_quant(input_fp, quant_dtype=dtypes.fp8)
        w1_q, w1_scale = pertoken_quant(w1_fp, quant_dtype=dtypes.fp8)
        w2_q, w2_scale = pertoken_quant(w2_fp, quant_dtype=dtypes.fp8)
    else:
        qinput, a1_scale = input_fp, None
        max_vals_w1 = torch.abs(w1_fp.to(torch.float32)).max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        w1_scale = max_vals_w1 / 127.0
        w1_q = (w1_fp / max_vals_w1 * 127.0).round().clamp(min=-128, max=127).to(torch.int8)
        max_vals_w2 = torch.abs(w2_fp.to(torch.float32)).max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        w2_scale = max_vals_w2 / 127.0
        w2_q = (w2_fp / max_vals_w2 * 127.0).round().clamp(min=-128, max=127).to(torch.int8)

    reference_w1, reference_w2 = w1_q, w2_q
    # Every registered channelwise W8A8 kernel consumes this packed layout,
    # including new shapes that do not yet have a runtime config.
    w1_q = w8a8_moe_repack_shuffle(w1_q.contiguous())
    w2_q = w8a8_moe_repack_shuffle(w2_q.contiguous())
    score = torch.randn((m, shape.e), dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = fused_topk(input_fp, score, shape.topk, True)
    return {
        "down_input": torch.randn((m * shape.topk, shape.n), device="cuda",
                                  dtype=torch.float16 if shape.dtype == "fp16" else torch.bfloat16) / 100,
        "input": qinput.contiguous(),
        "a1_scale": a1_scale.contiguous() if a1_scale is not None else None,
        "w1": w1_q.contiguous(),
        "w1_scale": w1_scale.contiguous(),
        "w2": w2_q.contiguous(),
        "w2_scale": w2_scale.contiguous(),
        "reference_w1": reference_w1,
        "reference_w2": reference_w2,
        "topk_weights": topk_weights.contiguous(),
        "topk_ids": topk_ids.contiguous(),
    }


def _make_sorted(data: Dict[str, Any], shape: Shape, bm: int, sorting: int):
    if sorting not in (0, 1):
        raise ValueError(f"unknown metadata layout F={sorting}")
    cache = data.setdefault("_sort_cache", {})
    cache_key = (int(bm), int(sorting))
    if cache_key in cache:
        return cache[cache_key]
    if sorting == 1:
        from moe_utils.moe_ops import moe_align_block_size

        result = moe_align_block_size(data["topk_ids"], bm, shape.e, None)
        cache[cache_key] = result
        return result

    from moe_utils.fused_moe_c import moe_sorting_ck
    from moe_utils.utils import dtypes

    sorted_token_ids, _, expert_ids, num_tokens_post_padded, _, _ = moe_sorting_ck(
        data["topk_ids"],
        data["topk_weights"],
        shape.e,
        shape.k,
        dtypes.bf16,
        bm,
        expert_mask=None,
    )
    result = (sorted_token_ids, expert_ids, num_tokens_post_padded)
    cache[cache_key] = result
    return result


def _valid_modes_for_shape(profile: QuantProfile, shape: Shape, stage: int,
                           config_declared_only: bool = False,
                           recommended_only: bool = False,
                           max_nloop: int = 5) -> List[int]:
    is_bottom = stage == 2
    modes = []
    for mode_i in _candidate_modes(profile, stage, shape.arch, config_declared_only):
        cand = profile.candidate(mode_i, shape)
        if cand.family in (2, 3) and profile.descriptor(mode_i, shape).nloop > max_nloop:
            continue
        if recommended_only and not profile.descriptor(mode_i).recommended(shape.m):
            continue
        if profile.validate(cand, shape, is_bottom) is None:
            modes.append(mode_i)
    return modes


def _w8a8_stage_reference(shape, data, stage, activation, activation_scale):
    import torch

    is_down = stage == "gemm2"
    width = shape.k if is_down else 2 * shape.n
    weight = data["reference_w2" if is_down else "reference_w1"]
    scale = data.get("reference_scale2" if is_down else "reference_scale1",
                     data["w2_scale" if is_down else "w1_scale"])
    a = activation.float()
    if activation_scale is not None:
        a = a * activation_scale.float().reshape(-1, 1)
    a = a.reshape(-1, a.shape[-1])
    reference = torch.empty((shape.m * shape.topk, width), device=a.device, dtype=torch.float32)
    routes = data["topk_ids"].reshape(-1)
    for expert in range(shape.e):
        indices = torch.nonzero(routes == expert, as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        input_indices = indices if is_down else indices // shape.topk
        w = weight[expert].float() * scale[expert].float().reshape(-1, 1)
        values = a[input_indices] @ w.T
        if is_down:
            values *= data["topk_weights"].reshape(-1)[indices, None]
        reference[indices] = values
    return reference.reshape(shape.m, shape.topk, width)


def _bench_w8a8_stage(
    shape: Shape,
    data: Dict[str, Any],
    config: Dict[str, Any],
    stage: str,
    quant: str,
    warmup: int,
    iters: int,
) -> DirectBenchRow:
    import torch

    from moe_utils.fused_moe_c import moe_kernel_prepare_input
    from moe_utils.moe_ops import moe_c_moe_gemm_marlin_w8a8, moe_c_moe_gemm_marlin_w8a8_fp8, moe_c_moe_gemm_marlin_w4a8
    from moe_utils.utils import dtypes, per_token_quant_hip

    mode_i = int(config["MODE"])
    cand = PROFILES[quant].candidate(mode_i, shape)
    bm = int(config.get("BLOCK_SIZE_M", cand.bm))
    sorting_name = "moe_align" if cand.sorting else "ck_sorting"
    is_fp8 = quant == "fp8_w8a8"
    output_dtype = torch.float16 if shape.dtype == "fp16" else dtypes.bf16

    def quantize_activation(a, b, b_scale):
        if is_fp8:
            return per_token_quant_hip(a, quant_dtype=torch.float8_e4m3fn)
        return moe_kernel_prepare_input(
            A=a,
            B=b,
            A_scale=None,
            B_scale=b_scale,
            use_fp8_w8a8=False,
            use_int8_w8a8=quant == "int8_w8a8",
            use_int8_w4a8=quant == "int8_w4a8",
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            use_fp4_w4a16=False,
            per_channel_quant=True,
            block_shape=None,
        )

    def invoke_gemm(a, b, c, a_scale, b_scale, routed_weight,
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    gemm_topk, mode_arg, real_topk, key_selected, real_size_k=0):
        if is_fp8:
            return moe_c_moe_gemm_marlin_w8a8_fp8(
                a, b, c, a_scale, b_scale, routed_weight,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                gemm_topk, mode_arg, real_topk, key_selected, real_size_k,
            )
        op = moe_c_moe_gemm_marlin_w4a8 if quant == "int8_w4a8" else moe_c_moe_gemm_marlin_w8a8
        return op(
            a, b, c, a_scale, b_scale, routed_weight,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            gemm_topk, mode_arg, real_topk, key_selected,
        )

    try:
        error = PROFILES[quant].validate(cand, shape, stage == "gemm2")
        if error:
            return DirectBenchRow(shape.m, stage, mode_i, bm, sorting_name, None, False, error)
        if bm != cand.bm:
            raise ValueError(f"BM={bm} does not match mode BM={cand.bm}")
        if stage == "gemm1":
            sorted_token_ids, expert_ids, num_tokens_post_padded = _make_sorted(data, shape, bm, cand.sorting)
            output = torch.empty((shape.m, shape.topk, 2 * shape.n), device="cuda", dtype=output_dtype)
            input_a, input_scale = data["input"], data["a1_scale"]
            if not is_fp8:
                input_a, input_scale = quantize_activation(data["input"], data["w1"], data["w1_scale"])

            def invoke():
                return invoke_gemm(
                    input_a, data["w1"], output, input_scale, data["w1_scale"], None,
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    shape.topk, mode_i, shape.topk, int(config.get("key_selected", shape.m)), shape.k,
                )
        else:
            if "_down_quant" not in data:
                data["_down_quant"] = quantize_activation(data["down_input"], data["w2"], data["w2_scale"])
            q2, a2_scale = data["_down_quant"]
            sorted_token_ids, expert_ids, num_tokens_post_padded = _make_sorted(
                data, shape, bm, cand.sorting)
            output = torch.empty((shape.m, shape.topk, shape.k), device="cuda", dtype=output_dtype)

            def invoke():
                return invoke_gemm(
                    q2, data["w2"], output, a2_scale, data["w2_scale"], data["topk_weights"],
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    1, mode_i, shape.topk, int(config.get("key_selected", shape.m)), 0,
                )
        invoke()
        torch.cuda.synchronize()
        reference_cache = data.setdefault("_reference_cache", {})
        reference_key = stage
        if reference_key not in reference_cache:
            reference_cache[reference_key] = _w8a8_stage_reference(
                shape, data, stage,
                q2 if stage == "gemm2" else input_a,
                a2_scale if stage == "gemm2" else input_scale,
            )
        reference = reference_cache[reference_key]
        # Avoid full-size FP32 comparison temporaries for large routed outputs.
        output_flat, reference_flat = output.reshape(-1), reference.reshape(-1)
        max_delta, matches = 0.0, True
        for start in range(0, output_flat.numel(), 1 << 24):
            actual = output_flat[start:start + (1 << 24)].float()
            expected = reference_flat[start:start + (1 << 24)]
            delta = (actual - expected).abs().max().item()
            max_delta = max(max_delta, delta)
            matches = torch.allclose(actual, expected, atol=1e-2, rtol=1e-2) and matches
        if not matches:
            return DirectBenchRow(shape.m, stage, mode_i, bm, sorting_name, None, False,
                                  f"correctness failed: max_delta={max_delta}", max_delta)
        latency_us = _event_time_us(invoke, warmup, iters)
        return DirectBenchRow(shape.m, stage, mode_i, bm, sorting_name, latency_us, True, max_delta=max_delta)
    except BaseException as exc:
        return DirectBenchRow(shape.m, stage, mode_i, bm, sorting_name, None, False, repr(exc))


def _bench_direct_child(args: argparse.Namespace) -> int:
    if args.device is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    sys.path.insert(0, str(ROOT))
    import torch

    torch.set_default_device("cuda")
    shape = Shape(args.e, args.n, args.k, args.topk, args.m[0], args.dtype or args.quant, args.arch, args.cu)
    data = _make_w8a8_inputs(shape.m, shape, args.quant)
    mode_i = int(args.direct_mode)
    cand = decode_mode(mode_i)
    config: Dict[str, Any] = {"MODE": mode_i, "BLOCK_SIZE_M": cand.bm, "key_selected": shape.m}
    row = _bench_w8a8_stage(shape, data, config, args.direct_stage, args.quant, args.warmup, args.iters)
    print("DIRECT_BENCH_JSON=" + json.dumps(asdict(row), sort_keys=True), flush=True)
    return 0 if row.ok else 2


def _run_direct_candidate_child(
    args: argparse.Namespace,
    m: int,
    stage: str,
    mode_i: int,
) -> DirectBenchRow:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--quant", args.quant,
        "--arch", args.arch,
        "--e", str(args.e),
        "--n", str(args.n),
        "--k", str(args.k),
        "--topk", str(args.topk),
        "--m", str(m),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--direct-child",
        "--direct-stage", stage,
        "--direct-mode", str(mode_i),
    ]
    if args.cu is not None:
        cmd.extend(["--cu", str(args.cu)])
    if args.dtype is not None:
        cmd.extend(["--dtype", args.dtype])
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.candidate_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        cand = decode_mode(mode_i)
        return DirectBenchRow(m, stage, mode_i, cand.bm, "moe_align" if cand.sorting else "ck_sorting", None, False, "timeout")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("DIRECT_BENCH_JSON="):
            try:
                payload = json.loads(line.split("=", 1)[1])
                return DirectBenchRow(**payload)
            except Exception as exc:
                cand = decode_mode(mode_i)
                return DirectBenchRow(m, stage, mode_i, cand.bm, "moe_align" if cand.sorting else "ck_sorting", None, False, f"json parse failed: {exc}")
    cand = decode_mode(mode_i)
    tail = (proc.stderr or proc.stdout)[-1000:].replace("\n", " | ")
    return DirectBenchRow(m, stage, mode_i, cand.bm, "moe_align" if cand.sorting else "ck_sorting", None, False, f"child rc={proc.returncode}: {tail}")


def bench_direct(args: argparse.Namespace) -> int:
    if args.device is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    sys.path.insert(0, str(ROOT))
    import torch

    torch.set_default_device("cuda")
    profile = PROFILES[args.quant]
    gemm1_configs = {} if args.sweep_direct else _load_shape_config(profile, args.arch, args.e, args.n, args.k, False, args.topk, args.cu)
    gemm2_configs = {} if args.sweep_direct else _load_shape_config(profile, args.arch, args.e, args.n, args.k, True, args.topk, args.cu)
    rows: List[DirectBenchRow] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cu_part = "none" if args.cu is None else str(args.cu)
    csv_path = args.out_dir / f"{args.quant}_{args.arch}_E{args.e}_N{args.n}_K{args.k}_topk{args.topk}_cu{cu_part}_direct_bench.csv"
    data = None
    preparation_errors = {}
    for m in args.m:
        data = None
        gc.collect()
        torch.cuda.empty_cache()
        shape = Shape(args.e, args.n, args.k, args.topk, m, args.dtype or args.quant, args.arch, args.cu)
        gemm1 = _select_config(gemm1_configs, m) if gemm1_configs else {}
        gemm2 = _select_config(gemm2_configs, m) if gemm2_configs else {}
        if args.sweep_direct:
            if args.direct_inprocess:
                try:
                    data = _make_w8a8_inputs(m, shape, args.quant)
                except torch.cuda.OutOfMemoryError as exc:
                    preparation_errors[str(m)] = str(exc)
                    (args.out_dir / "preparation_errors.json").write_text(json.dumps(preparation_errors, indent=2) + "\n")
                    print(f"M={m}: input preparation OOM: {exc}", flush=True)
                    continue
                for mode_i in _valid_modes_for_shape(profile, shape, 1, args.sweep_config_modes, args.recommended_only, args.max_nloop):
                    cand = decode_mode(mode_i)
                    config = {"MODE": mode_i, "BLOCK_SIZE_M": cand.bm, "key_selected": m}
                    rows.append(_bench_w8a8_stage(shape, data, config, "gemm1", args.quant, args.warmup, args.iters))
                for mode_i in _valid_modes_for_shape(profile, shape, 2, args.sweep_config_modes, args.recommended_only, args.max_nloop):
                    cand = decode_mode(mode_i)
                    config = {
                        "MODE": mode_i,
                        "BLOCK_SIZE_M": cand.bm,
                        "key_selected": m,
                    }
                    rows.append(_bench_w8a8_stage(shape, data, config, "gemm2", args.quant, args.warmup, args.iters))
            else:
                for mode_i in _valid_modes_for_shape(profile, shape, 1, args.sweep_config_modes, args.recommended_only, args.max_nloop):
                    row = _run_direct_candidate_child(args, m, "gemm1", mode_i)
                    rows.append(row)
                for mode_i in _valid_modes_for_shape(profile, shape, 2, args.sweep_config_modes, args.recommended_only, args.max_nloop):
                    row = _run_direct_candidate_child(args, m, "gemm2", mode_i)
                    rows.append(row)
        else:
            data = _make_w8a8_inputs(m, shape, args.quant)
            rows.append(_bench_w8a8_stage(shape, data, gemm1, "gemm1", args.quant, args.warmup, args.iters))
            rows.append(_bench_w8a8_stage(shape, data, gemm2, "gemm2", args.quant, args.warmup, args.iters))

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(DirectBenchRow.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        print(f"completed M={m}, candidate rows={len(rows)}", flush=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DirectBenchRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(f"direct bench written: {csv_path}")
    print("M,stage,mode,bm,sorting,latency_us,ok,error")
    for row in rows:
        latency = "" if row.latency_us is None else f"{row.latency_us:.3f}"
        print(f"{row.m},{row.stage},{row.mode},{row.bm},{row.sorting},{latency},{row.ok},{row.error}")
    if args.sweep_direct:
        print("best summary:")
        missing_best = False
        summary = {"shape": {"e": args.e, "n": args.n, "k": args.k, "topk": args.topk,
                             "cu": args.cu, "arch": args.arch, "quant": args.quant, "dtype": args.dtype},
                   "objective": "gemm1_us + gemm2_us, same BM and routing", "m_results": {},
                   "preparation_errors": preparation_errors}
        configs = {"gemm1": {}, "gemm2": {}}
        for m in args.m:
            groups = _best_bm_pairs(rows, m)
            best = min(groups, key=lambda group: group["latency_us"]) if groups else None
            summary["m_results"][str(m)] = {"bm_groups": groups, "best": best}
            if best is None:
                print(f"M={m}: no passing pair with the same BM and routing")
                missing_best = True
                continue
            for stage in ("gemm1", "gemm2"):
                configs[stage][str(m)] = {"MODE": best[stage]["mode"], "BLOCK_SIZE_M": best["bm"]}
            print(f"M={m} BM={best['bm']} routing={best['sorting']} modes={best['gemm1']['mode']}/{best['gemm2']['mode']} sum_us={best['latency_us']:.3f}")
        (args.out_dir / "best_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for stage in ("gemm1", "gemm2"):
            name = _config_file_names(args.e, args.n, args.k, args.quant, stage == "gemm2", args.topk, args.cu)[0]
            (args.out_dir / name).write_text(json.dumps(configs[stage], indent=2) + "\n")
        indtype = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}[args.dtype or "bf16"]
        csv_rows = rows_from_pair(configs["gemm1"], configs["gemm2"], args.arch, args.quant,
                                  args.e, args.n, args.k, args.topk, args.cu, (indtype,),
                                  {m: row["best"]["latency_us"] for m, row in summary["m_results"].items()
                                   if row["best"] is not None})
        write_config_csv(str(args.out_dir / CSV_NAMES[args.quant]), csv_rows)
        return 2 if missing_best else 0
    return 0 if rows and all(row.ok for row in rows) else 2


def _best_bm_pairs(rows: List[DirectBenchRow], m: int) -> List[Dict[str, Any]]:
    passed = [r for r in rows if r.m == m and r.ok and r.latency_us is not None]
    groups = []
    for bm, sorting in sorted({(r.bm, r.sorting) for r in passed}):
        up = [r for r in passed if (r.bm, r.sorting) == (bm, sorting) and r.stage == "gemm1"]
        down = [r for r in passed if (r.bm, r.sorting) == (bm, sorting) and r.stage == "gemm2"]
        if not up or not down:
            continue
        g1 = min(up, key=lambda r: r.latency_us)
        g2 = min(down, key=lambda r: r.latency_us)
        groups.append({"bm": bm, "sorting": sorting, "gemm1": asdict(g1), "gemm2": asdict(g2),
                       "latency_us": g1.latency_us + g2.latency_us})
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=sorted(PROFILES), required=True)
    parser.add_argument("--arch", default="gfx936")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--e", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--cu", "--num-cu", dest="cu", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--max-nloop", type=int, default=5)
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "tune/tune_results/hipc_audit")
    parser.add_argument("--run-correctness", action="store_true", help="run the existing end-to-end test after audit")
    parser.add_argument("--list-modes", action="store_true", help="export per-quant descriptors and shape rejection reasons without a GPU")
    parser.add_argument("--recommended-only", action="store_true", help="restrict sweep to each descriptor's recommended M range")
    parser.add_argument("--bench-direct", action="store_true", help="benchmark current config with direct moe_c GEMM calls")
    parser.add_argument("--sweep-direct", action="store_true", help="with --bench-direct, sweep instantiated FP8 modes instead of current config")
    parser.add_argument("--sweep-config-modes", action="store_true", help="with --sweep-direct, only sweep modes already declared by runtime configs")
    parser.add_argument("--direct-inprocess", action="store_true", help="with --sweep-direct, run candidates in one process for speed")
    parser.add_argument("--candidate-timeout", type=int, default=90)
    parser.add_argument("--direct-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--direct-stage", choices=("gemm1", "gemm2"), default="gemm1", help=argparse.SUPPRESS)
    parser.add_argument("--direct-mode", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--m", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.cu = resolve_cu(args.arch, args.cu)

    if args.list_modes:
        profile = PROFILES[args.quant]
        entries = []
        for desc in profile.descriptors.values():
            shapes = [Shape(args.e, args.n, args.k, args.topk, m,
                            args.dtype or args.quant, args.arch, args.cu) for m in args.m]
            entries.append({**asdict(desc), "tile_n": desc.tile_n,
                            "sorting_name": desc.sorting_name, "token_encoding": desc.token_encoding,
                            "shape_checks": [{"m": shape.m, "error": desc.validate(shape),
                                              "tile_n": desc.for_shape(shape).tile_n,
                                              "recommended": desc.recommended(shape.m)} for shape in shapes]})
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / f"{args.quant}_mode_descriptors.json"
        path.write_text(json.dumps(entries, indent=2) + "\n")
        print(f"[{args.quant}] descriptors={len(entries)} written to {path}")
        return 0

    if args.bench_direct or args.direct_child:
        if args.perf_only or args.skip_correctness:
            parser.error("independent GEMM tuning requires correctness before timing")
        if min(args.e, args.n, args.k, args.topk, *args.m) <= 0:
            parser.error("direct tuning requires positive E/N/K/topk/M")
        if args.warmup < 0 or args.iters < 1:
            parser.error("warmup must be nonnegative and iters must be positive")

    if args.device is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    if args.direct_child:
        return _bench_direct_child(args)
    if args.bench_direct:
        return bench_direct(args)
    profile = PROFILES[args.quant]
    results = audit_configs(profile, args.arch, args.out_dir, (args.e, args.n, args.k))
    print(f"[{profile.name}/{args.arch}] files={len({r.file for r in results})} entries={len(results)} "
          f"passed={sum(r.ok for r in results)} failed={sum(not r.ok for r in results)}")
    if any(not r.ok for r in results):
        for row in results:
            if not row.ok:
                print(f"CONFIG_FAIL file={row.file} token={row.token} mode={row.mode}: {row.error}")
        return 2
    if args.run_correctness:
        cmd = test_command(profile, args)
        print("RUN:", " ".join(cmd), flush=True)
        return subprocess.call(cmd, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
