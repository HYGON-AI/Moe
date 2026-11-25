#!/usr/bin/env python3
"""Unified HIPC moe_c config auditor and tune entry point.

The HIPC mode namespace is shared by the HIPC quantizers and has the
fixed-width form ``S F K MM VVV``::

    S   stage (1=GEMM1, 2=GEMM2)
    F   metadata layout (0=CK sorting, 1=moe_align)
    K   kernel family
    MM  BLOCK_SIZE_M / 16
    VVV profile (the profile owns N_LOOP and other compile-time constants)

This file deliberately keeps quantizer-specific legality in descriptors.  The
common flow decodes modes, validates shapes, and tunes independent GEMM stages.
Quantizer-specific input preparation, references and pairing preserve the runtime
contract for each profile.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from operator import attrgetter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe_utils.channel_mode import mode
from moe_utils.w4a16_modes import (BSTAT_FALLBACK_TILE, DECODE_BK, DECODE_K, W4A16_CSV_NAME, W4A16_GROUP_SIZE,
                                   W4A16_MODE_SCHEMA, W4A16_MODES_BY_STAGE, W4A16_SPECS, resolve_template)
from tune.hipc_runtime import CHECKOUT_ROOT, CONFIG_ROOT, IN_AITER, fused_moe, ops
from tune.w4a16_tune_utils import W4A16_TOLERANCES, W4A16BenchRow, bench_candidate, make_inputs


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
    group_size: int = W4A16_GROUP_SIZE


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
        effective = self.for_shape(shape)
        out_n = shape.k if self.stage == 2 else 2 * shape.n
        reduce_k = shape.n if self.stage == 2 else shape.k
        if self.quant == "int8_w8a8" and self.family == 3:
            rows = shape.m if self.stage == 1 else shape.m * shape.topk
            if rows * reduce_k >= 1 << 31 or out_n * reduce_k >= 1 << 31:
                return "WASP exceeds the 2 GiB input/expert buffer limit"
        # Zero dimensions are unknown in config filenames, not a tuning shape.
        if out_n:
            if effective.exact_out_n is not None and out_n != effective.exact_out_n:
                return f"out_n={out_n}, expected {effective.exact_out_n}"
            tile_n = effective.tile_n
            if not effective.supports_n_tail and out_n % tile_n:
                return f"out_n={out_n} is not divisible by tile_n={tile_n}"
        if reduce_k:
            if effective.supported_reduce_k and reduce_k not in effective.supported_reduce_k:
                return f"reduce_k={reduce_k} is not in {effective.supported_reduce_k}"
            if reduce_k < effective.min_reduce_k:
                return f"reduce_k={reduce_k} is below minimum {effective.min_reduce_k}"
            remainder = reduce_k % effective.k_step
            if remainder and remainder != effective.k_tail:
                return f"reduce_k={reduce_k} needs step={effective.k_step}, tail={effective.k_tail}"
        return None


@dataclass(frozen=True)
class W4A16ModeDescriptor(ModeDescriptor):
    implementation: str = ""
    template: Optional[Dict[str, int]] = None
    fp16_overrides: Optional[Dict[str, int]] = None
    unfused_mode: Optional[int] = None
    selection_notes: Tuple[str, ...] = ()

    def for_shape(self, shape: Shape) -> "W4A16ModeDescriptor":
        out_n = shape.k if self.stage == 2 else 2 * shape.n
        reduce_k = shape.n if self.stage == 2 else shape.k
        implementation, params, notes = resolve_template(
            self.implementation, self.template, self.fp16_overrides,
            self.stage, shape.dtype, out_n, reduce_k,
            output_rows=shape.m * shape.topk,
        )
        fields = _w4a16_launch_fields(self.stage, implementation, params)
        return replace(self, implementation=implementation, template=params,
                       fp16_overrides=None, selection_notes=tuple(notes), **fields)

    def validate(self, shape: Shape) -> Optional[str]:
        reduce_k = shape.k if self.stage == 1 else shape.n
        if self.implementation == "legacy_decode" and shape.group_size == 128:
            return "legacy decode requires group_size=32"
        if (self.implementation == "legacy_decode" and self.template["st"] == 1
                and reduce_k and reduce_k != 256 and reduce_k % (8 * self.template["bk"])):
            return f"legacy single-stage decode cannot pack scales for reduce_k={reduce_k}"
        return super().validate(shape)


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


def _detect_arch(default: str = "gfx936") -> str:
    """gcnArchName of device 0 (honouring --device via HIP_VISIBLE_DEVICES in the environment)."""
    try:
        import torch

        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return default


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
    for stage, candidates in ((1, gemm1), (2, gemm2)):
        candidates += [(stage, sorting, 3, bm, profile)
                       for sorting in (0, 1) for bm in (128, 256)
                       for profile in (101, 102)]
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
    if quant == W4A16_QUANT:
        spec = W4A16_SPECS[cand.mode]
        unfused_mode = (mode(cand.stage, cand.sorting, cand.family, cand.bm, spec.unfused_profile)
                        if spec.unfused_profile is not None else None)
        return W4A16ModeDescriptor(
            quant=quant, mode=cand.mode, stage=cand.stage, sorting=cand.sorting,
            family=cand.family, bm=cand.bm, implementation=spec.implementation,
            template=spec.template, fp16_overrides=spec.fp16_overrides,
            unfused_mode=unfused_mode,
            **_w4a16_launch_fields(cand.stage, spec.implementation, spec.template),
        )
    if quant == "int8_w8a8" and cand.family == 3:
        waves_m = cand.profile - 100
        return ModeDescriptor(
            quant=quant, mode=cand.mode, stage=cand.stage, sorting=cand.sorting,
            family=cand.family, kernel="launch_moe_w8a8_int8_wasp", bm=cand.bm,
            bn=256, bk=128, wm=cand.bm // waves_m, wn=32 * waves_m, wk=32,
            stages=2, nloop=1, k_step=128, min_reduce_k=128,
            input_dtypes=("bf16",), recommended_m_min=513,
            source="csrc_for_aiter/936_938/moe_w8a8_int8_wasp.h",
        )
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


def _w4a16_launch_fields(stage: int, implementation: str, params: Dict[str, int]) -> Dict[str, Any]:
    fields = dict(nloop=1, supports_n_tail=False, supported_reduce_k=(), k_step=256,
                  bn=0, bk=128, wm=0, wn=0, wk=0, stages=2)
    if implementation == "legacy_decode":
        suffix = "first" if stage == 1 else "second"
        fields.update(kernel=f"launch_moe_w4a16_{suffix}_stage_decode", stages=params["st"],
                      k_step=2 * params["bk"] if params["st"] == 2 else params["bk"],
                      # Keep a full output tile for these retune-only candidates.
                      **{key: params[key] for key in ("bn", "bk", "wm", "wn", "wk")})
        source = "moe_w4a16_opt.h"
    elif implementation == "decode":
        fields.update(kernel="launch_moe_w4a16_decode", wn=params["nn"] * 16,
                      bn=params["wn"] * params["nn"] * 16, bk=DECODE_BK, wk=params.get("spw", 0) * DECODE_BK,
                      supported_reduce_k=DECODE_K[stage], k_step=params["kw"] * DECODE_BK)
        source = "moe_w4a16_decode.h"
    else:
        fields.update(kernel="launch_moe_w4a16_prefill", wn=params["nn"] * 16,
                      bn=params["warps"] * params["nn"] * 16,
                      nloop=params.get("nch", 1), stages=params.get("st", 2))
        if implementation == "bstat":
            fields.update(kernel="launch_moe_w4a16_g2_bstat", supported_reduce_k=(256,))
        source = "moe_w4a16_prefill.h"
    return dict(fields, source=f"csrc_for_aiter/936_938/{source}")


W4A16_QUANT = "int4_w4a16"
DEFAULT_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
DIRECT_SEED = 0
DIRECT_DTYPES = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}
PROFILES: Dict[str, QuantProfile] = {
    "int4_w4a16": QuantProfile("int4_w4a16", ("int4_w4a16",), (1,), (0, 1), (), W4A16_MODES_BY_STAGE),
    "fp8_w8a8": QuantProfile("fp8_w8a8", ("fp8_w8a8",), (0, 1), (0, 1, 2), (201, 202, 203, 204), _w8a8_fp8_modes()),
    "int8_w8a8": QuantProfile("int8_w8a8", ("int8_w8a8",), (0, 1), (0, 1, 2, 3), (201, 202, 203, 204), _w8a8_int8_modes()),
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
    runtime = fused_moe()
    CSV_NAMES, QUANT_TYPES = runtime.CSV_NAMES, runtime.QUANT_TYPES

    results: List[BenchResult] = []
    records = []
    csv_path = CONFIG_ROOT / CSV_NAMES[profile.name]
    if csv_path.is_file():
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["arch"] != arch or row["quant_type"] != QUANT_TYPES[profile.name]:
                    continue
                for stage, mode_i in enumerate(row["sol_id"].split("+"), 1):
                    file_shape = Shape(int(row["expert"]), int(row["inter_dim"]), int(row["model_dim"]),
                                       int(row["topk"] or 0), int(row["token"]),
                                       "fp16" if row["indtype"] == "torch.float16" else "bf16", arch,
                                       int(row["cu_num"]) if row["cu_num"] else None)
                    records.append((csv_path, {row["token"]: {"MODE": int(mode_i)}}, stage == 2, file_shape))
    elif profile.name != W4A16_QUANT:  # archived W4A16 JSON IDs are incompatible
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
                str(path.relative_to(CHECKOUT_ROOT)), file_shape.topk, file_shape.cu,
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
    runtime = fused_moe()
    CSV_NAMES, find_csv_config = runtime.CSV_NAMES, runtime.find_csv_config

    if (CONFIG_ROOT / CSV_NAMES[profile.name]).is_file():
        result = find_csv_config(str(CONFIG_ROOT), arch, profile.name, e, n, k,
                                 topk, cu, is_bottom)
        if result is None:
            raise FileNotFoundError(f"No CSV config for {profile.name}/{arch}, N={n}, K={k}")
        return result
    if profile.name == W4A16_QUANT:
        raise FileNotFoundError(f"Missing W4A16 CSV: {CONFIG_ROOT / CSV_NAMES[profile.name]}")
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
    runtime = fused_moe()
    csv_path = CONFIG_ROOT / runtime.CSV_NAMES[profile.name]
    modes = set()
    if csv_path.is_file():
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["arch"] == arch and row["quant_type"] == runtime.QUANT_TYPES[profile.name]:
                    modes.add(int(row["sol_id"].split("+")[stage - 1]))
    elif profile.name != W4A16_QUANT:
        for path in _arch_files(arch, profile):
            if _is_bottom(path) != (stage == 2):
                continue
            try:
                with path.open() as stream:
                    modes.update(int(item["MODE"]) for item in json.load(stream).values())
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(modes.intersection(profile.candidate_modes(stage)))


def _stage_descriptors(profile: QuantProfile, stage: int, selected=None,
                       *, arch=None, config_declared_only=False) -> Tuple[ModeDescriptor, ...]:
    modes = profile.candidate_modes(stage)
    if selected is not None:
        invalid = sorted(set(selected) - set(modes))
        if invalid:
            raise ValueError(f"Unknown GEMM{stage} {profile.name} modes: {invalid}")
        modes = sorted(set(selected))
    elif config_declared_only and arch:
        modes = _config_declared_modes(profile, arch, stage) or modes
    return tuple(profile.descriptor(value) for value in modes)


def _make_w8a8_inputs(m: int, shape: Shape, quant: str) -> Dict[str, Any]:
    import torch

    fused_topk = fused_moe().fused_topk
    from moe_utils.shuffle import w8a8_moe_repack_shuffle
    from moe_utils.utils import dtypes, pertoken_quant

    torch.manual_seed(DIRECT_SEED)
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


def _make_sorted(data: Dict[str, Any], shape: Shape, bm: int, sorting: int, *, align=None):
    if sorting not in (0, 1):
        raise ValueError(f"unknown metadata layout F={sorting}")
    cache = data.setdefault("_sort_cache", {})
    cache_key = (int(bm), int(sorting))
    if cache_key in cache:
        return cache[cache_key]
    if sorting == 1:
        if align is None:
            align = (fused_moe() if IN_AITER else ops()).moe_align_block_size

        result = align(data["topk_ids"], bm, shape.e, None)
        cache[cache_key] = result
        return result

    moe_sorting_ck = fused_moe().moe_sorting_ck
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
                           max_nloop: Optional[int] = 5, *,
                           descriptors: Optional[Tuple[ModeDescriptor, ...]] = None,
                           on_unsupported=None) -> List[int]:
    is_bottom = stage == 2
    modes = []
    if descriptors is None:
        descriptors = _stage_descriptors(profile, stage, arch=shape.arch, config_declared_only=config_declared_only)
    for desc in descriptors:
        if max_nloop is not None and desc.family in (2, 3) and desc.for_shape(shape).nloop > max_nloop:
            continue
        if recommended_only and not desc.recommended(shape.m):
            continue
        error = profile.validate(decode_mode(desc.mode), shape, is_bottom)
        if error is None:
            modes.append(desc.mode)
        elif on_unsupported is not None:
            on_unsupported(desc, error)
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

    from moe_utils.utils import dtypes, per_token_quant_hip

    runtime, kernel_ops = fused_moe(), ops()
    mode_i = int(config["MODE"])
    cand = PROFILES[quant].candidate(mode_i, shape)
    bm = int(config.get("BLOCK_SIZE_M", cand.bm))
    sorting_name = "moe_align" if cand.sorting else "ck_sorting"
    is_fp8 = quant == "fp8_w8a8"
    output_dtype = torch.float16 if shape.dtype == "fp16" else dtypes.bf16

    def quantize_activation(a, b, b_scale):
        if is_fp8:
            return per_token_quant_hip(a, quant_dtype=torch.float8_e4m3fn)
        return runtime.moe_kernel_prepare_input(
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
            return kernel_ops.moe_c_moe_gemm_marlin_w8a8_fp8(
                a, b, c, a_scale, b_scale, routed_weight,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                gemm_topk, mode_arg, real_topk, key_selected, real_size_k,
            )
        op = (kernel_ops.moe_c_moe_gemm_marlin_w4a8 if quant == "int8_w4a8"
              else kernel_ops.moe_c_moe_gemm_marlin_w8a8)
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


def _run_direct_sweep(args, shape, stages, *, progress_csv=None, checkpoint=None):
    """Run the same stage sweep with either resumable batches or per-mode isolation."""
    import torch

    profile = PROFILES[args.quant]
    resumable = checkpoint is not None
    completed = _read_w4a16_checkpoint(checkpoint) if resumable else {}
    rows = list(completed.values())
    preparation_errors = {}
    if resumable:
        moe_align_block_size = fused_moe().moe_align_block_size

        state_path = args.w4a16_worker_state
        dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16
        torch.manual_seed(DIRECT_SEED)

    def save(row):
        if resumable:
            # Commit the result before clearing the in-flight marker.
            _append_w4a16_checkpoint(checkpoint, row)
            completed[(row.m, row.stage, row.mode)] = row
            latency = "" if row.latency_us is None else f"{row.latency_us:.2f}us"
            print(f"M={row.m} {row.stage} mode={row.mode} BM={row.bm} {row.status} {latency} {row.error}", flush=True)
        rows.append(row)

    for m in args.m:
        workload = replace(shape, m=m)
        if not resumable:
            gc.collect()
            torch.cuda.empty_cache()

        def unsupported(desc, error):
            save(W4A16BenchRow(m, f"gemm{desc.stage}", desc.mode, desc.bm, status="unsupported", error=error))

        pending = []
        for stage, descriptors in stages.items():
            if resumable:
                descriptors = tuple(d for d in descriptors if (m, f"gemm{stage}", d.mode) not in completed)
            modes = _valid_modes_for_shape(
                profile, workload, stage, recommended_only=args.recommended_only,
                max_nloop=None if resumable else args.max_nloop, descriptors=descriptors,
                on_unsupported=unsupported if resumable else None,
            )
            pending.extend(profile.descriptor(mode_i) for mode_i in modes)

        if pending:
            if resumable:
                topk_ids, *inputs = make_inputs(workload, dtype, DIRECT_SEED)
                sort_cache = {}
                data = {"topk_ids": topk_ids, "topk_weights": inputs[1].topk_weights, "_sort_cache": sort_cache}
            elif args.direct_inprocess:
                try:
                    data = _make_w8a8_inputs(m, workload, args.quant)
                except torch.cuda.OutOfMemoryError as exc:
                    preparation_errors[str(m)] = str(exc)
                    (args.out_dir / "preparation_errors.json").write_text(json.dumps(preparation_errors, indent=2) + "\n")
                    print(f"M={m}: input preparation OOM: {exc}", flush=True)
                    continue

            for desc in pending:
                name = f"gemm{desc.stage}"
                if resumable:
                    key = (desc.bm, desc.sorting)
                    needs_routing = key not in sort_cache
                    routing = _make_sorted(data, workload, *key, align=moe_align_block_size)
                    if needs_routing:
                        torch.cuda.synchronize()  # a routing failure is not a GEMM failure
                    marker = dict(checkpoint=str(checkpoint), m=m, stage=name, mode=desc.mode, bm=desc.bm)
                    tmp = state_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(marker))
                    tmp.replace(state_path)
                    row = bench_candidate(m, desc, inputs[desc.stage - 1], routing, args.warmup, args.iters, _event_time_us)
                elif args.direct_inprocess:
                    config = {"MODE": desc.mode, "BLOCK_SIZE_M": desc.bm, "key_selected": m}
                    row = _bench_w8a8_stage(workload, data, config, name, args.quant, args.warmup, args.iters)
                else:
                    row = _run_direct_candidate_child(args, m, name, desc.mode)
                save(row)
                if resumable:
                    state_path.unlink()
            if resumable:
                del data, inputs, routing, sort_cache, topk_ids
                torch.cuda.empty_cache()
            elif args.direct_inprocess:
                del data

        if progress_csv is not None:
            _write_bench_csv(progress_csv, DirectBenchRow, rows)
            print(f"completed M={m}, candidate rows={len(rows)}", flush=True)
    return rows, preparation_errors


def bench_direct(args: argparse.Namespace, stages) -> int:
    if args.device is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)
    sys.path.insert(0, str(ROOT))
    import torch

    torch.set_default_device("cuda")
    profile = PROFILES[args.quant]
    shape = Shape(args.e, args.n, args.k, args.topk, args.m[0], args.dtype or args.quant, args.arch, args.cu)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cu_part = "none" if args.cu is None else str(args.cu)
    csv_path = args.out_dir / f"{args.quant}_{args.arch}_E{args.e}_N{args.n}_K{args.k}_topk{args.topk}_cu{cu_part}_direct_bench.csv"
    if args.sweep_direct:
        rows, preparation_errors = _run_direct_sweep(args, shape, stages, progress_csv=csv_path)
    else:
        gemm1_configs = _load_shape_config(profile, args.arch, args.e, args.n, args.k, False, args.topk, args.cu)
        gemm2_configs = _load_shape_config(profile, args.arch, args.e, args.n, args.k, True, args.topk, args.cu)
        rows = []
        for m in args.m:
            gc.collect()
            torch.cuda.empty_cache()
            workload = replace(shape, m=m)
            data = _make_w8a8_inputs(m, workload, args.quant)
            for name, configs in (("gemm1", gemm1_configs), ("gemm2", gemm2_configs)):
                config = _select_config(configs, m) if configs else {}
                rows.append(_bench_w8a8_stage(workload, data, config, name, args.quant, args.warmup, args.iters))
            del data
            _write_bench_csv(csv_path, DirectBenchRow, rows)
            print(f"completed M={m}, candidate rows={len(rows)}", flush=True)

    _write_bench_csv(csv_path, DirectBenchRow, rows)
    print(f"direct bench written: {csv_path}")
    print("M,stage,mode,bm,sorting,latency_us,ok,error")
    for row in rows:
        latency = "" if row.latency_us is None else f"{row.latency_us:.3f}"
        print(f"{row.m},{row.stage},{row.mode},{row.bm},{row.sorting},{latency},{row.ok},{row.error}")
    if args.sweep_direct:
        CSV_NAMES = fused_moe().CSV_NAMES

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
        times = {m: row["best"]["latency_us"] for m, row in summary["m_results"].items()
                 if row["best"] is not None}
        _write_tuned_pair_csv(args.out_dir / CSV_NAMES[args.quant], args.quant, args, configs, times)
        return 2 if missing_best else 0
    return 0 if rows and all(row.ok for row in rows) else 2


def _write_bench_csv(path, row_type, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_type.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_tuned_pair_csv(path, quant, shape, configs, times):
    runtime = fused_moe()

    dtype = DIRECT_DTYPES[shape.dtype or "bf16"]
    group = {"group_size": shape.group_size} if quant == W4A16_QUANT else {}
    rows = runtime.rows_from_pair(configs["gemm1"], configs["gemm2"], shape.arch, quant,
                                 shape.e, shape.n, shape.k, shape.topk, shape.cu, (dtype,), times, **group)
    runtime.write_config_csv(path, rows)


def _fastest_stage_pair(rows, score=attrgetter("latency_us")):
    up = min((r for r in rows if r.stage == "gemm1"), key=score, default=None)
    down = min((r for r in rows if r.stage == "gemm2"), key=score, default=None)
    if up is not None and down is not None:
        return up, down
    return None


def _best_bm_pairs(rows: List[DirectBenchRow], m: int) -> List[Dict[str, Any]]:
    passed = [r for r in rows if r.m == m and r.ok and r.latency_us is not None]
    groups = []
    for bm, sorting in sorted({(r.bm, r.sorting) for r in passed}):
        pair = _fastest_stage_pair([r for r in passed if (r.bm, r.sorting) == (bm, sorting)])
        if pair is None:
            continue
        g1, g2 = pair
        groups.append({"bm": bm, "sorting": sorting, "gemm1": asdict(g1), "gemm2": asdict(g2),
                       "latency_us": g1.latency_us + g2.latency_us})
    return groups


def _best_w4a16_pair(rows: List[W4A16BenchRow], m: int) -> Optional[Dict[str, Any]]:
    """Fastest passing GEMM1 and GEMM2, chosen independently: their BMs may differ because the
    runtime aligns the routing once per distinct BM (dual align)."""
    passed = [r for r in rows if r.m == m and r.status == "ok"]
    pair = _fastest_stage_pair(passed, lambda r: r.latency_us + (r.epilogue_us if r.stage == "gemm1" else 0))
    if pair is None:
        return None
    g1, g2 = pair
    return {"gemm1": asdict(g1), "gemm2": asdict(g2), "dual_routing": g1.bm != g2.bm,
            "latency_us": g1.latency_us + g1.epilogue_us + g2.latency_us}


def _w4a16_trial_settings(args, stages):
    return dict(warmup=args.warmup, iters=args.iters, seed=DIRECT_SEED, **W4A16_TOLERANCES,
                backend="aiter_jit" if IN_AITER else "standalone", recommended_only=args.recommended_only,
                modes={stage: [desc.mode for desc in descs] for stage, descs in stages.items()})


def _read_w4a16_checkpoint(path: Path):
    if not path.is_file():
        return {}
    with path.open() as stream:
        rows = (W4A16BenchRow(**json.loads(line)) for line in stream if line.strip())
        return {(row.m, row.stage, row.mode): row for row in rows}


def _append_w4a16_checkpoint(path: Path, row: W4A16BenchRow):
    with path.open("a") as stream:
        stream.write(json.dumps(asdict(row)) + "\n")


def _record_w4a16_crash(state_path: Path, rc: int) -> bool:
    if not state_path.is_file():
        return False
    marker = json.loads(state_path.read_text())
    checkpoint = Path(marker["checkpoint"])
    key = (marker["m"], marker["stage"], marker["mode"])
    # A worker can exit after committing its result and before clearing the marker.
    if key not in _read_w4a16_checkpoint(checkpoint):
        row = W4A16BenchRow(*key, marker["bm"], status="crashed", error=f"worker exited with {rc}")
        _append_w4a16_checkpoint(checkpoint, row)
        print(f"M={row.m} {row.stage} mode={row.mode} crashed ({rc}); resuming", flush=True)
    state_path.unlink()
    return True


def _supervise_w4a16(args, argv) -> int:
    # The marker belongs to this invocation, even if unrelated sweeps share out-dir.
    with tempfile.TemporaryDirectory(prefix="hipc-w4a16-") as directory:
        state_path = Path(directory) / "inflight.json"
        if args.fresh:
            state_path.with_suffix(".fresh").touch()
        cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--w4a16-worker-state", str(state_path)]
        while True:
            rc = subprocess.call(cmd)
            if rc in (0, 2) or not _record_w4a16_crash(state_path, rc):
                return rc


def _write_w4a16_results(args, shape, settings, rows, tag) -> int:
    rows_path = args.out_dir / f"{tag}_candidates.csv"
    rows = sorted((row for row in rows if row.m in args.m), key=lambda r: (r.m, r.stage, r.mode))
    _write_bench_csv(rows_path, W4A16BenchRow, rows)
    summary = {"mode_schema": W4A16_MODE_SCHEMA,
               "shape": {"arch": shape.arch, "cu": shape.cu, "e": shape.e, "n": shape.n,
                         "k": shape.k, "topk": shape.topk, "dtype": shape.dtype, "group_size": shape.group_size},
               "settings": settings,
               "objective": "gemm1_us (+ silu_and_mul_us if unfused) + gemm2_us; BMs may differ (dual align)",
               "m_results": {}}
    new_rows = [row for row in rows if W4A16_SPECS[row.mode].implementation != "legacy_decode"]
    configs = {"gemm1": {}, "gemm2": {}}
    times = {}
    for m in args.m:
        best = _best_w4a16_pair(rows, m)
        summary["m_results"][str(m)] = {"best": best, "best_without_legacy_decode": _best_w4a16_pair(new_rows, m)}
        if best is None:
            print(f"M={m}: no passing GEMM1/GEMM2 candidate pair", flush=True)
            continue
        for stage in configs:
            configs[stage][m] = {"MODE": best[stage]["mode"], "BLOCK_SIZE_M": best[stage]["bm"]}
        times[str(m)] = best["latency_us"]
        print(f"M={m} modes={best['gemm1']['mode']}+{best['gemm2']['mode']} "
              f"BM={best['gemm1']['bm']}/{best['gemm2']['bm']} sum_us={best['latency_us']:.2f}", flush=True)
    (args.out_dir / f"{tag}_best_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if times:
        config_csv = args.out_dir / W4A16_CSV_NAME
        _write_tuned_pair_csv(config_csv, W4A16_QUANT, shape, configs, times)
        print(f"tuned rows merged into {config_csv}", flush=True)
    print(f"candidates: {rows_path}", flush=True)
    return 0 if len(times) == len(args.m) else 2


def _bench_w4a16(args, stages) -> int:
    import torch

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch, cu = device.gcnArchName.split(":")[0], device.multi_processor_count
    if args.arch is not None and args.arch != arch:
        raise ValueError(f"Requested --arch={args.arch}, but the active GPU is {arch}")
    if args.cu is not None and args.cu != cu:
        raise ValueError(f"Requested --cu={args.cu}, but the active GPU has {cu} CUs")
    shape = Shape(args.e, args.n, args.k, args.topk, args.m[0], args.dtype, arch, cu, args.group_size)
    settings = _w4a16_trial_settings(args, stages)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    # Keep existing group-32 checkpoint names and give group-128 its own files.
    group_tag = "_gs128" if shape.group_size == 128 else ""
    tag = (f"{W4A16_MODE_SCHEMA}{group_tag}_{arch}_E{shape.e}_N{shape.n}_K{shape.k}_topk{shape.topk}"
           f"_cu{cu}_{shape.dtype}_{signature}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir.resolve() / f"{tag}_candidates.jsonl"
    state_path = args.w4a16_worker_state
    fresh = state_path.with_suffix(".fresh")
    if fresh.is_file():
        if checkpoint.is_file():
            checkpoint.unlink()
        fresh.unlink()
    rows, _ = _run_direct_sweep(args, shape, stages, checkpoint=checkpoint)
    return _write_w4a16_results(args, shape, settings, rows, tag)


def _tune_w4a16(args, parser, argv, stages) -> int:
    if not args.bench_direct or not args.sweep_direct:
        parser.error("--quant int4_w4a16 requires --bench-direct --sweep-direct (or --list-modes)")
    if (args.perf_only or args.skip_correctness or args.sweep_config_modes or args.direct_inprocess
            or args.run_correctness or args.direct_child):
        parser.error("W4A16 sweeps always validate candidates in a supervised worker; "
                     "the channelwise config, audit and in-process options do not apply")
    args.dtype = args.dtype or "bf16"
    if args.dtype not in DIRECT_DTYPES:
        parser.error("W4A16 requires --dtype bf16 or fp16")
    if args.k % 128 or args.n % 128:
        parser.error("W4A16 K and N must be multiples of 128")
    if args.topk > args.e:
        parser.error("topk must be <= E")
    args.m = list(dict.fromkeys(args.m))
    return _bench_w4a16(args, stages) if args.w4a16_worker_state else _supervise_w4a16(args, argv)


def _list_modes(args: argparse.Namespace) -> int:
    profile = PROFILES[args.quant]
    shapes = [Shape(args.e, args.n, args.k, args.topk, m,
                    args.dtype or ("bf16" if args.quant == W4A16_QUANT else args.quant),
                    args.arch, args.cu, args.group_size) for m in args.m]
    entries = []
    for desc in profile.descriptors.values():
        entry = dict(asdict(desc), tile_n=desc.tile_n, sorting_name=desc.sorting_name,
                     token_encoding=desc.token_encoding, shape_checks=[])
        w4a16 = isinstance(desc, W4A16ModeDescriptor)
        if w4a16:
            entry.update(profile=desc.mode % 1000, mode_schema=W4A16_MODE_SCHEMA)
        for shape in shapes:
            effective = desc.for_shape(shape)
            check = dict(m=shape.m, arch=shape.arch, cu=shape.cu, dtype=shape.dtype,
                         error=desc.validate(shape), tile_n=effective.tile_n,
                         recommended=desc.recommended(shape.m), effective_kernel=effective.kernel,
                         tile=dict(bm=effective.bm, bn=effective.bn, bk=effective.bk, nloop=effective.nloop))
            if w4a16:
                check.update(group_size=shape.group_size, effective_template=effective.template,
                             selection_notes=effective.selection_notes)
                spec = W4A16_SPECS[desc.mode]
                if effective.implementation == "decode":
                    params = effective.template
                    check["tile"].update(
                        reduce_k=shape.k if spec.stage == 1 else shape.n,
                        prefetch_depth=spec.template["pd"],
                        effective_prefetch_depth=params["pd"] if "spw" in params else None,
                        epilogue=("DownWeighted" if spec.stage == 2 else
                                  "UpSilu" if params["silu"] else "Up"),
                    )
                elif effective.implementation == "prefill":
                    tile = BSTAT_FALLBACK_TILE if spec.implementation == "bstat" else spec.prefill_tile
                    if shape.dtype == "fp16" and spec.fp16_prefill_tile:
                        tile = spec.fp16_prefill_tile
                    check["prefill_tile"] = dict(
                        bm=spec.bm, bn=tile.bn, bk=128, nloop=tile.nloop,
                        effective_nloop=effective.template["nch"], pipeline=tile.pipeline,
                        epilogue=("DownWeighted" if spec.stage == 2 else
                                  "UpSilu" if effective.template["silu"] else "Up"),
                    )
                elif effective.implementation == "bstat":
                    check["prefill_tile"] = dict(
                        bm=desc.bm, bn=effective.template["warps"] * effective.template["nn"] * 16,
                        bk=128, pipeline="ResidentB", epilogue="DownWeighted",
                    )
            entry["shape_checks"].append(check)
        entries.append(entry)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.quant}_mode_descriptors.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"[{args.quant}] descriptors={len(entries)} written to {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=sorted(PROFILES), required=True)
    parser.add_argument("--arch", default=None,
                        help="default: detected from the device (gfx936 when no GPU is visible)")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--e", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--group-size", type=int, choices=(32, 128),
                        help="INT4 W4A16 group size (default: 32)")
    parser.add_argument("--cu", "--num-cu", dest="cu", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--max-nloop", type=int, default=5, help="maximum NLoop for channelwise loop families")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "tune/tune_results/hipc_audit")
    parser.add_argument("--run-correctness", action="store_true", help="run the existing end-to-end test after audit")
    parser.add_argument("--list-modes", action="store_true", help="export per-quant descriptors and shape rejection reasons without a GPU")
    parser.add_argument("--recommended-only", action="store_true", help="restrict sweep to each descriptor's recommended M range")
    parser.add_argument("--bench-direct", action="store_true", help="benchmark current config with direct moe_c GEMM calls")
    parser.add_argument("--sweep-direct", action="store_true", help="with --bench-direct, sweep legal instantiated modes instead of current config")
    parser.add_argument("--sweep-config-modes", action="store_true", help="with --sweep-direct, only sweep modes already declared by runtime configs")
    parser.add_argument("--direct-inprocess", action="store_true", help="channelwise sweep: run candidates in one process for speed")
    parser.add_argument("--candidate-timeout", type=int, default=90, help="channelwise worker timeout (seconds)")
    parser.add_argument("--direct-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--direct-stage", choices=("gemm1", "gemm2"), default="gemm1", help=argparse.SUPPRESS)
    parser.add_argument("--direct-mode", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--m", nargs="+", type=int, default=list(DEFAULT_M),
                        help=f"token counts (default: {' '.join(map(str, DEFAULT_M))})")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gemm1-modes", nargs="+", type=int, help="restrict GEMM1 in a direct sweep")
    parser.add_argument("--gemm2-modes", nargs="+", type=int, help="restrict GEMM2 in a direct sweep")
    parser.add_argument("--fresh", action="store_true", help="restart a resumable sweep instead of using its checkpoint")
    parser.add_argument("--w4a16-worker-state", type=Path, help=argparse.SUPPRESS)
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    if args.group_size is not None and args.quant != W4A16_QUANT:
        parser.error("--group-size applies only to --quant int4_w4a16")
    args.group_size = args.group_size or W4A16_GROUP_SIZE
    if args.device is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)  # before CUDA initialises
    # W4A16 workers check explicit arch/CU against the actual GPU. List mode needs no GPU.
    if args.quant != W4A16_QUANT:
        args.arch = args.arch or _detect_arch()
        args.cu = resolve_cu(args.arch, args.cu)
    if args.list_modes:
        args.arch = args.arch or "gfx936"
        return _list_modes(args)
    if args.bench_direct or args.direct_child or args.quant == W4A16_QUANT:
        if args.perf_only or args.skip_correctness:
            parser.error("independent GEMM tuning requires correctness before timing")
        if min(args.e, args.n, args.k, args.topk, *args.m) <= 0:
            parser.error("direct tuning requires positive E/N/K/topk/M")
        if args.warmup < 0 or args.iters < 1:
            parser.error("warmup must be nonnegative and iters must be positive")
    profile = PROFILES[args.quant]
    stages = {}
    if args.bench_direct and args.sweep_direct:
        try:
            stages = {
                stage: _stage_descriptors(profile, stage, selected, arch=args.arch,
                                          config_declared_only=args.sweep_config_modes)
                for stage, selected in ((1, args.gemm1_modes), (2, args.gemm2_modes))
            }
        except ValueError as exc:
            parser.error(str(exc))
    elif args.gemm1_modes is not None or args.gemm2_modes is not None:
        parser.error("--gemm1-modes and --gemm2-modes require --bench-direct --sweep-direct")
    if args.quant == W4A16_QUANT:
        return _tune_w4a16(args, parser, argv, stages)

    if args.direct_child:
        return _bench_direct_child(args)
    if args.bench_direct:
        return bench_direct(args, stages)
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
