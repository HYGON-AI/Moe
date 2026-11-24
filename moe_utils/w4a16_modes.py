# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""W4A16 template catalog shared by runtime, tuning and C++ table generation."""

from dataclasses import dataclass
from typing import Dict, Optional

from .channel_mode import mode


W4A16_MODE_SCHEMA = "w4a16"
W4A16_CSV_NAME = "tuned_moe_c_w4a16_group.csv"
W4A16_CSV_QUANT = "int4_w4a16_group"
W4A16_GROUP_SIZE = 32  # GROUP_K of the launchers


@dataclass(frozen=True)
class ModeSpec:
    stage: int
    family: int
    bm: int
    profile: int
    implementation: str
    template: Dict[str, int]
    unfused_profile: Optional[int] = None
    fp16_overrides: Optional[Dict[str, int]] = None
    prefill_tile: Optional["PrefillTile"] = None
    fp16_prefill_tile: Optional["PrefillTile"] = None


# Compiler and LDS settings belong to the pipeline, not to the tile entries.
# The mode table generator emits the same pipeline definitions for HIP.
@dataclass(frozen=True)
class PrefillPipeline:
    st: int = 2
    prefetch_a: bool = False
    wpe: int = 4
    scm: int = 4
    ah: int = 1
    opt: int = 0
    wide_n: bool = False  # four 16-column fragments per wave rather than two
    small_bm_wpe: Optional[int] = None


PREFILL_PIPELINES = {
    "DoubleBuffered": PrefillPipeline(prefetch_a=True, wpe=0),
    "StreamDoubleSwizzled": PrefillPipeline(prefetch_a=True, wpe=0, opt=4),
    "StreamDoubleWide": PrefillPipeline(wide_n=True),
    "StreamSingle": PrefillPipeline(st=1, prefetch_a=True, wpe=5, opt=4, small_bm_wpe=6),
    "StreamHalfK": PrefillPipeline(ah=2),
    "StreamHalfKRelaxed": PrefillPipeline(ah=2, wpe=2),
    "CompactSwizzled": PrefillPipeline(prefetch_a=True, wpe=0, opt=7),
    "ResidentA": PrefillPipeline(),
    "ResidentARelaxed": PrefillPipeline(wpe=3),
    "ResidentAMinWaves": PrefillPipeline(wpe=2),
    "ResidentAWide": PrefillPipeline(wide_n=True, scm=2),
    "ResidentAWideMinWaves": PrefillPipeline(wide_n=True, scm=2, wpe=2),
    "ResidentASwizzled": PrefillPipeline(opt=7),
    "ResidentASwizzledRelaxed": PrefillPipeline(wpe=3, opt=7),
}


@dataclass(frozen=True)
class PrefillTile:
    bn: int
    nloop: int
    pipeline: str

    def template(self, bm: int, silu: bool) -> Dict[str, int]:
        p = PREFILL_PIPELINES[self.pipeline]
        nn = 4 if p.wide_n else 2
        assert self.bn > 0 and self.bn % (16 * nn) == 0
        wpe = p.small_bm_wpe if bm == 32 and p.small_bm_wpe is not None else p.wpe
        return dict(warps=self.bn // (16 * nn), nn=nn, nch=self.nloop, st=p.st,
                    prefetch_a=p.prefetch_a, wpe=wpe, scm=p.scm, ah=p.ah,
                    opt=p.opt, silu=silu)


BSTAT_FALLBACK_TILE = PrefillTile(128, 1, "CompactSwizzled")
DECODE_BK = 128


def mode_specs():
    """Declare existing templates in terms of the new profiles."""
    modes = []

    # Legacy decode uses the 10v (ST2) and 19v (ST1) ranges for retuning.
    # Rows are indexed by (stage, BM, ST) and contain (BN, BK, WM, WN, WK).
    legacy = {
        (1, 16, 2): [(16, 512, 16, 16, 128), (32, 512, 16, 32, 128),
                     (32, 256, 16, 16, 128), (256, 128, 16, 64, 128), (128, 256, 16, 64, 128)],
        (1, 32, 2): [(128, 256, 32, 64, 128), (256, 128, 32, 64, 128)],
        (1, 48, 2): [(128, 128, 48, 32, 128), (256, 128, 48, 64, 128)],
        (1, 64, 2): [(128, 128, 64, 32, 128), (256, 128, 64, 64, 128)],
        (1, 128, 2): [(128, 128, 64, 64, 128)],
        (1, 128, 1): [(256, 64, 64, 64, 64), (128, 64, 64, 64, 64)],
        (2, 16, 2): [(64, 128, 16, 16, 128), (128, 128, 16, 32, 128), (256, 128, 16, 64, 128)],
        (2, 32, 2): [(256, 128, 32, 64, 128), (128, 256, 32, 64, 128)],
        (2, 48, 2): [(128, 128, 48, 32, 128), (256, 128, 48, 64, 128)],
        (2, 64, 2): [(128, 128, 64, 32, 128), (256, 128, 64, 64, 128)],
        (2, 128, 2): [(128, 128, 64, 64, 128)],
        (2, 128, 1): [(256, 64, 64, 64, 64), (128, 64, 64, 64, 64), (384, 64, 64, 64, 64)],
    }
    for (stage, bm, st), tiles in legacy.items():
        for variant, (bn, bk, wm, wn, wk) in enumerate(tiles, 1):
            profile = (100 if st == 2 else 190) + variant
            params = dict(bn=bn, bk=bk, wm=wm, wn=wn, wk=wk, st=st)
            modes.append(ModeSpec(stage, 0, bm, profile, "legacy_decode", params))

    # Decode 1xy / 2xy. x encodes the BN tile and y is the requested prefetch depth.
    # Keep the expanded wave layout in the descriptor for kernel inspection.
    layouts = {64: 1, 128: 2, 256: 3}
    for stage, bm, bn, depths in ((1, 16, 64, (2, 4, 8)), (1, 32, 128, (2,)),
                                 (2, 16, 128, (2,)), (2, 16, 256, (2,))):
        nn = 2
        wn = bn // (16 * nn)
        kw = 2 if bn == 64 else 1
        for pd in depths:
            profile = 100 + 10 * layouts[bn] + pd
            params = dict(nm=bm // 16, wn=wn, kw=kw, nn=nn, pd=pd, silu=False)
            modes.append(ModeSpec(stage, 0, bm, profile, "decode", params))
            if stage == 1:
                modes.append(ModeSpec(stage, 0, bm, profile + 100, "decode",
                                      dict(params, silu=True), profile))

    def prefill(stage, bm, profile, tile, *, silu=False, fp16=None):
        template = tile.template(bm, silu)
        overrides = {key: value for key, value in fp16.template(bm, silu).items()
                     if value != template[key]} if fp16 else None
        modes.append(ModeSpec(stage, 1, bm, profile, "prefill", template,
                              profile - 100 if silu else None, overrides or None, tile, fp16))

    # Streamed A: profile, BMs, BN, pipeline, has SiLU twin. NLoop is always 1.
    streamed = (
        (311, (48,), 128, "DoubleBuffered", True),
        (312, (64,), 256, "DoubleBuffered", False),
        (313, (64,), 512, "StreamDoubleWide", True),
        (321, (80, 128, 160, 192, 256), 256, "StreamHalfK", True),
        (322, (160, 192, 256), 256, "StreamHalfKRelaxed", True),
        (331, (48,), 128, "StreamDoubleSwizzled", True),
        (341, (48,), 128, "StreamSingle", True),
        (342, (32,), 128, "StreamSingle", True),
    )
    for profile, bms, bn, pipeline, twin in streamed:
        tile = PrefillTile(bn, 1, pipeline)
        for bm in bms:
            prefill(1, bm, profile, tile)
            if twin:
                prefill(1, bm, profile + 100, tile, silu=True)
    prefill(2, 256, 322, PrefillTile(256, 1, "StreamHalfKRelaxed"))

    # Resident A 5xy: x is the requested NLoop; y selects BN and the pipeline.
    # The optional FP16 tile keeps the existing layout/register-budget override.
    resident = {
        0: (128, "CompactSwizzled"),
        1: (256, "DoubleBuffered"),
        2: (512, "ResidentAWide"),
        3: (256, "ResidentA"),
        4: (256, "ResidentARelaxed"),
        5: (256, "ResidentAMinWaves"),
        6: (512, "ResidentAWideMinWaves"),
        7: (512, "ResidentA"),
        8: (256, "CompactSwizzled"),
        9: (256, "ResidentASwizzled"),
    }
    for bm, profiles in ((32, (540,)), (48, (541, 548)),
                         (64, (521, 522, 541, 542, 548, 583, 589)),
                         (80, (584,)), (128, (546, 547, 585))):
        for profile in profiles:
            nloop, variant = divmod(profile % 100, 10)
            bn, pipeline = resident[variant]
            fp16 = {
                2: PrefillTile(256, 4, "DoubleBuffered"),
                3: PrefillTile(bn, nloop, "ResidentARelaxed"),
                9: PrefillTile(bn, nloop, "ResidentASwizzledRelaxed"),
            }.get(variant)
            prefill(2, bm, profile, PrefillTile(bn, nloop, pipeline), fp16=fp16)

    modes.append(ModeSpec(2, 1, 32, 611, "bstat", dict(warps=16, nn=2, wpe=4, opt=6)))
    return modes


DECODE_K = {1: (1024, 2048, 3072, 4096, 6144, 7168), 2: (256, 512, 1024, 2048)}


def resolve_template(implementation, template, fp16_overrides, stage, dtype, out_n, reduce_k, *, output_rows=0):
    """Mirror dtype and shape branches of the target launchers for inspection."""
    params = dict(template)
    reasons = []
    if dtype == "fp16" and fp16_overrides:
        params.update(fp16_overrides)
        reasons.append("fp16 template override")
    if implementation == "bstat":
        fallback_reason = None
        if reduce_k and reduce_k != 256:
            fallback_reason = "B-stat requires K=256"
        elif output_rows * out_n * 2 >= (1 << 31) - 1:
            fallback_reason = "B-stat output exceeds the signed int32 byte limit"
        if fallback_reason:
            reasons.append(fallback_reason)
            implementation = "prefill"
            params = BSTAT_FALLBACK_TILE.template(32, False)
    if implementation == "prefill" and params["nch"] > 1:
        tile_n = params["warps"] * params["nn"] * 16 * params["nch"]
        if (reduce_k and reduce_k != 256) or (out_n and out_n % tile_n):
            params["nch"] = 1
            reasons.append("resident A requires K=256 and N divisible by {}".format(tile_n))
    if implementation == "decode" and reduce_k in DECODE_K[stage]:
        params["spw"] = reduce_k // (DECODE_BK * params["kw"])
        params["pd"] = min(params["pd"], params["spw"])
    return implementation, params, reasons


W4A16_SPECS = {mode(s.stage, 1, s.family, s.bm, s.profile): s for s in mode_specs()}
W4A16_MODES_BY_STAGE = {
    stage: tuple(value for value, spec in W4A16_SPECS.items() if spec.stage == stage)
    for stage in (1, 2)
}


def w4a16_spec(mode_id: int) -> ModeSpec:
    try:
        return W4A16_SPECS[int(mode_id)]
    except KeyError:
        raise ValueError(f"Unknown {W4A16_MODE_SCHEMA} mode: {mode_id}") from None


def w4a16_unfused_mode(mode_id: int) -> Optional[int]:
    spec = w4a16_spec(mode_id)
    if spec.unfused_profile is None:
        return None
    return mode(spec.stage, 1, spec.family, spec.bm, spec.unfused_profile)
