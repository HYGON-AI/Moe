# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Test for get_aiter_moe_config_w4a16 and aiter_moe_w4a16

import argparse
import torch
import sys
from pathlib import Path

import pandas as pd
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_utils.utils import fused_topk, torch_moe
from moe_utils.utils import dtypes
from moe_utils.utils import checkAllclose, logger, perftest, quantize_weights, scalar_types
from moe_utils.moe import (
    get_aiter_moe_config,
    aiter_moe,
    aiter_moe_shfl_scale,
    MoeSolutionType,
    MoeQuantType,
    AiterMoeConfig,
)

torch.set_default_device("cuda")

# Solution-type strings are shared by moe_utils and aiter.moe.
SOL_MOE_C = MoeSolutionType.MOE_C  # "moe_c"
SOL_ASM = "asm"
SOL_TRITON = "triton"

# --backend value -> spec_sol_type passed to get_aiter_moe_config (None = auto).
BACKEND_CHOICES = {
    "moe-c": SOL_MOE_C,
    "asm": SOL_ASM,
    "triton": SOL_TRITON,
    "auto": None,
}


class _MoeApi:
    """The get-config / run / scale-shuffle entry points of one implementation."""

    def __init__(self, get_config, run, shfl_scale):
        self.get_aiter_moe_config = get_config
        self.aiter_moe = run
        self.aiter_moe_shfl_scale = shfl_scale


def _moe_api(backend):
    """moe_c runs through the standalone ``moe_utils``; ASM / Triton (and auto,
    which may resolve to either) only exist in the ``aiter`` package, so it is
    imported lazily and the default moe-c run does not depend on it."""
    if backend == SOL_MOE_C:
        return _MoeApi(get_aiter_moe_config, aiter_moe, aiter_moe_shfl_scale)
    try:
        from aiter import moe as aiter_moe_mod
    except ImportError as exc:
        raise RuntimeError(
            f"backend={backend or 'auto'} needs the aiter package (aiter.moe), "
            f"which could not be imported: {exc}") from exc
    return _MoeApi(aiter_moe_mod.get_aiter_moe_config, aiter_moe_mod.aiter_moe,
                   aiter_moe_mod.aiter_moe_shfl_scale)


def _unsupported_reason(backend, dtype, has_zp):
    """Combinations whose config lookup succeeds but whose kernel cannot run them."""
    if backend == SOL_MOE_C and has_zp:
        # The moe_c W4A16 kernels hard-code the zero point to 8 and ignore qzeros.
        return "moe_c only supports symmetric weights (no zero points)"
    if backend == SOL_ASM and dtype != torch.bfloat16:
        # asm_fmoe_stage1 raises "unsupport current input type" for fp16.
        return "asm W4A16 only supports bf16 activations"
    return None


# ---------------------------------------------------------------------------
# Weight quantization helpers (adapted from test_moe_wna16.py)
# ---------------------------------------------------------------------------

def _quantize_w4a16_weights(w_fp, group_size, has_zp, pack_for_backend):
    """Quantize a single expert weight matrix to int4.

    Args:
        w_fp: Floating-point weight ``[out_features, in_features]``.
        group_size: Quantization group size along K.
        has_zp: Whether to produce zero-points.
        pack_for_backend: ``"moe_c"`` / ``"asm"`` / ``"triton"``; decides the
            qzero packing: ASM packs two groups per byte along K/group,
            moe_c and Triton pack two output rows per byte along N.

    Returns:
        (weight_ref, qweight, scales, qzeros_or_None)
    """
    quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    weight, qweight, scales, qzeros = quantize_weights(
        w_fp.T, quant_type, group_size, has_zp, False)
    weight = weight.T
    qweight = qweight.T.contiguous().to(torch.uint8)
    scales = scales.T

    if has_zp:
        qzeros = qzeros.T.contiguous().to(torch.uint8)

    qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
    if has_zp:
        if pack_for_backend == SOL_ASM:
            qzeros = qzeros[:, 1::2] * 16 + qzeros[:, ::2]
        elif pack_for_backend in (SOL_MOE_C, SOL_TRITON):
            qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]
        else:
            raise ValueError(f"unsupported W4A16 zero-point backend: {pack_for_backend}")

    return weight, qweight, scales, qzeros if has_zp else None


def prepare_w4a16_inputs(m, k, n, e, topk, group_size, has_zp, dtype,
                         backend,
                         config: AiterMoeConfig):
    """Build all tensors needed to run a w4a16 MOE test.

    Returns a dict of tensors keyed by name.
    """
    pack_factor = 2  # int4

    input_tensor = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1_fp = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2_fp = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    # Allocate packed weight storage
    w1_qweight = torch.empty((e, 2 * n, k // pack_factor), device="cuda",
                             dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // pack_factor), device="cuda",
                             dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size), device="cuda",
                            dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size), device="cuda",
                            dtype=dtype)

    # ASM packs qzeros along K/group, moe_c / Triton along N.
    if backend == SOL_ASM:
        w1_qzeros = torch.empty(
            (e, 2 * n, k // group_size // pack_factor), device="cuda",
            dtype=torch.uint8)
        w2_qzeros = torch.empty(
            (e, k, n // group_size // pack_factor), device="cuda",
            dtype=torch.uint8)
    else:
        w1_qzeros = torch.empty(
            (e, 2 * n // pack_factor, k // group_size), device="cuda",
            dtype=torch.uint8)
        w2_qzeros = torch.empty(
            (e, k // pack_factor, n // group_size), device="cuda",
            dtype=torch.uint8)
    if not has_zp:
        # The native entry points require a qzero tensor even for symmetric
        # W4A16.  Encode the midpoint (8) in both nibbles.
        w1_qzeros.fill_(0x88)
        w2_qzeros.fill_(0x88)

    w1_ref = w1_fp.clone()
    w2_ref = w2_fp.clone()

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w_fp_e, w_ref, w_qw, w_sc, w_zp = (
                w1_fp, w1_ref, w1_qweight, w1_scales, w1_qzeros)
        else:
            w_fp_e, w_ref, w_qw, w_sc, w_zp = (
                w2_fp, w2_ref, w2_qweight, w2_scales, w2_qzeros)
        weight, qweight, scales, qzeros = _quantize_w4a16_weights(
            w_fp_e[expert_id], group_size, has_zp, backend)
        w_ref[expert_id] = weight
        w_qw[expert_id] = qweight
        w_sc[expert_id] = scales
        if has_zp and w_zp is not None:
            w_zp[expert_id] = qzeros

    # Both the BW reference test and the active MOE_C MODE=4/2 kernel use
    # canonical packed qweights.  Only the scale layout is backend-configured.
    w1_qweight_final = w1_qweight
    w2_qweight_final = w2_qweight

    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, True)

    return {
        "input": input_tensor,
        "w1_ref": w1_ref,
        "w2_ref": w2_ref,
        "w1_qweight": w1_qweight_final,
        "w2_qweight": w2_qweight_final,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "w1_qzeros": w1_qzeros,
        "w2_qzeros": w2_qzeros,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Test: get_aiter_moe_config (w4a16)
# ---------------------------------------------------------------------------

def test_get_config(m, k, n, e, topk, group_size, dtype, backend=SOL_MOE_C):
    """Test that the W4A16 config of ``backend`` (None = auto) is available."""
    N1 = 2 * n  # gate + up
    N2 = k      # down / hidden_size
    K = k       # model dimension (uncompressed)

    status, moe_cfg = _moe_api(backend).get_aiter_moe_config(
        M=m, E=e, N1=N1, N2=N2, K=K,
        top_k=topk, block_size=group_size, dtype=dtype,
        quant_type=MoeQuantType.W4A16,
        spec_sol_type=backend,
    )

    if status:
        if backend is not None:
            assert moe_cfg.solution_type == backend, (
                f"requested backend {backend}, "
                f"got {moe_cfg.solution_type}"
            )
        assert moe_cfg.config is not None, \
            "status=True but config is None"
        assert moe_cfg.quant_type == MoeQuantType.W4A16
        logger.info(
            f"[get_config_w4a16] {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
            f"solution={moe_cfg.solution_type}, "
            f"config keys={list(moe_cfg.config.keys())}"
        )
    else:
        assert moe_cfg.solution_type is None, \
            "status=False but solution_type is not None"
        assert moe_cfg.config is None, \
            "status=False but config is not None"
        logger.info(
            f"[get_config_w4a16] {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
            f"no solution found (expected on unsupported configs)"
        )

    return status, moe_cfg


# ---------------------------------------------------------------------------
# Test: aiter_moe end-to-end for w4a16
# ---------------------------------------------------------------------------

@perftest(num_warmup=1, num_iters=2)
def _run_torch_ref(hidden_states, w1, w2, topk_weights, topk_ids):
    return torch_moe(hidden_states, w1, w2, topk_weights, topk_ids)


@perftest(num_warmup=10, num_iters=100, num_rotate_args=1)
def _run_aiter_moe_perf(hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_config,
        inplace,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        global_num_experts,
        expert_map,
        routed_scaling_factor,
        activation,
        moe_fn=None):

        if inplace:
            mortal_input = hidden_states.clone()  # 保证inplace操作的正确性
        else:
            mortal_input = hidden_states

        return (moe_fn or aiter_moe)(mortal_input, w1, w2, topk_weights, topk_ids, moe_config, inplace, activation, w1_scale, w2_scale, w1_zp, w2_zp,
                     a1_scale, a2_scale, block_shape, global_num_experts, expert_map, routed_scaling_factor, output_dtype=hidden_states.dtype)


def test_aiter_moe_w4a16(m, k, n, e, topk, group_size, has_zp, dtype, inplace, routed_scaling_factor,
                         backend=SOL_MOE_C):
    """End-to-end: get config -> run aiter_moe -> compare with torch
    reference.  ``backend`` is moe_c / asm / triton, or None for auto."""
    N1 = 2 * n
    N2 = k
    K = k

    api = _moe_api(backend)
    status, moe_cfg = api.get_aiter_moe_config(
        M=m, E=e, N1=N1, N2=N2, K=K,
        top_k=topk, block_size=group_size, dtype=dtype,
        quant_type=MoeQuantType.W4A16,
        spec_sol_type=backend,
    )

    if not status:
        logger.info(
            f"[aiter_moe_w4a16] SKIP {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}: "
            f"no {backend or 'auto'} backend available"
        )
        return None

    if backend is not None:
        assert moe_cfg.solution_type == backend, (
            f"requested backend {backend}, got {moe_cfg.solution_type}"
        )
    backend = moe_cfg.solution_type
    reason = _unsupported_reason(backend, dtype, has_zp)
    if reason:
        logger.info(f"[aiter_moe_w4a16] SKIP {m=}, backend={backend}: {reason}")
        return None
    # The weights below are prepared in the canonical (unshuffled) layout.
    assert not moe_cfg.need_shuffle, (
        f"backend={backend} config requires shuffled weights "
        f"(need_shuffle=True), which this test does not prepare: {moe_cfg.config}"
    )

    data = prepare_w4a16_inputs(
        m, k, n, e, topk, group_size, has_zp, dtype, backend, moe_cfg)

    logger.info(
        f"[aiter_moe_w4a16] {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
        f"backend={backend}, config={moe_cfg.config}, "
        f"need_shuffle={moe_cfg.need_shuffle}, "
        f"need_shuffle_scale={moe_cfg.need_shuffle_scale}, "
        f"block_shape={[0, group_size]}"
    )

    if moe_cfg.need_shuffle_scale:
        data["w1_scales"], data["w2_scales"] = api.aiter_moe_shfl_scale(
            data["w1_scales"], data["w2_scales"], moe_cfg
        )

    # Torch reference
    ref_out, _ = _run_torch_ref(
        data["input"], data["w1_ref"], data["w2_ref"],
        data["topk_weights"], data["topk_ids"],
    )

    # generic aiter_moe dispatch with w4a16 config
    block_shape = [0, group_size] if group_size else None
    aiter_us = 1.0
    # aiter_out = aiter_moe(
    aiter_out, aiter_us = _run_aiter_moe_perf(
        hidden_states=data["input"],
        w1=data["w1_qweight"],
        w2=data["w2_qweight"],
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        moe_config=moe_cfg,
        inplace=inplace,
        activation="silu",
        w1_scale=data["w1_scales"],
        w2_scale=data["w2_scales"],
        w1_zp=data["w1_qzeros"],
        w2_zp=data["w2_qzeros"],
        a1_scale=None,
        a2_scale=None,
        block_shape=block_shape,
        global_num_experts=e,
        expert_map=None,
        routed_scaling_factor=routed_scaling_factor,
        moe_fn=api.aiter_moe,
    )

    msg = (f"[aiter_moe_w4a16] {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
           f"backend={backend}")
    assert torch.isfinite(aiter_out).all(), (
        "Non-finite output in test_aiter_moe_w4a16: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {group_size=}, "
        f"{dtype=}, backend={backend}"
    )
    check_ret = checkAllclose(ref_out.to(aiter_out.dtype), aiter_out, rtol=0.01, atol=0.01, msg=msg)
    assert check_ret <= 0.03, (
        "Accuracy check failed in test_aiter_moe_w4a16: "
        f"{m=}, {k=}, {n=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
        f"{group_size=}, {has_zp=}, {dtype=}, {inplace=}, "
        f"{routed_scaling_factor=}, backend={backend}, "
        f"error_ratio={check_ret:.6f}, tolerance=0.03"
    )
    ret_output = "passed" if check_ret == 0 else (1-check_ret)
    return {"m": m, "N1": N1, "N2": N2, "K": K, "e":e, "topk":topk,"backend": backend, "us": aiter_us, "accuracy": ret_output}

# ---------------------------------------------------------------------------
# Main: run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failed_cases = []

    def run_accuracy_case(test_func, *args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except AssertionError as exc:
            failed_cases.append(str(exc))
            print(f"[ACCURACY FAILED] {exc}", flush=True)
            return None

    # Shapes that get a config and pass accuracy (gfx938, group_size 32, measured
    # with M=1/64).  Unsupported shapes are skipped; a run where nothing ran exits non-zero.
    #   moe-c : config keyed by (E, N) only: E=256 with N=256, or E=384 with
    #           N in {256, 512, 1024, 2048}.  K and topk are free (K=2048/4096/7168,
    #           topk=4/8 verified).  bf16 and fp16.  Symmetric weights only.
    #   asm   : N=256, K=7168, topk<=8 (gw32 kernel), any E (128/256/384 verified).
    #           bf16 only.  Zero points supported.
    #   triton: installed configs are E=384, topk=8, N in {256, 512, 1024, 2048},
    #           K>=2048 and a multiple of 1024 (KMin=2048,KMod=1024).  bf16 and
    #           fp16.  Zero points supported.
    #   auto  : aiter picks per dtype (bf16: asm > triton > moe_c; fp16: moe_c > triton).
    parser = argparse.ArgumentParser(description="Test aiter_moe with group-wise w4a16 quantization")
    parser.add_argument("--backend", choices=tuple(BACKEND_CHOICES), default="moe-c",
                        help="moe-c: standalone moe_utils (default); asm / triton: aiter "
                             "package; auto: aiter picks by priority")
    parser.add_argument("--e", type=int, default=256)
    parser.add_argument("--n", type=int, default=256, help="intermediate size")
    parser.add_argument("--k", type=int, default=4096, help="model dim")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", "--m", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--has_zp", action="store_true", default=False)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--output", default=None,
                        help="CSV path (default test_aiter_moe_with_config_w4a16[_<backend>].csv)")
    args = parser.parse_args()

    backend = BACKEND_CHOICES[args.backend]
    dtype = {"bf16": dtypes.bf16, "fp16": dtypes.fp16}[args.dtype]
    group_size, has_zp = args.group_size, args.has_zp
    e, topk, k, n = args.e, args.topk, args.k, args.n
    inplace = True
    routed_scaling_factor = 1.0
    output = args.output or ("test_aiter_moe_with_config_w4a16.csv" if backend == SOL_MOE_C
                             else f"test_aiter_moe_with_config_w4a16_{args.backend}.csv")

    # --- Part 1: test get_aiter_moe_config (w4a16) ---
    logger.info("=" * 60)
    logger.info(f"Part 1: Testing get_aiter_moe_config for w4a16, backend={args.backend}")
    logger.info("=" * 60)

    test_tokens = args.tokens
    for m in test_tokens:
        test_get_config(m, k, n, e, topk, group_size, dtype, backend)

    # --- Part 2: test aiter_moe end-to-end (w4a16) ---
    logger.info("=" * 60)
    logger.info(f"Part 2: Testing aiter_moe end-to-end for w4a16, backend={args.backend}")
    logger.info("=" * 60)

    rows = []
    for m in test_tokens:
        ret = run_accuracy_case(
            test_aiter_moe_w4a16,
            m, k, n, e, topk, group_size, has_zp, dtype, inplace,
            routed_scaling_factor, backend,
        )
        if ret is not None:
            rows.append(ret)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output, index=False)
        logger.info(f"summary:\n{df}")

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
    if not rows:
        # An explicitly requested backend that never ran is a failure, not a pass.
        sys.exit(f"no case ran for backend={args.backend} ({e=}, {n=}, {k=}, {topk=}, "
                 f"group_size={group_size}, dtype={args.dtype}); see the supported "
                 f"shapes at the top of __main__")
