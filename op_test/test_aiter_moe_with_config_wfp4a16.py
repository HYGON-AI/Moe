# Test for AITER moe_c WFP4A16.

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_utils.fused_moe_c import moe_align_block_size
from moe_utils.golden import run_wfp4a16_triton_v2_golden
from moe_utils.moe import (
    MoeQuantType,
    MoeSolutionType,
    aiter_moe,
    aiter_moe_shfl_scale,
    get_aiter_moe_config,
)
from moe_utils.utils import checkAllclose, dtypes, fused_topk, get_gfx, logger, perftest
    

torch.set_default_device("cuda")

_FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device="cuda",
    dtype=torch.float32,
)


def _pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 1::2] << 4).bitwise_or(codes[..., ::2]).contiguous()


def _make_wfp4a16_weight(
    shape: Tuple[int, int],
    group_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = shape
    codes = torch.randint(
        0, 16, (out_features, in_features), device="cuda", dtype=torch.uint8)
    scales_u8 = torch.randint(
        121, 126, (out_features, in_features // group_size),
        device="cuda", dtype=torch.uint8)
    return _pack_fp4_codes(codes), scales_u8.contiguous()


def prepare_wfp4a16_inputs(
    m: int,
    k: int,
    n: int,
    e: int,
    topk: int,
    group_size: int,
    dtype: torch.dtype,
):
    """Prepare WFP4A16 MXFP4 groupwise inputs.

    Weight scale shuffle is intentionally handled by the test flow after
    get_aiter_moe_config, matching the w8a8_channelwise CI structure.
    """
    input_tensor = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    w1_qweight = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size), device="cuda", dtype=torch.uint8)
    w2_scales = torch.empty((e, k, n // group_size), device="cuda", dtype=torch.uint8)

    for expert_id in range(e):
        qweight, scales = _make_wfp4a16_weight((2 * n, k), group_size, dtype)
        w1_qweight[expert_id] = qweight
        w1_scales[expert_id] = scales

        qweight, scales = _make_wfp4a16_weight((k, n), group_size, dtype)
        w2_qweight[expert_id] = qweight
        w2_scales[expert_id] = scales

    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, True)
    return {
        "input": input_tensor,
        "w1_qweight": w1_qweight,
        "w2_qweight": w2_qweight,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "w1_qzeros": None,
        "w2_qzeros": None,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


@perftest(num_warmup=1, num_iters=2)
def _run_triton_ref(
    hidden_states,
    w1_qweight,
    w2_qweight,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    config,
    down_config,
    group_size,
    activation,
):
    return run_wfp4a16_triton_v2_golden(
        hidden_states,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        down_config,
        group_size=group_size,
        output_dtype=hidden_states.dtype,
        activation=activation,
        routed_scaling_factor=1.0,
        gemm1_alpha=4.0 if activation == "situ" else None,
        gemm1_limit=25.0 if activation == "situ" else None,
        apply_router_weight_on_input=False,
        no_combine=False,
    )


@perftest(num_warmup=1, num_iters=2, num_rotate_args=1, testGraph=False)
def _run_aiter_moe_perf(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    moe_config,
    inplace,
    activation,
    w1_scale,
    w2_scale,
    w1_zp,
    w2_zp,
    a1_scale,
    a2_scale,
    block_shape,
    global_num_experts,
    expert_map,
    out_dtype,
    use_shuffle,
):
    mortal_input = hidden_states.clone() if inplace else hidden_states
    return aiter_moe(
        mortal_input,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_config,
        inplace,
        activation,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        global_num_experts,
        expert_map,
        routed_scaling_factor=1.0,
        use_weight_shuffle=use_shuffle,
        output_dtype=out_dtype,
    )


def test_get_config(m, k, n, e, topk, group_size, dtype):
    """Test get_aiter_moe_config for WFP4A16 / fp4_w4a16."""
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=group_size,
        dtype=dtype,
        quant_type=MoeQuantType.WFP4A16,
    )

    tag = "get_config_fp4_w4a16"
    if status:
        assert moe_cfg.quant_type == MoeQuantType.WFP4A16
        assert moe_cfg.solution_type == MoeSolutionType.MOE_C
        assert moe_cfg.config is not None
        logger.info(
            f"[{tag}] {m=}, solution={moe_cfg.solution_type}, "
            f"config keys={list(moe_cfg.config.keys())}"
        )
    else:
        assert moe_cfg.solution_type is None
        assert moe_cfg.config is None
        logger.info(f"[{tag}] {m=}, no solution found")

    return status, moe_cfg


def test_aiter_moe_wfp4a16(m, k, n, e, topk, group_size, in_dtype, out_dtype, inplace, activation):
    """End-to-end test of aiter_moe with WFP4A16 MXFP4 groupwise weights."""
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=group_size,
        dtype=in_dtype,
        quant_type=MoeQuantType.WFP4A16,
    )
    tag = "aiter_moe_wfp4a16"
    if not status:
        logger.info(
            f"[{tag}] SKIP {m=}, N={n}, K={k}, E={e}, topk={topk}: no backend")
        return None

    data = prepare_wfp4a16_inputs(m, k, n, e, topk, group_size, in_dtype)
    w1_scale, w2_scale = data["w1_scales"], data["w2_scales"]
    if moe_cfg.need_shuffle_scale:
        w1_scale, w2_scale = aiter_moe_shfl_scale(w1_scale, w2_scale, moe_cfg)

    block_shape = [0, group_size]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        data["topk_ids"],
        moe_cfg.config["BLOCK_SIZE_M"],
        e,
        None,
    )
    triton_cfg = {
        "BLOCK_SIZE_M": moe_cfg.config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
    }
    ref_out, triton_golden_us = _run_triton_ref(
        data["input"], data["w1_qweight"], data["w2_qweight"],
        data["topk_weights"], data["topk_ids"],
        w1_scale, w2_scale,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        triton_cfg, None, group_size, activation)

    aiter_out, aiter_us = _run_aiter_moe_perf(
        data["input"],
        data["w1_qweight"],
        data["w2_qweight"],
        data["topk_weights"],
        data["topk_ids"],
        moe_cfg,
        inplace,
        activation,
        w1_scale,
        w2_scale,
        data["w1_qzeros"],
        data["w2_qzeros"],
        None,
        None,
        block_shape,
        e,
        None,
        out_dtype,
        use_shuffle=0,
    )

    msg = f"[{tag}] {m=}, N={n}, K={k}, E={e}, topk={topk}, activation={activation}, backend={moe_cfg.solution_type}"
    assert torch.isfinite(aiter_out).all(), (
        "Non-finite output in test_aiter_moe_wfp4a16: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {group_size=}, "
        f"{in_dtype=}, {out_dtype=}, backend={moe_cfg.solution_type}"
    )
    check_ret = checkAllclose(ref_out, aiter_out, rtol=0.01, atol=0.125, msg=msg)
    assert check_ret <= 0.03, (
        "Accuracy check failed in test_aiter_moe_wfp4a16: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {group_size=}, "
        f"{in_dtype=}, {out_dtype=}, {inplace=}, {activation=}, "
        f"backend={moe_cfg.solution_type}, error_ratio={check_ret:.6f}, tolerance=0.03"
    )
    return {
        "m": m,
        "N1": 2 * n,
        "N2": k,
        "K": k,
        "e": e,
        "topk": topk,
        "activation": activation,
        "quant_type": MoeQuantType.WFP4A16,
        "backend": moe_cfg.solution_type,
        "sgl_triton_v2_us": triton_golden_us,
        "us": aiter_us,
        "speedup": triton_golden_us / aiter_us if aiter_us else None,
        "accuracy": "passed" if check_ret == 0 else (1 - check_ret),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test aiter_moe with fp4_w4a16 / WFP4A16 quantization",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--m", type=int, nargs="+", default=[ 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192])
    # parser.add_argument("--m", type=int, nargs="+", default=[ 46])
    parser.add_argument("--k", type=int, default=3584)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--e", type=int, default=896)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--activation", type=str, default="situ")
    parser.add_argument("--inplace", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    failed_cases = []

    arch = get_gfx().lower()
    if arch in {"gfx920", "gfx936"}:
        logger.info(
            f"SKIP fp4_w4a16 on {arch}: operator support is TODO"
        )
        raise SystemExit(0)

    def run_accuracy_case(test_func, *args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except AssertionError as exc:
            failed_cases.append(str(exc))
            print(f"[ACCURACY FAILED] {exc}", flush=True)
            return None
    args = parse_args()

    arch = get_gfx().lower()
    if arch in {"gfx920", "gfx936"}:
        logger.info(
            f"SKIP fp4_w4a16 on {arch}: operator support is TODO"
        )
        raise SystemExit(0)

    in_dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16
    out_dtype = in_dtype

    logger.info("=" * 60)
    logger.info("Part 1: Testing get_aiter_moe_config for fp4_w4a16")
    logger.info("=" * 60)
    for m in args.m:
        test_get_config(m, args.k, args.n, args.e, args.topk, args.group_size, in_dtype)

    logger.info("=" * 60)
    logger.info("Part 2: Testing aiter_moe end-to-end for fp4_w4a16")
    logger.info("=" * 60)
    rows = []
    for m in args.m:
        ret = run_accuracy_case(
            test_aiter_moe_wfp4a16,
            m, args.k, args.n, args.e, args.topk, args.group_size,
            in_dtype, out_dtype, args.inplace, args.activation,
        )
        if ret is not None:
            rows.append(ret)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("wfp4a16.csv", index=False)
        logger.info(f"summary:\n{df}")

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
