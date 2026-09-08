# Test for AITER moe_c WFP4A8 groupwise MXFP4.

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from moe_utils.utils import fused_topk
    from moe_utils.fused_moe_c import moe_align_block_size, get_moe_configs_marlin
    from moe_utils.golden import run_wfp4a8_groupwise_triton_golden
    from moe_utils.moe import (
        MoeQuantType,
        get_aiter_moe_config,
        aiter_moe_shfl_weight,
        aiter_moe,
    )
    from moe_utils.utils import checkAllclose, logger, perftest
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from moe_utils.utils import fused_topk
    from moe_utils.fused_moe_c import moe_align_block_size, get_moe_configs_marlin
    from moe_utils.golden import run_wfp4a8_groupwise_triton_golden
    from moe_utils.moe import (
        MoeQuantType,
        get_aiter_moe_config,
        aiter_moe_shfl_weight,
        aiter_moe,
    )
    from moe_utils.utils import checkAllclose, logger, perftest


torch.set_default_device("cuda")


def _pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    return ((codes[..., 1::2] << 4) | codes[..., ::2]).contiguous()


def _make_fp4_weight(shape: Tuple[int, int]) -> torch.Tensor:
    out_features, in_features = shape
    codes = torch.randint(
        0, 16, (out_features, in_features), device="cuda", dtype=torch.uint8
    )
    return _pack_fp4_codes(codes)


def prepare_wfp4a8_groupwise_inputs(
    m: int,
    k: int,
    n: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    group_size: int = 32,
):
    torch.manual_seed(20260813 + m)
    hidden_states = (torch.randn((m, k), device="cuda", dtype=torch.float32) / 10).to(dtype)
    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(hidden_states, score, topk, True)

    w1_qweight = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    for expert_id in range(e):
        w1_qweight[expert_id] = _make_fp4_weight((2 * n, k))
        w2_qweight[expert_id] = _make_fp4_weight((k, n))

    w1_scales = torch.randint(
        121, 126, (e, 2 * n, k // group_size), device="cuda", dtype=torch.uint8
    )
    w2_scales = torch.randint(
        121, 126, (e, k, n // group_size), device="cuda", dtype=torch.uint8
    )

    return {
        "input": hidden_states,
        "w1_qweight": w1_qweight.contiguous(),
        "w2_qweight": w2_qweight.contiguous(),
        "w1_scales": w1_scales.contiguous(),
        "w2_scales": w2_scales.contiguous(),
        "topk_weights": topk_weights.contiguous(),
        "topk_ids": topk_ids.contiguous(),
    }


@perftest(num_warmup=1, num_iters=2, testGraph=False)
def _run_triton_golden_perf(*args, **kwargs):
    return run_wfp4a8_groupwise_triton_golden(*args, **kwargs)


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
        1.0,
        output_dtype=out_dtype,
    )


def test_get_config(
    m: int,
    k: int,
    n: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    group_size: int,
):
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=group_size,
        dtype=dtype,
        quant_type=MoeQuantType.WFP4A8,
    )

    tag = f"get_config_wfp4a8_gw{group_size}"
    if status:
        assert moe_cfg.quant_type == MoeQuantType.WFP4A8
        assert moe_cfg.config is not None
        logger.info(
            f"[{tag}] {m=}, solution={moe_cfg.solution_type}, "
            f"need_shuffle={moe_cfg.need_shuffle}, config={moe_cfg.config}"
        )
    else:
        assert moe_cfg.config is None
        logger.info(f"[{tag}] {m=}, no solution found")
    return status, moe_cfg


def _groupwise_down_config(
    m: int,
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    topk: int,
    group_size: int,
    mode2: int | None = None,
):
    configs = get_moe_configs_marlin(
        E=w2_shape[0],
        N=w2_shape[2] * 2,
        dtype="fp4_w4a8",
        block_n=0,
        block_k=group_size,
        is_bottom=True,
        use_moe_wna16_cuda=True,
        K=w2_shape[1],
    )
    if configs is None:
        raise RuntimeError(f"No WFP4A8 groupwise GEMM2 config for m={m}")
    key_selected = min(configs.keys(), key=lambda x: abs(x - m))
    down_config = dict(configs[key_selected])
    down_config["key_selected"] = key_selected
    if mode2 is not None:
        down_config = dict(down_config)
        down_config["MODE"] = mode2
    return down_config


def test_aiter_moe_wfp4a8_groupwise(
    m, k, n, e, topk, group_size, dtype, activation, mode1=None, mode2=None, a_quant="per_token"
):
    block_size = [0, group_size]
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=group_size,
        dtype=dtype,
        quant_type=MoeQuantType.WFP4A8,
    )

    tag = f"aiter_moe_wfp4a8_gw{group_size}"
    if not status:
        logger.info(f"[{tag}] SKIP {m=}: no backend available")
        return None
    if mode1 is not None:
        config = dict(moe_cfg.config)
        config["MODE"] = mode1
        moe_cfg.config = config

    data = prepare_wfp4a8_groupwise_inputs(m, k, n, e, topk, dtype, group_size)
    config = moe_cfg.config
    down_config = _groupwise_down_config(
        m,
        tuple(data["w1_qweight"].shape),
        tuple(data["w2_qweight"].shape),
        topk,
        group_size,
        mode2,
    )
    w1_input, w2_input = data["w1_qweight"], data["w2_qweight"]
    if moe_cfg.need_shuffle:
        w1_input, w2_input = aiter_moe_shfl_weight(
            data["w1_qweight"], data["w2_qweight"], moe_cfg, block_shape=block_size
        )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        data["topk_ids"], config["BLOCK_SIZE_M"], e, None
    )
    triton_cfg = {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
    }
    triton_down_cfg = {
        "BLOCK_SIZE_M": down_config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
    }
    golden_out, golden_us = _run_triton_golden_perf(
        data["input"],
        data["w1_qweight"],
        data["w2_qweight"],
        data["topk_weights"],
        data["topk_ids"],
        data["w1_scales"],
        data["w2_scales"],
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        triton_cfg,
        triton_down_cfg,
        group_size=group_size,
        output_dtype=dtype,
        activation=activation,
        gemm1_alpha=4.0 if activation == "situ" else None,
        gemm1_limit=25.0 if activation == "situ" else None,
        a_quant=a_quant,
    )
    old_a_quant = os.environ.get("AITER_WFP4A8_A_QUANT")
    if a_quant == "per_token_group":
        os.environ["AITER_WFP4A8_A_QUANT"] = "1"
    else:
        os.environ.pop("AITER_WFP4A8_A_QUANT", None)
    try:
        aiter_out, aiter_us = _run_aiter_moe_perf(
            hidden_states=data["input"],
            w1=w1_input,
            w2=w2_input,
            topk_weights=data["topk_weights"],
            topk_ids=data["topk_ids"],
            moe_config=moe_cfg,
            inplace=False,
            activation=activation,
            w1_scale=data["w1_scales"],
            w2_scale=data["w2_scales"],
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=block_size,
            global_num_experts=e,
            expert_map=None,
            out_dtype=dtype,
        )
    finally:
        if old_a_quant is None:
            os.environ.pop("AITER_WFP4A8_A_QUANT", None)
        else:
            os.environ["AITER_WFP4A8_A_QUANT"] = old_a_quant
    assert torch.isfinite(aiter_out).all(), (
        "Non-finite output in test_aiter_moe_wfp4a8_groupwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {group_size=}, {dtype=}, "
        f"{activation=}, {a_quant=}, backend={moe_cfg.solution_type}"
    )
    accuracy = checkAllclose(golden_out, aiter_out, atol=0.2, rtol=0.1, msg=f"m={m}")
    assert accuracy <= 0.03, (
        "Accuracy check failed in test_aiter_moe_wfp4a8_groupwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {group_size=}, {dtype=}, "
        f"{activation=}, {a_quant=}, mode1={config['MODE']}, "
        f"mode2={down_config['MODE']}, backend={moe_cfg.solution_type}, "
        f"error_ratio={accuracy:.6f}, tolerance=0.03"
    )
    return {
        "m": m,
        "mode1": config["MODE"],
        "mode2": down_config["MODE"],
        "golden_us": golden_us,
        "moe_c_us": aiter_us,
        "speedup": golden_us / aiter_us if aiter_us else float("nan"),
        "correctness": "passed" if accuracy <= 0.03 else "failed",
    }


def main():
    failed_cases = []

    def run_accuracy_case(test_func, *args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except AssertionError as exc:
            failed_cases.append(str(exc))
            print(f"[ACCURACY FAILED] {exc}", flush=True)
            return None
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", nargs="+", type=int, default=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384])
    parser.add_argument("--k", type=int, default=3584)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--e", type=int, default=896)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--activation", type=str, default="situ")
    parser.add_argument("--mode1", type=int, default=None)
    parser.add_argument("--mode2", type=int, default=None)
    parser.add_argument("--a-quant", choices=("per_token", "per_token_group"), default="per_token")
    args = parser.parse_args()

    rows = []
    logger.info("=" * 60)
    logger.info("Part 1: Testing get_aiter_moe_config for WFP4A8 groupwise")
    logger.info("=" * 60)
    for m in args.m:
        test_get_config(
            m,
            args.k,
            args.n,
            args.e,
            args.topk,
            torch.bfloat16,
            args.group_size,
        )

    logger.info("=" * 60)
    logger.info("Part 2: Testing aiter_moe end-to-end for WFP4A8 groupwise")
    logger.info("=" * 60)
    for m in args.m:
        ret = run_accuracy_case(
            test_aiter_moe_wfp4a8_groupwise,
                m=m,
                k=args.k,
                n=args.n,
                e=args.e,
                topk=args.topk,
                group_size=args.group_size,
                dtype=torch.bfloat16,
                activation=args.activation,
                mode1=args.mode1,
                mode2=args.mode2,
                a_quant=args.a_quant,
        )
        if ret is not None:
            rows.append(ret)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("wfp4a8_groupwise.csv", index=False)
        logger.info(f"summary:\n{df}")
        print(df.to_string(index=False))

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )


if __name__ == "__main__":
    main()
