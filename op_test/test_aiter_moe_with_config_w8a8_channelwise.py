# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Test for get_aiter_moe_config and aiter_moe with w8a8 channel-wise quantization

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_utils.moe import (
    get_moe_config,
    moe,
    moe_shfl_weight,
    MoeSolutionType,
    MoeQuantType,
)
from moe_utils.golden import run_fp8_channelwise_golden
from moe_utils.utils import (
    checkAllclose,
    compare_tensors,
    dtypes,
    fused_topk,
    logger,
    perftest,
    pertoken_quant,
    torch_moe,
)


torch.set_default_device("cuda")


@perftest(num_warmup=1, num_iters=2)
def _run_torch_ref(hidden_states, w1, w2, topk_weights, topk_ids):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )


@perftest(num_warmup=10, num_iters=100, num_rotate_args=1,testGraph=True)
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

    if inplace:
        mortal_input = hidden_states.clone()  # 保证inplace操作的正确性
    else:
        mortal_input = hidden_states

    return moe(
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
        use_weight_shuffle=use_shuffle,
        output_dtype = out_dtype
    )


def prepare_w8a8_channelwise_inputs(m, k, n, e, topk, dtype, quant_type=MoeQuantType.W8A8):
    """Prepare channel-wise quantized w8a8 inputs.

    For int8 (W8A8): weights quantized to torch.int8, scales = max_val / 127.
    For fp8  (FP8_W8A8): weights quantized to float8 via pertoken_quant.
    Scale shape: (e, out_dim, 1) — one scale per output channel.
    block_shape is None for channel-wise.
    """
    torch.manual_seed(0)

    if dtype == dtypes.fp8:
        input_tensor = torch.randn((m, k), dtype=dtypes.fp32, device="cuda") / 100
        w1_fp = torch.randn((e, 2 * n, k), dtype=dtypes.fp32, device="cuda")
        w2_fp = torch.randn((e, k, n), dtype=dtypes.fp32, device="cuda")

    else:
        input_tensor = torch.randn((m, k), dtype=dtype, device="cuda") / 100
        w1_fp = torch.randn((e, 2 * n, k), dtype=dtype, device="cuda")
        w2_fp = torch.randn((e, k, n), dtype=dtype, device="cuda")

    if quant_type == MoeQuantType.FP8_W8A8:
        # FP8 channel-wise quantization via pertoken_quant
        input_tensor_q, a1_scales = None, None
        if dtype == dtypes.fp8:
            input_tensor_q, a1_scales = pertoken_quant(input_tensor, quant_dtype=dtypes.fp8)
        w1_qweight, w1_scales = pertoken_quant(w1_fp, quant_dtype=dtypes.fp8)
        w2_qweight, w2_scales = pertoken_quant(w2_fp, quant_dtype=dtypes.fp8)
    else:
        # INT8 channel-wise quantization: max per output channel
        max_vals_w1 = torch.abs(w1_fp.to(torch.float32)).max(dim=-1, keepdim=True)[0]
        max_vals_w1 = max_vals_w1.clamp(min=1e-5)
        w1_scales = max_vals_w1 / 127.0  # (e, 2*n, 1)
        w1_qweight = (w1_fp / max_vals_w1 * 127.0).round().clamp(min=-128, max=127).to(torch.int8)

        max_vals_w2 = torch.abs(w2_fp.to(torch.float32)).max(dim=-1, keepdim=True)[0]
        max_vals_w2 = max_vals_w2.clamp(min=1e-5)
        w2_scales = max_vals_w2 / 127.0  # (e, k, 1)
        w2_qweight = (w2_fp / max_vals_w2 * 127.0).round().clamp(min=-128, max=127).to(torch.int8)

    if dtype == dtypes.fp8:
        score = torch.randn((m, e), dtype=dtypes.fp32, device="cuda")
    else:
        score = torch.randn((m, e), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, True)

    return {
        "input": input_tensor_q if dtype == dtypes.fp8 else input_tensor,
        "a1_scales": a1_scales if dtype == dtypes.fp8 else None,
        "w1_ref": w1_fp,
        "w2_ref": w2_fp,
        "w1_qweight": w1_qweight,
        "w2_qweight": w2_qweight,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "input_ref": input_tensor,  # original fp32/bf16 input for reference
    }


def test_get_config(m, k, n, e, topk, dtype, quant_type=MoeQuantType.W8A8, use_shuffle=0):
    """Test get_moe_config for channel-wise w8a8 (block_size=0)."""
    status, moe_cfg = get_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=0,
        dtype=dtype,
        quant_type=quant_type,
        use_shuffle=use_shuffle,
    )

    tag = f"get_config_{quant_type}_cw"
    if status:
        assert moe_cfg.quant_type == quant_type
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


def test_aiter_moe_w8a8_channelwise(
    m,
    k,
    n,
    e,
    topk,
    in_dtype,
    out_dtype,
    quant_type=MoeQuantType.W8A8,
    inplace=False,
    use_shuffle=0,
    show_compare=False,
):
    """End-to-end test of aiter_moe with channel-wise w8a8 (int8 or fp8)."""
    status, moe_cfg = get_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=0,
        dtype=in_dtype,
        quant_type=quant_type,
        use_shuffle=use_shuffle,
    )

    tag = f"aiter_moe_{quant_type}_cw"
    if not status:
        logger.info(f"[{tag}] SKIP {m=}: no backend available")
        return None

    data = prepare_w8a8_channelwise_inputs(m, k, n, e, topk, in_dtype, quant_type)
    if quant_type == MoeQuantType.FP8_W8A8:
        ref_out = run_fp8_channelwise_golden(
            data["input"],
            data["w1_qweight"],
            data["w2_qweight"],
            data["topk_weights"],
            data["topk_ids"],
            out_dtype,
            data["w1_scales"],
            data["w2_scales"],
            data["a1_scales"],
            e,
        )
    else:
        ref_out, _ = _run_torch_ref(
            data["input_ref"],
            data["w1_ref"],
            data["w2_ref"],
            data["topk_weights"],
            data["topk_ids"],
        )

    # Shuffle weights if the backend requires it (e.g. moe_c marlin layout)
    w1_input, w2_input = data["w1_qweight"], data["w2_qweight"]
    if moe_cfg.need_shuffle:
        w1_input, w2_input = moe_shfl_weight(
            data["w1_qweight"], data["w2_qweight"], moe_cfg
        )

    aiter_out, aiter_us = _run_aiter_moe_perf(
        hidden_states=data["input"],
        w1=w1_input,
        w2=w2_input,
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        moe_config=moe_cfg,
        inplace=inplace,
        activation="silu",
        w1_scale=data["w1_scales"],
        w2_scale=data["w2_scales"],
        w1_zp=None,
        w2_zp=None,
        a1_scale=data["a1_scales"],
        a2_scale=None,
        block_shape=None,
        global_num_experts=e,
        expert_map=None,
        out_dtype=out_dtype,
        use_shuffle=use_shuffle,
    )


    # compare_tensors(aiter_out, ref_out.to(aiter_out.dtype))
    # Compare in aiter_out's dtype since reference dtype may differ by backend.
    msg = f"[{tag}] {m=}, backend={moe_cfg.solution_type}"
    assert torch.isfinite(aiter_out).all(), (
        "Non-finite output in test_aiter_moe_w8a8_channelwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {in_dtype=}, {out_dtype=}, "
        f"{quant_type=}, backend={moe_cfg.solution_type}"
    )
    check_ret = checkAllclose(ref_out.to(aiter_out.dtype), aiter_out, rtol=0.01, atol=0.5, msg=msg)
    if show_compare:
        compare_tensors(aiter_out, ref_out.to(aiter_out.dtype), atol=0.5, rtol=0.01)
    assert check_ret <= 0.03, (
        "Accuracy check failed in test_aiter_moe_w8a8_channelwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {in_dtype=}, {out_dtype=}, "
        f"{quant_type=}, {inplace=}, {use_shuffle=}, "
        f"backend={moe_cfg.solution_type}, need_shuffle={moe_cfg.need_shuffle}, "
        f"error_ratio={check_ret:.6f}, tolerance=0.03"
    )
    passed = "passed" if check_ret == 0 else (1-check_ret)
    return {
        "m": m,
        "quant_type": quant_type,
        "backend": moe_cfg.solution_type,
        "us": aiter_us,
        "accuracy": passed,
    }


if __name__ == "__main__":
    failed_cases = []

    def run_accuracy_case(test_func, *args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except AssertionError as exc:
            failed_cases.append(str(exc))
            print(f"[ACCURACY FAILED] {exc}", flush=True)
            return None
    parser = argparse.ArgumentParser(
        description="Test aiter_moe with channel-wise w8a8 quantization",
    )
    parser.add_argument(
        "--quant",
        choices=["int8", "fp8"],
        default="fp8",
        help="Quantization type: int8 (MoeQuantType.W8A8) or fp8 (MoeQuantType.FP8_W8A8)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=None,
        help="Token sizes to run. Default runs the full CI sweep.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print detailed compare_tensors output for each executed case.",
    )
    args = parser.parse_args()

    quant_type = MoeQuantType.FP8_W8A8 if args.quant == "fp8" else MoeQuantType.W8A8
    inplace = False     # in_dtype != out_dtype时，不能为True

    # for moe_c backend, it does not support n=320 for now;
    # for triton backend, it can run with n=320 in NMZ;
    in_dtype = dtypes.fp8 if quant_type == MoeQuantType.FP8_W8A8 else dtypes.bf16
    out_dtype = dtypes.bf16
    e = 384 if quant_type == MoeQuantType.FP8_W8A8 else 256
    topk = 8
    k = 4096
    n = 512 if quant_type == MoeQuantType.FP8_W8A8 else 384
    use_shuffle = 0

    logger.info("=" * 60)
    logger.info(f"Part 1: Testing get_moe_config for {quant_type} channel-wise")
    logger.info("=" * 60)
    test_tokens = args.tokens or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 , 4096, 6144 , 8192 , 16384]
    for m in test_tokens:
        test_get_config(m, k, n, e, topk, in_dtype, quant_type, use_shuffle)

    logger.info("=" * 60)
    logger.info(f"Part 2: Testing moe end-to-end for {quant_type} channel-wise")
    logger.info("=" * 60)
    df = []
    for m in test_tokens:
        ret = run_accuracy_case(
            test_aiter_moe_w8a8_channelwise,
            m, k, n, e, topk, in_dtype, out_dtype, quant_type, inplace,
            use_shuffle, args.compare,
        )
        if ret is not None:
            df.append(ret)
    if df:
        df = pd.DataFrame(df)
        df.to_csv("w8a8_channelwise.csv", index=False)
        logger.info(f"summary:\n{df}")

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
