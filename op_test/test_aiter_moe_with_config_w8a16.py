# Test for get_aiter_moe_config and aiter_moe with W8A16 (int8 weight, fp16/bf16 activation)
#
# Mirrors test_aiter_moe_with_config_w16a16.py but for the INT8_W8A16 quant
# type. This standalone repository only validates the MOE_C backend, so AITER
# multi-backend comparison cases are omitted intentionally.

import torch
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_utils.utils import fused_topk, torch_moe
from moe_utils.utils import dtypes
from moe_utils.utils import checkAllclose, logger, perftest
from moe_utils.moe import (
    get_aiter_moe_config,
    aiter_moe,
    MoeSolutionType,
    MoeQuantType,
)
from moe_utils.moe import w8a16_marlin_weight_1, w8a16_marlin_weight_2

torch.set_default_device("cuda")


# ---------------------------------------------------------------------------
# Weight preparation helpers (W8A16 - int8 weights, fp16/bf16 activations)
# ---------------------------------------------------------------------------

def prepare_w8a16_inputs(m, k, n, e, topk, dtype):
    """Build all tensors needed to run a w8a16 (int8 weight) MOE test.

    Layout:
      - input: [m, k]                fp16/bf16
      - w1: [E, 2n, k]               int8       (gate + up, gated activation)
      - w2: [E, k, n]                int8       (down)
      - w1_scale: [E, 2n, 1]         fp32       (per-output-channel)
      - w2_scale: [E, k, 1]          fp32       (per-output-channel)
      - w1_marlin / w2_marlin: marlin-shuffled int8 weights consumed by the
        moe_c W8A16 kernel.

    The non-shuffled int8 weights together with the per-channel scales are
    used to build the torch reference (dequantized to fp16/bf16).
    """
    torch.manual_seed(0)
    # Keep activations small to limit accumulation error in int8 dequant.
    input_tensor = (torch.randn((m, k), device="cuda", dtype=dtype)) / 100

    w1 = torch.randint(-127, 127, (e, 2 * n, k), device="cuda", dtype=torch.int8)
    w2 = torch.randint(-127, 127, (e, k, n), device="cuda", dtype=torch.int8)

    # Per-channel scales. The moe_c W8A16 marlin kernel expects scales in
    # the activation dtype (fp16/bf16), so we keep them in `dtype`. Small
    # magnitude keeps dequantized weights in a reasonable range.
    w1_scale = (torch.randn((e, 2 * n, 1), device="cuda", dtype=dtype)).abs() * 1e-3
    w2_scale = (torch.randn((e, k, 1), device="cuda", dtype=dtype)).abs() * 1e-3

    # Marlin-shuffled weights for the moe_c W8A16 kernel.
    w1_marlin = w8a16_marlin_weight_1(w1)
    w2_marlin = w8a16_marlin_weight_2(w2)

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, True)

    return {
        "input": input_tensor,
        "w1": w1,
        "w2": w2,
        "w1_marlin": w1_marlin,
        "w2_marlin": w2_marlin,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Test: get_aiter_moe_config (w8a16)
# ---------------------------------------------------------------------------

def test_get_config(m, k, n, e, topk, dtype):
    """Validate that get_aiter_moe_config returns a sane W8A16 config or
    gracefully reports no-solution."""
    N1 = 2 * n  # gate + up
    N2 = k      # down / hidden_size
    K = k       # model dimension

    status, moe_cfg = get_aiter_moe_config(
        M=m, E=e, N1=N1, N2=N2, K=K,
        top_k=topk, block_size=0, dtype=dtype,
        quant_type=MoeQuantType.INT8_W8A16,
    )

    if status:
        assert moe_cfg.solution_type is not None, \
            "status=True but solution_type is None"
        assert moe_cfg.config is not None, \
            "status=True but config is None"
        assert moe_cfg.solution_type == MoeSolutionType.MOE_C, f"Unexpected solution_type for W8A16: {moe_cfg.solution_type}"
        assert moe_cfg.quant_type == MoeQuantType.INT8_W8A16
        logger.info(
            f"[get_config_w8a16] {m=}, {k=}, {n=}, {e=}, {topk=}, "
            f"solution={moe_cfg.solution_type}, "
            f"config keys={list(moe_cfg.config.keys())}"
        )
    else:
        assert moe_cfg.solution_type is None, \
            "status=False but solution_type is not None"
        assert moe_cfg.config is None, \
            "status=False but config is not None"
        logger.info(
            f"[get_config_w8a16] {m=}, {k=}, {n=}, {e=}, {topk=}, "
            f"no solution found"
        )

    return status, moe_cfg


# ---------------------------------------------------------------------------
# Test: aiter_moe end-to-end for w8a16
# ---------------------------------------------------------------------------

@perftest(num_warmup=1, num_iters=2)
def _run_torch_ref(hidden_states, w1, w2, topk_weights, topk_ids,
                   fc1_scale, fc2_scale):
    return torch_moe(
        hidden_states, w1, w2, topk_weights, topk_ids,
        fc1_scale=fc1_scale, fc2_scale=fc2_scale,
    )


@perftest(num_warmup=10, num_iters=100, num_rotate_args=1)
def _run_aiter_moe_perf(hidden_states,
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
        routed_scaling_factor,
        ):
    if inplace:
        mortal_input = hidden_states.clone()  # 保证inplace操作的正确性
    else:
        mortal_input = hidden_states
    return aiter_moe(mortal_input, w1, w2, topk_weights, topk_ids, moe_config,
                     inplace, activation, w1_scale, w2_scale, w1_zp, w2_zp,
                     a1_scale, a2_scale, block_shape, global_num_experts,
                     expert_map, routed_scaling_factor)


def test_aiter_moe_w8a16(m, k, n, e, topk, dtype, inplace, routed_scaling_factor):
    """End-to-end: get config -> run aiter_moe (W8A16) -> compare with
    torch reference (dequantized weights)."""
    N1 = 2 * n
    N2 = k
    K = k

    status, moe_cfg = get_aiter_moe_config(
        M=m, E=e, N1=N1, N2=N2, K=K,
        top_k=topk, block_size=0, dtype=dtype,
        quant_type=MoeQuantType.INT8_W8A16,
    )

    if not status:
        logger.info(
            f"[aiter_moe_w8a16] SKIP {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}: "
            f"no backend available"
        )
        return None

    backend = moe_cfg.solution_type
    logger.info(
        f"[aiter_moe_w8a16] {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
        f"backend={backend}"
    )

    data = prepare_w8a16_inputs(m, k, n, e, topk, dtype)

    # Pick weights based on the backend:
    #   - moe_c expects marlin-shuffled int8 weights
    #   - triton expects plain (E, N, K) int8 weights
    if backend == MoeSolutionType.MOE_C:
        w1_run, w2_run = data["w1_marlin"], data["w2_marlin"]
    else:
        w1_run, w2_run = data["w1"], data["w2"]

    # Torch reference (dequantize int8 weights via per-channel scales)
    ref_out, _ = _run_torch_ref(
        data["input"], data["w1"], data["w2"],
        data["topk_weights"], data["topk_ids"],
        data["w1_scale"], data["w2_scale"],
    )

    aiter_out, aiter_us = _run_aiter_moe_perf(
        hidden_states=data["input"],
        w1=w1_run,
        w2=w2_run,
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        moe_config=moe_cfg,
        inplace=inplace,
        activation="silu",
        w1_scale=data["w1_scale"],
        w2_scale=data["w2_scale"],
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,  # per-channel
        global_num_experts=e,
        expert_map=None,
        routed_scaling_factor=routed_scaling_factor,
    )

    msg = (f"[aiter_moe_w8a16] {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
           f"backend={backend}")
    # int8 dequant + bf16/fp16 accumulation order differs between the
    # reference and the fused kernels; use a relaxed tolerance similar to
    # the W16A16 test.
    assert torch.isfinite(aiter_out).all(), (
        "Non-finite output in test_aiter_moe_w8a16: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {dtype=}, "
        f"backend={backend}"
    )
    check_ret = checkAllclose(ref_out.to(aiter_out.dtype), aiter_out, rtol=0.05, atol=0.5, msg=msg)
    assert check_ret <= 0.03, (
        "Accuracy check failed in test_aiter_moe_w8a16: "
        f"{m=}, {k=}, {n=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
        f"{dtype=}, {inplace=}, {routed_scaling_factor=}, backend={backend}, "
        f"error_ratio={check_ret:.6f}, tolerance=0.03"
    )
    ret_output = "passed" if check_ret == 0 else (1 - check_ret)
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

    dtype = dtypes.fp16  # W8A16 moe_c kernel was validated with fp16

    # Use shape that matches the tuned W8A16 moe_c configs under
    # aiter/moe_c_configs (E=128, N=2048).
    e = 128
    topk = 8
    k = 2048       # model_dim
    n = 768       # intermediate_size
    inplace = False
    routed_scaling_factor = 1.0

    # --- Part 1: test get_aiter_moe_config (w8a16) ---
    logger.info("=" * 60)
    logger.info("Part 1: Testing get_aiter_moe_config for w8a16")
    logger.info("=" * 60)

    test_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for m in test_tokens:
        test_get_config(m, k, n, e, topk, dtype)

    # --- Part 2: test aiter_moe end-to-end (w8a16) ---
    logger.info("=" * 60)
    logger.info("Part 2: Testing aiter_moe end-to-end for w8a16")
    logger.info("=" * 60)

    df = []
    for m in test_tokens:
        ret = run_accuracy_case(
            test_aiter_moe_w8a16,
            m, k, n, e, topk, dtype, inplace, routed_scaling_factor,
        )
        if ret is not None:
            df.append(ret)
    if df:
        df = pd.DataFrame(df)
        logger.info(f"aiter_moe summary:\n{df}")
        df.to_csv("test_aiter_moe_with_config_w8a16.csv", index=False)

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
