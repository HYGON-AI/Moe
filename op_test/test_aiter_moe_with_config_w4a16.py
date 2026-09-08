# Test for get_aiter_moe_config_w4a16 and aiter_moe_w4a16

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
from moe_utils.moe import w4a16_marlin_weight_1, w4a16_marlin_weight_2

torch.set_default_device("cuda")


# ---------------------------------------------------------------------------
# Weight quantization helpers (adapted from test_moe_wna16.py)
# ---------------------------------------------------------------------------

def _quantize_w4a16_weights(w_fp, group_size, has_zp, pack_for_backend):
    """Quantize a single expert weight matrix to int4.

    Args:
        w_fp: Floating-point weight ``[out_features, in_features]``.
        group_size: Quantization group size along K.
        has_zp: Whether to produce zero-points.
        pack_for_backend: currently ``"moe_c"`` in this standalone repository.

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
        qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

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

    if has_zp:
        w1_qzeros = torch.empty(
            (e, 2 * n // pack_factor, k // group_size), device="cuda",
            dtype=torch.uint8)
        w2_qzeros = torch.empty(
            (e, k // pack_factor, n // group_size), device="cuda",
            dtype=torch.uint8)
    else:
        # w1_qzeros = None
        # w2_qzeros = None
        w1_qzeros = torch.empty(
                (e, 2 * n // pack_factor, k // group_size), device="cuda",
                dtype=torch.uint8)
        w2_qzeros = torch.empty(
            (e, k // pack_factor, n // group_size), device="cuda",
            dtype=torch.uint8)

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

    # For moe_c backend, apply marlin weight shuffle
    if backend == "moe_c":
        # w1_qweight_final = w4a16_marlin_weight_1(w1_qweight)
        # w2_qweight_final = w4a16_marlin_weight_2(w2_qweight)
        # w1_qweight_final = w1_qweight_final.view(-1).view(
        #     torch.uint8).view(*w1_qweight.shape)
        # w2_qweight_final = w2_qweight_final.view(-1).view(
        #     torch.uint8).view(*w2_qweight.shape)
        if config.need_shuffle_scale:
            w1_scales, w2_scales = aiter_moe_shfl_scale(w1_scales,w2_scales,config)
        # pmt_factor = 4
        # pmt_factor2 = 4
        # w1_scales_tm = w1_scales.reshape(e,2*n,(k // group_size // pmt_factor),pmt_factor).transpose(2,3).contiguous()
        # w1_scales = w1_scales_tm.reshape(e,2*n,k // group_size)
        # w2_scales_tm = w2_scales.reshape(e,k,(n // group_size // pmt_factor2),pmt_factor2).transpose(2,3).contiguous()
        # w2_scales = w2_scales_tm.reshape(e,k,n // group_size)
        w1_qweight_final = w1_qweight
        w2_qweight_final = w2_qweight
    else:
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

def test_get_config(m, k, n, e, topk, group_size, dtype):
    """Test that get_aiter_moe_config returns a valid w4a16 config or
    gracefully reports no-solution."""
    N1 = 2 * n  # gate + up
    N2 = k      # down / hidden_size
    K = k       # model dimension (uncompressed)

    status, moe_cfg = get_aiter_moe_config(
        M=m, E=e, N1=N1, N2=N2, K=K,
        top_k=topk, block_size=group_size, dtype=dtype,
        quant_type=MoeQuantType.W4A16,
    )

    if status:
        assert moe_cfg.solution_type is not None, \
            "status=True but solution_type is None"
        assert moe_cfg.config is not None, \
            "status=True but config is None"
        assert moe_cfg.solution_type == MoeSolutionType.MOE_C, f"Unexpected solution_type: {moe_cfg.solution_type}"
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
        activation):

        if inplace:
            mortal_input = hidden_states.clone()  # 保证inplace操作的正确性
        else:
            mortal_input = hidden_states

        return aiter_moe(mortal_input, w1, w2, topk_weights, topk_ids, moe_config, inplace, activation, w1_scale, w2_scale, w1_zp, w2_zp, 
                     a1_scale, a2_scale, block_shape, global_num_experts, expert_map, routed_scaling_factor, output_dtype=hidden_states.dtype)


def test_aiter_moe_w4a16(m, k, n, e, topk, group_size, has_zp, dtype, inplace, routed_scaling_factor):
    """End-to-end: get config -> run aiter_moe -> compare with torch
    reference."""
    N1 = 2 * n
    N2 = k
    K = k

    status, moe_cfg = get_aiter_moe_config(
        M=m, E=e, N1=N1, N2=N2, K=K,
        top_k=topk, block_size=group_size, dtype=dtype,
        quant_type=MoeQuantType.W4A16,
    )

    if not status:
        logger.info(
            f"[aiter_moe_w4a16] SKIP {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}: "
            f"no backend available"
        )
        return None

    backend = moe_cfg.solution_type
    logger.info(
        f"[aiter_moe_w4a16] {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}, "
        f"backend={backend}"
    )

    data = prepare_w4a16_inputs(
        m, k, n, e, topk, group_size, has_zp, dtype, backend, moe_cfg)

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

    dtype = dtypes.bf16
    group_size = 32
    has_zp = False
    e = 384
    topk = 8
    k = 7168       # model_dim
    n = 256        # intermediate_size
    inplace = True
    routed_scaling_factor = 1.0

    # --- Part 1: test get_aiter_moe_config (w4a16) ---
    logger.info("=" * 60)
    logger.info("Part 1: Testing get_aiter_moe_config for w4a16")
    logger.info("=" * 60)

    test_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # test_tokens = [4096]
    for m in test_tokens:
        test_get_config(m, k, n, e, topk, group_size, dtype)

    # --- Part 2: test aiter_moe end-to-end (w4a16) ---
    logger.info("=" * 60)
    logger.info("Part 2: Testing aiter_moe end-to-end for w4a16")
    logger.info("=" * 60)

    df = []
    for m in test_tokens:
        # for n in [256, 512, 1024, 2048]:
        ret = run_accuracy_case(
            test_aiter_moe_w4a16,
            m, k, n, e, topk, group_size, has_zp, dtype, inplace,
            routed_scaling_factor,
        )
        if ret is not None:
            df.append(ret)
    if df:
        df = pd.DataFrame(df)
        df.to_csv("test_aiter_moe_with_config_w4a16.csv", index=False)
        logger.info(f"summary:\n{df}")

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
