# Test for get_aiter_moe_config and aiter_moe with w8a8

import torch
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_utils.utils import fused_topk
from moe_utils.utils import dtypes
from moe_utils.utils import checkAllclose, logger, perftest
from moe_utils.moe import (
    get_aiter_moe_config,
    aiter_moe,
    MoeSolutionType,
    MoeQuantType,
)


torch.set_default_device("cuda")


def torch_moe_blockscale(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    dtype,
    scale_blks,
    fc1_scale,
    fc2_scale,
):
    compute_type = torch.float32
    hidden_states = hidden_states.to(compute_type)
    w1 = w1.to(compute_type)
    w2 = w2.to(compute_type)
    token_num, topk = topk_ids.shape
    expert, model_dim, inter_dim = w2.shape
    blk_n, blk_k = scale_blks

    nblk_n = inter_dim // blk_n
    nblk_k = model_dim // blk_k
    fc1_scale_full = fc1_scale.view(-1, 1).repeat(1, blk_n * blk_k).view(
        expert, -1, nblk_k, blk_n, blk_k)
    fc1_scale_full = fc1_scale_full.permute(0, 1, 3, 2, 4).contiguous().view(
        expert, 2 * inter_dim, model_dim)

    fc2_scale_full = fc2_scale.view(-1, 1).repeat(1, blk_n * blk_k).view(
        expert, model_dim // blk_n, inter_dim // blk_k, blk_n, blk_k)
    fc2_scale_full = fc2_scale_full.permute(0, 1, 3, 2, 4).contiguous().view(
        expert, model_dim, inter_dim)

    w1 = w1 * fc1_scale_full
    w2 = w2 * fc2_scale_full

    hidden_states = hidden_states.view(token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros((token_num, topk, model_dim), dtype=compute_type, device=hidden_states.device)

    for expert_id in range(w1.shape[0]):
        mask = topk_ids == expert_id
        if mask.sum() == 0:
            continue
        sub_tokens = hidden_states[mask]
        act_input = sub_tokens @ w1[expert_id].transpose(0, 1)
        gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
        act_out = torch.nn.functional.silu(gate) * up
        out[mask] = act_out @ w2[expert_id].transpose(0, 1)

    return (out * topk_weight.view(token_num, -1, 1)).sum(dim=1).to(dtype)


@perftest(num_warmup=1, num_iters=2)
def _run_torch_ref(hidden_states, w1, w2, topk_weights, topk_ids, dtype, block_shape, w1_scale, w2_scale):
    return torch_moe_blockscale(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype,
        block_shape,
        w1_scale,
        w2_scale,
    )


@perftest(num_warmup=10, num_iters=100, num_rotate_args=1)
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
):

    if inplace:
        mortal_input = hidden_states.clone()  # 保证inplace操作的正确性
    else:
        mortal_input = hidden_states

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
    )


def prepare_w8a8_inputs(m, k, n, e, topk, block_shape, dtype):
    torch.manual_seed(0)
    factor_for_scale = 1e-2
    int8_info = torch.iinfo(torch.int8)
    int8_max, int8_min = int8_info.max, int8_info.min

    input_tensor = torch.randn((m, k), dtype=dtype, device="cuda") / 10
    w1_fp = (torch.rand((e, 2 * n, k), dtype=dtype, device="cuda") - 0.5) * 2 * int8_max
    w2_fp = (torch.rand((e, k, n), dtype=torch.float32, device="cuda") - 0.5) * 2 * int8_max
    w1_qweight = w1_fp.clamp(min=int8_min, max=int8_max).to(torch.int8)
    w2_qweight = w2_fp.clamp(min=int8_min, max=int8_max).to(torch.int8)

    block_n, block_k = block_shape
    n_tiles_w1 = (2 * n + block_n - 1) // block_n
    n_tiles_w2 = (k + block_n - 1) // block_n
    k_tiles_w1 = (k + block_k - 1) // block_k
    k_tiles_w2 = (n + block_k - 1) // block_k
    w1_scales = torch.rand((e, n_tiles_w1, k_tiles_w1), dtype=torch.float32, device="cuda") * factor_for_scale
    w2_scales = torch.rand((e, n_tiles_w2, k_tiles_w2), dtype=torch.float32, device="cuda") * factor_for_scale
    score = torch.randn((m, e), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, False)

    return {
        "input": input_tensor,
        "w1_ref": w1_qweight,
        "w2_ref": w2_qweight,
        "w1_qweight": w1_qweight,
        "w2_qweight": w2_qweight,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def test_get_config(m, k, n, e, topk, block_shape, dtype):
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=block_shape[1],
        dtype=dtype,
        quant_type=MoeQuantType.W8A8,
    )

    if status:
        assert moe_cfg.quant_type == MoeQuantType.W8A8
        assert moe_cfg.solution_type == MoeSolutionType.MOE_C
        assert moe_cfg.config is not None
        logger.info(
            f"[get_config_w8a8] {m=}, solution={moe_cfg.solution_type}, config keys={list(moe_cfg.config.keys())}"
        )
    else:
        assert moe_cfg.solution_type is None
        assert moe_cfg.config is None
        logger.info(f"[get_config_w8a8] {m=}, no solution found")

    return status, moe_cfg


def test_aiter_moe_w8a8(m, k, n, e, topk, block_shape, dtype):
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=block_shape[1],
        dtype=dtype,
        quant_type=MoeQuantType.W8A8,
    )

    if not status:
        logger.info(f"[aiter_moe_w8a8] SKIP {m=}: no backend available")
        return None

    data = prepare_w8a8_inputs(m, k, n, e, topk, block_shape, dtype)
    ref_out, _ = _run_torch_ref(
        data["input"],
        data["w1_ref"],
        data["w2_ref"],
        data["topk_weights"],
        data["topk_ids"],
        dtype,
        block_shape,
        data["w1_scales"],
        data["w2_scales"],
    )

    aiter_us = 1.0
    # aiter_out = aiter_moe(
    aiter_out, aiter_us = _run_aiter_moe_perf(
        hidden_states=data["input"],
        w1=data["w1_qweight"],
        w2=data["w2_qweight"],
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        moe_config=moe_cfg,
        inplace=False,
        activation="silu",
        w1_scale=data["w1_scales"],
        w2_scale=data["w2_scales"],
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=list(block_shape),
        global_num_experts=e,
        expert_map=None,
    )

    msg = f"[aiter_moe_w8a8] {m=}, backend={moe_cfg.solution_type}"
    assert torch.isfinite(aiter_out).all(), (
        "Non-finite output in test_aiter_moe_w8a8_blockwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {block_shape=}, {dtype=}, "
        f"backend={moe_cfg.solution_type}"
    )
    check_ret = checkAllclose(ref_out, aiter_out, rtol=0.01, atol=100, msg=msg)
    assert check_ret <= 0.03, (
        "Accuracy check failed in test_aiter_moe_w8a8_blockwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {block_shape=}, {dtype=}, "
        f"backend={moe_cfg.solution_type}, error_ratio={check_ret:.6f}, tolerance=0.03"
    )
    passed = "passed" if check_ret == 0 else (1-check_ret)
    return {"m": m, "backend": moe_cfg.solution_type, "us": aiter_us, "accuracy": passed}


if __name__ == "__main__":
    failed_cases = []

    def run_accuracy_case(test_func, *args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except AssertionError as exc:
            failed_cases.append(str(exc))
            print(f"[ACCURACY FAILED] {exc}", flush=True)
            return None
    dtype = dtypes.fp16
    block_shape = (128, 128)
    e = 256
    topk = 8
    k = 7168
    n = 256

    logger.info("=" * 60)
    logger.info("Part 1: Testing get_aiter_moe_config for w8a8")
    logger.info("=" * 60)
    test_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for m in test_tokens:
        test_get_config(m, k, n, e, topk, block_shape, dtype)

    logger.info("=" * 60)
    logger.info("Part 2: Testing aiter_moe end-to-end for w8a8")
    logger.info("=" * 60)
    df = []
    for m in test_tokens:
        ret = run_accuracy_case(
            test_aiter_moe_w8a8, m, k, n, e, topk, block_shape, dtype
        )
        if ret is not None:
            df.append(ret)
    if df:
        df = pd.DataFrame(df)
        logger.info(f"summary:\n{df}")
        df.to_csv("test_aiter_moe_with_config_w8a8_blockwise.csv", index=False)

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
