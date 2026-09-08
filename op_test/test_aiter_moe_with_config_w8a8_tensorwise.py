# Test for get_aiter_moe_config and aiter_moe with w8a8 tensor-wise quantization

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

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
from moe_utils.moe import w8a8_moe_repack_shuffle as moe_layout_shuffle_gemm1, w8a8_moe_repack_shuffle as moe_layout_shuffle_gemm2
from moe_utils.utils import pertoken_quant


torch.set_default_device("cuda")


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2
) -> None:
    """
    比较两个任意维度的PyTorch张量的差异，支持绝对误差和相对误差阈值。
    直接输出详细比较结果，包括每个元素在原始张量中的多维坐标。
    无返回值。

    参数:
        tensor1: 第一个张量（如Triton模型输出）
        tensor2: 第二个张量（如PyTorch模型输出）
        atol: 绝对误差阈值，默认1e-5
        rtol: 相对误差阈值，默认1e-8
    """
    # -------------------------- 1. 输入合法性校验 --------------------------
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量（torch.Tensor）")
    
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"张量形状不匹配！ tensor1形状: {tensor1.shape}, tensor2形状: {tensor2.shape}")
    
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
        print(f"警告：张量设备不一致，已将tensor2转移到{tensor1.device}")

    # -------------------------- 2. 核心差异计算 --------------------------
    abs_diff = torch.abs(tensor1 - tensor2)
    denom = torch.maximum(torch.abs(tensor1), torch.abs(tensor2))
    rel_diff = abs_diff / (denom + 1e-12)
    match_mask = (abs_diff <= atol) | (rel_diff <= rtol)

    # -------------------------- 3. 展平张量 --------------------------
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    abs_diff_flat = abs_diff.flatten()
    match_mask_flat = match_mask.flatten()

    # -------------------------- 4. 收集一维索引 --------------------------
    def get_indices_1d(mask: torch.Tensor) -> list:
        indices = torch.nonzero(mask).squeeze(dim=1)
        return indices.tolist() if indices.numel() > 0 else []

    match_indices_1d = get_indices_1d(match_mask_flat)
    mismatch_indices_1d = get_indices_1d(~match_mask_flat)

    # -------------------------- 5. 总体统计信息 --------------------------
    total = tensor1_flat.numel()
    matched = len(match_indices_1d)
    mismatched = len(mismatch_indices_1d)
    match_rate = matched / total if total > 0 else 0.0
    max_abs_diff = abs_diff.max().item() if total > 0 else 0.0
    avg_abs_diff = abs_diff.mean().item() if total > 0 else 0.0

    # -------------------------- 6. 格式化输出 --------------------------
    print("=" * 60)
    print("张量比较结果汇总")
    print("=" * 60)
    print(f"张量形状: {tensor1.shape} | 总元素数: {total}")
    print(f"阈值设置: 绝对误差(atol)={atol:.2e}, 相对误差(rtol)={rtol:.2e}")
    print("-" * 60)
    print(f"匹配元素数: {matched} ({match_rate:.2%})")
    print(f"不匹配元素数: {mismatched} ({1 - match_rate:.2%})")
    print(f"最大绝对差异: {max_abs_diff:.6f}")
    print(f"平均绝对差异: {avg_abs_diff:.6f}")
    print("=" * 60)

    # -------------------------- 7. 输出匹配/不匹配示例 --------------------------
    def print_sample(name: str, indices_1d: list, max_samples: int = 3, elem_per_sample: int = 10) -> None:
        if not indices_1d:
            print(f"\n【{name}样本】无数据")
            return
        
        print(f"\n【{name}样本】（最多展示{max_samples}组，每组{elem_per_sample}个元素）")
        print("-" * 50)
        
        num_samples = min(max_samples, (len(indices_1d) + elem_per_sample - 1) // elem_per_sample)
        
        for i in range(num_samples):
            start = i * elem_per_sample
            end = start + elem_per_sample
            sample_indices_1d = indices_1d[start:end]
            
            # <<< 关键改动: 使用 torch.unravel_index 转换为多维坐标 >>>
            # 此函数能处理任意维度
            sample_coords = torch.unravel_index(torch.tensor(sample_indices_1d), tensor1.shape)
            # 将结果从张量元组转换为坐标元组列表
            sample_coords_list = list(zip(*[coord.tolist() for coord in sample_coords]))
            
            print(f"\n第{i+1}组:")
            print(f"  原始多维坐标: {sample_coords_list}")
            print(f"  tensor1: {[round(tensor1_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  tensor2: {[round(tensor2_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  绝对差异: {[round(abs_diff_flat[idx].item(), 6) for idx in sample_indices_1d]}")

    print_sample("匹配", match_indices_1d, max_samples=2)
    print_sample("不匹配", mismatch_indices_1d, max_samples=3)
    print("\n" + "=" * 60)


def _run_aiter_moe(
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
        output_dtype=out_dtype,
    )


@perftest(num_warmup=10, num_iters=100, num_rotate_args=1, testGraph=True)
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
        output_dtype=out_dtype,
    )


def _quantize_tensorwise_int8(weight):
    max_vals = torch.abs(weight.to(torch.float32)).amax(dim=(1, 2), keepdim=True)
    max_vals = max_vals.clamp(min=1e-5)
    scales = max_vals / 127.0
    qweight = (weight / max_vals * 127.0).round().clamp(min=-128, max=127).to(torch.int8)
    return qweight, scales


def _quantize_tensorwise_fp8(weight):
    finfo = torch.finfo(dtypes.fp8)
    max_vals = torch.abs(weight.to(torch.float32)).amax(dim=(1, 2), keepdim=True)
    max_vals = max_vals.clamp(min=1e-5)
    scales = max_vals / finfo.max
    qweight = (weight / scales).clamp(min=finfo.min, max=finfo.max).to(dtypes.fp8)
    return qweight, scales


def prepare_w8a8_tensorwise_inputs(m, k, n, e, topk, dtype, quant_type=MoeQuantType.W8A8):
    """Prepare tensor-wise quantized w8a8 inputs.

    For int8 (W8A8): weights quantized to torch.int8 with one scale per expert.
    For fp8  (FP8_W8A8): weights quantized to float8 with one scale per expert.
    Scale shape must be (e, 1, 1), which selects the tensorwise Marlin path.
    block_shape is None for this path.
    """
    torch.manual_seed(0)

    if dtype == dtypes.fp8:
        input_tensor = torch.randn((m, k), dtype=dtypes.fp32, device="cuda") / 10
        w1_fp = torch.randn((e, 2 * n, k), dtype=dtypes.fp32, device="cuda")
        w2_fp = torch.randn((e, k, n), dtype=dtypes.fp32, device="cuda")
    else:
        input_tensor = torch.randn((m, k), dtype=dtype, device="cuda") / 10
        w1_fp = torch.randn((e, 2 * n, k), dtype=dtype, device="cuda")
        w2_fp = torch.randn((e, k, n), dtype=dtype, device="cuda")

    input_for_aiter = input_tensor
    a1_scales = None
    if quant_type == MoeQuantType.FP8_W8A8:
        # Activation remains per-token quantized; tensorwise only applies to B scales.
        if dtype == dtypes.fp8:
            input_for_aiter, a1_scales = pertoken_quant(input_tensor, quant_dtype=dtypes.fp8)
        w1_qweight, w1_scales = _quantize_tensorwise_fp8(w1_fp)
        w2_qweight, w2_scales = _quantize_tensorwise_fp8(w2_fp)

    else:
        w1_qweight, w1_scales = _quantize_tensorwise_int8(w1_fp)
        w2_qweight, w2_scales = _quantize_tensorwise_int8(w2_fp)

    if dtype == dtypes.fp8:
        score = torch.randn((m, e), dtype=dtypes.fp32, device="cuda")
    else:
        score = torch.randn((m, e), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, True)

    # moe_c backend needs layout-shuffled weights
    w1_qweight_shuffle = moe_layout_shuffle_gemm1(w1_qweight).view(*w1_qweight.shape)
    w2_qweight_shuffle = moe_layout_shuffle_gemm2(w2_qweight).view(*w2_qweight.shape)

    return {
        "input": input_for_aiter,
        "a1_scales": a1_scales,
        "w1_qweight": w1_qweight,
        "w2_qweight": w2_qweight,
        "w1_qweight_shuffle": w1_qweight_shuffle,
        "w2_qweight_shuffle": w2_qweight_shuffle,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "w1_scales_channelwise": w1_scales.expand(e, 2 * n, 1).contiguous(),
        "w2_scales_channelwise": w2_scales.expand(e, k, 1).contiguous(),
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def test_get_config(m, k, n, e, topk, dtype, quant_type=MoeQuantType.W8A8):
    """Test get_aiter_moe_config for tensor-wise w8a8 (block_size=0)."""
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=0,
        dtype=dtype,
        quant_type=quant_type,
    )

    tag = f"get_config_{quant_type}_tw"
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


def test_aiter_moe_w8a8_tensorwise(
    m,
    k,
    n,
    e,
    topk,
    in_dtype,
    out_dtype,
    quant_type=MoeQuantType.W8A8,
    inplace=False,
    show_compare=False,
):
    """End-to-end test of aiter_moe with tensor-wise w8a8 (int8 or fp8)."""
    status, moe_cfg = get_aiter_moe_config(
        M=m,
        E=e,
        N1=2 * n,
        N2=k,
        K=k,
        top_k=topk,
        block_size=0,
        dtype=in_dtype,
        quant_type=quant_type,
    )

    tag = f"aiter_moe_{quant_type}_tw"
    if not status:
        logger.info(f"[{tag}] SKIP {m=}: no backend available")
        return None

    # Tensorwise scale shape (E, 1, 1) is currently implemented by the moe_c Marlin path.
    if moe_cfg.solution_type != MoeSolutionType.MOE_C:
        logger.info(f"[{tag}] SKIP {m=}: tensorwise requires moe_c, got {moe_cfg.solution_type}")
        return None

    data = prepare_w8a8_tensorwise_inputs(m, k, n, e, topk, in_dtype, quant_type)

    # The reference uses the existing channelwise moe_c path with scales expanded
    # from (E, 1, 1) to (E, out_dim, 1). This isolates tensorwise kernel logic.
    ref_out = _run_aiter_moe(
        hidden_states=data["input"],
        w1=data["w1_qweight_shuffle"],
        w2=data["w2_qweight_shuffle"],
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        moe_config=moe_cfg,
        inplace=inplace,
        activation="silu",
        w1_scale=data["w1_scales_channelwise"],
        w2_scale=data["w2_scales_channelwise"],
        w1_zp=None,
        w2_zp=None,
        a1_scale=data["a1_scales"],
        a2_scale=None,
        block_shape=None,
        global_num_experts=e,
        expert_map=None,
        out_dtype=out_dtype,
    )

    aiter_out = _run_aiter_moe(
        hidden_states=data["input"],
        w1=data["w1_qweight_shuffle"],
        w2=data["w2_qweight_shuffle"],
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
    )

    msg = f"[{tag}] {m=} {k=} {n=} {e=}, backend={moe_cfg.solution_type}"
    if show_compare:
        compare_tensors(aiter_out, ref_out)
    assert torch.isfinite(aiter_out).all(), (
        "Non-finite output in test_aiter_moe_w8a8_tensorwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {in_dtype=}, {out_dtype=}, "
        f"{quant_type=}, backend={moe_cfg.solution_type}"
    )
    check_ret = checkAllclose(ref_out.to(aiter_out.dtype), aiter_out, rtol=0.01, atol=0.01, msg=msg)
    assert check_ret <= 0.03, (
        "Accuracy check failed in test_aiter_moe_w8a8_tensorwise: "
        f"{m=}, {k=}, {n=}, {e=}, {topk=}, {in_dtype=}, {out_dtype=}, "
        f"{quant_type=}, {inplace=}, backend={moe_cfg.solution_type}, "
        f"error_ratio={check_ret:.6f}, tolerance=0.03"
    )
    _, aiter_us = _run_aiter_moe_perf(
        hidden_states=data["input"],
        w1=data["w1_qweight_shuffle"],
        w2=data["w2_qweight_shuffle"],
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
    )
    passed = "passed" if check_ret == 0 else (1 - check_ret)
    return {"m": m, "quant_type": quant_type, "backend": moe_cfg.solution_type, "us": aiter_us, "accuracy": passed}


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
        description="Test aiter_moe with tensor-wise w8a8 quantization",
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
    inplace = False  # in_dtype != out_dtype时，不能为True

    in_dtype = dtypes.fp8 if quant_type == MoeQuantType.FP8_W8A8 else dtypes.bf16
    out_dtype = dtypes.bf16
    e = 256
    topk = 8
    k = 2048
    n = 2048 if quant_type == MoeQuantType.FP8_W8A8 else 512
    # for k in [2048,4096,6144,7168]:
    # for n in [128,256,512,1024,2048]:

    logger.info("=" * 60)
    logger.info(f"Part 1: Testing get_aiter_moe_config for {quant_type} tensor-wise")
    logger.info("=" * 60)
    test_tokens = args.tokens or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]
    for m in test_tokens:
        test_get_config(m, k, n, e, topk, in_dtype, quant_type)

    logger.info("=" * 60)
    logger.info(f"Part 2: Testing aiter_moe end-to-end for {quant_type} tensor-wise")
    logger.info("=" * 60)
    df = []
    for m in test_tokens:
        ret = run_accuracy_case(
            test_aiter_moe_w8a8_tensorwise,
            m, k, n, e, topk, in_dtype, out_dtype, quant_type, inplace,
            args.compare,
        )
        if ret is not None:
            df.append(ret)
    if df:
        df = pd.DataFrame(df)
        df.to_csv("w8a8_tensorwise.csv", index=False)
        logger.info(f"summary:\n{df}")

    if failed_cases:
        details = "\n".join(
            f"[{index}] {message}"
            for index, message in enumerate(failed_cases, start=1)
        )
        raise AssertionError(
            f"{len(failed_cases)} accuracy case(s) failed:\n{details}"
        )
