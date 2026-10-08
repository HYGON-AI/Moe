#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Exercise prefill at small M and decode at large M through standalone MoE-C."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=["fp8", "int8", "w4a8", "all"], default="all")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    import torch
    import moe_utils.fused_moe_c as runtime
    from moe_utils.utils import dtypes
    from moe_utils.utils import torch_moe
    from moe_utils.moe import (MoeConfig, MoeQuantType, MoeSolutionType, moe,
                              moe_shfl_weight)
    from moe_utils.golden import run_w4a8_perchannel_golden
    from moe_utils.utils import checkAllclose
    from op_test.test_aiter_moe_with_config_w4a8_perchannel import prepare_w4a8_inputs
    from op_test.test_aiter_moe_with_config_w8a8_channelwise import prepare_w8a8_channelwise_inputs
    out_dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16

    profiles = {
        "fp8": (MoeQuantType.FP8_W8A8, "fp8_w8a8", [
            (16, 32, 10202101, 20202101), (1024, 16, 10001102, 20001102),
            (16, 32, 11102102, 21102104)]),
        "int8": (MoeQuantType.W8A8, "int8_w8a8", [
            (16, 32, 10202101, 20202101), (1024, 16, 10001102, 20001102)]),
        "w4a8": (MoeQuantType.W4A8, "int8_w4a8", [
            (16, 16, 11101201, 21101201), (1024, 16, 11001104, 21001101)]),
    }
    original_loader = runtime.get_moe_configs_marlin
    original_align = runtime.moe_align_block_size
    original_sort = runtime.moe_sorting_ck
    total = 0
    for name, (quant, config_dtype, cases) in profiles.items():
        if args.quant not in (name, "all"):
            continue
        for m, bm, up, down in cases:
            route_calls = []
            down_override = {}
            def align(*args, **kwargs):
                route_calls.append("moe_align")
                return original_align(*args, **kwargs)
            def sort(*args, **kwargs):
                route_calls.append("ck_sorting")
                return original_sort(*args, **kwargs)
            def load_configs(E, N, dtype, block_n=None, block_k=None,
                             is_bottom=False, use_moe_wna16_cuda=False, **kwargs):
                if E == 4 and dtype == config_dtype:
                    config = {"MODE": down if is_bottom else up, "BLOCK_SIZE_M": bm}
                    if is_bottom:
                        config.update(down_override)
                    return {m: config}
                return original_loader(E, N, dtype, block_n, block_k,
                                       is_bottom=is_bottom,
                                       use_moe_wna16_cuda=use_moe_wna16_cuda, **kwargs)

            runtime.get_moe_configs_marlin = load_configs
            runtime.moe_align_block_size = align
            runtime.moe_sorting_ck = sort
            try:
                torch.manual_seed(0)
                cfg = MoeConfig(quant_type=quant, solution_type=MoeSolutionType.MOE_C,
                                config={"MODE": up, "BLOCK_SIZE_M": bm}, need_shuffle=True)
                if name == "w4a8":
                    data = prepare_w4a8_inputs(m, 4096, 256, 4, 2, out_dtype)
                    golden = run_w4a8_perchannel_golden(
                        data["input"], data["w1_ref"], data["w2_ref"],
                        data["topk_weights"], data["topk_ids"], out_dtype,
                        data["w1_scales"], data["w2_scales"])
                else:
                    data = prepare_w8a8_channelwise_inputs(m, 4096, 256, 4, 2, out_dtype, quant)
                    golden = torch_moe(data["input_ref"], data["w1_ref"], data["w2_ref"],
                                       data["topk_weights"], data["topk_ids"])
                w1, w2 = data["w1_qweight"], data["w2_qweight"]
                if cfg.need_shuffle:
                    w1, w2 = moe_shfl_weight(w1, w2, cfg)
                def run():
                    return moe(
                        hidden_states=data["input"], w1=w1, w2=w2,
                        topk_weights=data["topk_weights"], topk_ids=data["topk_ids"],
                        moe_config=cfg, inplace=False, activation="silu",
                        w1_scale=data["w1_scales"] / 16 if name == "w4a8" else data["w1_scales"],
                        w2_scale=data["w2_scales"] / 16 if name == "w4a8" else data["w2_scales"],
                        w1_zp=None, w2_zp=None, a1_scale=data.get("a1_scales"),
                        a2_scale=None, block_shape=None, global_num_experts=4,
                        expert_map=None, output_dtype=out_dtype,
                        use_weight_shuffle=cfg.need_shuffle)
                output = run()
                torch.cuda.synchronize()
                expected_route = "moe_align" if up // 1000000 % 10 else "ck_sorting"
                assert route_calls == [expected_route], route_calls
                assert torch.isfinite(output).all()
                error = checkAllclose(golden.to(output.dtype), output, atol=0.5, rtol=0.01)
                assert error <= 0.03, (name, m, error)
                for override in ({"BLOCK_SIZE_M": bm + 16},
                                 {"MODE": down + (-1000000 if down // 1000000 % 10 else 1000000)}):
                    down_override.update(override)
                    route_calls.clear()
                    try:
                        run()
                    except ValueError as exc:
                        assert "same" in str(exc), str(exc)
                    else:
                        raise AssertionError(f"incompatible stage pair accepted: {override}")
                    assert not route_calls, route_calls
                    down_override.clear()
                from moe_utils import moe_ops
                op = {
                    "fp8": moe_ops.moe_c_moe_gemm_marlin_w8a8_fp8,
                    "int8": moe_ops.moe_c_moe_gemm_marlin_w8a8,
                    "w4a8": moe_ops.moe_c_moe_gemm_marlin_w4a8,
                }[name]
                qdtype = torch.float8_e4m3fn if name == "fp8" else torch.int8
                qinput = torch.zeros((m, 4096), dtype=qdtype, device=w1.device)
                scratch = torch.empty((m, 2, 512), dtype=out_dtype, device=w1.device)
                scales = torch.ones((m, 1), dtype=torch.float32, device=w1.device)
                ids = torch.zeros((bm,), dtype=torch.int32, device=w1.device)
                experts = torch.zeros((1,), dtype=torch.int32, device=w1.device)
                postpad = torch.zeros((1,), dtype=torch.int32, device=w1.device)
                invalid_modes = [down, up // 1000 * 1000 + 999]
                if name != "fp8":
                    invalid_modes.append(up + (1000000 if name == "int8" else -1000000))
                for invalid_mode in invalid_modes:
                    call_args = [qinput, w1, scratch, scales, data["w1_scales"],
                                 None, ids, experts, postpad, 2, invalid_mode, 2, m]
                    if name == "fp8":
                        call_args.append(4096)
                    try:
                        op(*call_args)
                    except RuntimeError as exc:
                        assert "mode" in str(exc).lower(), str(exc)
                    else:
                        raise AssertionError(f"invalid mode {invalid_mode} did not raise")
                total += 1
                print(f"[PASS] quant={name} M={m} BM={bm} up={up} down={down} error_ratio={error}", flush=True)
            finally:
                runtime.get_moe_configs_marlin = original_loader
                runtime.moe_align_block_size = original_align
                runtime.moe_sorting_ck = original_sort
    print(f"[done] passed={total}", flush=True)


if __name__ == "__main__":
    main()
