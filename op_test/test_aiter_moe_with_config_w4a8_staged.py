# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Stage-level benchmark for standalone W4A8 per-channel MoE.

Measurement model and CLI flags are shared with the other staged benchmarks,
see ``staged_common.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch

import staged_common as sc
from moe_utils.golden import run_w4a8_perchannel_golden
from moe_utils.moe import (
    MoeQuantType,
    MoeSolutionType,
    get_moe_config,
    moe,
    moe_shfl_weight,
)
from moe_utils.utils import checkAllclose, dtypes
from test_aiter_moe_with_config_w4a8_perchannel import prepare_w4a8_inputs

__test__ = False


def build_case(m: int, k: int, n: int, e: int, topk: int, dtype: torch.dtype):
    """Return ``(run, check, config)`` for one M; inputs are built once."""
    status, cfg = get_moe_config(
        M=m, E=e, N1=2 * n, N2=k, K=k, top_k=topk, block_size=None,
        dtype=dtype, quant_type=MoeQuantType.W4A8,
        spec_sol_type=MoeSolutionType.MOE_C, use_shuffle=0,
    )
    if not status or cfg.solution_type != "moe_c":
        raise RuntimeError(f"No moe_c W4A8 per-channel config for M={m}")

    data = prepare_w4a8_inputs(m, k, n, e, topk, dtype)
    w1, w2 = data["w1_qweight"], data["w2_qweight"]
    if cfg.need_shuffle:
        w1, w2 = moe_shfl_weight(w1, w2, cfg)
    # Scaled outside the timed callable, exactly like the endpoint test.
    w1_scale, w2_scale = data["w1_scales"] / 16, data["w2_scales"] / 16

    def run() -> torch.Tensor:
        return moe(
            data["input"], w1, w2, data["topk_weights"], data["topk_ids"], cfg,
            inplace=False, activation="silu",
            w1_scale=w1_scale, w2_scale=w2_scale, w1_zp=None, w2_zp=None,
            a1_scale=None, a2_scale=None, block_shape=None,
            global_num_experts=e, expert_map=None,
            use_weight_shuffle=False, output_dtype=dtype,
        )

    def check(output: torch.Tensor) -> None:
        assert torch.isfinite(output).all(), "non-finite output"
        ref = run_w4a8_perchannel_golden(
            data["input"], data["w1_ref"], data["w2_ref"], data["topk_weights"],
            data["topk_ids"], dtype, data["w1_scales"], data["w2_scales"],
        )
        ratio = checkAllclose(ref.to(output.dtype), output, rtol=0.01, atol=0.2,
                              msg="staged W4A8", printLog=False)
        assert ratio <= 0.03, f"error_ratio={ratio:.6f} > 0.03"

    return run, check, cfg


def run_case(m: int, args) -> Dict[str, Any]:
    dtype = sc.parse_dtype(args.dtype)
    run, check, cfg = build_case(m, args.k, args.n, args.e, args.topk, dtype)
    row = sc.measure(run, warmup=args.warmup, iters=args.iters,
                     check=None if args.no_check else check)
    row.update(m=m, MODE=int(cfg.config["MODE"]),
               BLOCK_SIZE_M=int(cfg.config["BLOCK_SIZE_M"]))
    return row


def main() -> None:
    args = sc.base_parser("Profile standalone W4A8 per-channel MoE stages",
                          e=256, n=256, k=7168).parse_args()
    args.build_dir = sc.select_build_dir(args.build_dir, ["libmoe_w4a8.so"])
    sc.run_sweep(args, lambda m: run_case(m, args))


if __name__ == "__main__":
    main()
