# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Stage-level benchmark for standalone W4A16 MoE.

Measurement model and CLI flags are shared with the other staged benchmarks,
see ``staged_common.py``.  W4A16 has no activation-quant stages, so
``quant1_us`` and ``quant2_us`` stay at zero.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

import staged_common as sc
import test_aiter_moe_with_config_w4a16 as ref

__test__ = False


def build_case(m: int, args, dtype: torch.dtype):
    """Return ``(run, check, config)`` for one M; inputs are built once."""
    n, k, e, topk, gs = args.n, args.k, args.e, args.topk, args.group_size
    status, cfg = ref.get_aiter_moe_config(
        M=m, E=e, N1=2 * n, N2=k, K=k, top_k=topk, block_size=gs, dtype=dtype,
        quant_type=ref.MoeQuantType.W4A16,
        spec_sol_type=ref.MoeSolutionType.MOE_C,
    )
    if not status or cfg.config is None:
        raise RuntimeError(f"No moe_c W4A16 config for M={m}")

    data = ref.prepare_w4a16_inputs(m, k, n, e, topk, gs, args.has_zp, dtype,
                                    ref.MoeSolutionType.MOE_C, cfg)
    if cfg.need_shuffle_scale:
        data["w1_scales"], data["w2_scales"] = ref.aiter_moe_shfl_scale(
            data["w1_scales"], data["w2_scales"], cfg)

    def run() -> torch.Tensor:
        return ref.aiter_moe(
            data["input"], data["w1_qweight"], data["w2_qweight"],
            data["topk_weights"], data["topk_ids"], cfg, False, "silu",
            data["w1_scales"], data["w2_scales"],
            data["w1_qzeros"], data["w2_qzeros"], None, None,
            [0, gs], e, None, 1.0, output_dtype=dtype,
        )

    def check(output: torch.Tensor) -> None:
        assert torch.isfinite(output).all(), "non-finite output"
        expected, _ = ref._run_torch_ref(
            data["input"], data["w1_ref"], data["w2_ref"],
            data["topk_weights"], data["topk_ids"])
        ratio = ref.checkAllclose(expected.to(output.dtype), output, rtol=0.01,
                                  atol=0.01, msg="staged W4A16", printLog=False)
        assert ratio <= 0.03, f"error_ratio={ratio:.6f} > 0.03"

    return run, check, cfg


def run_case(m: int, args) -> Dict[str, Any]:
    dtype = sc.parse_dtype(args.dtype)
    run, check, cfg = build_case(m, args, dtype)
    # The endpoint test does not graph-capture W4A16, so neither do we.
    row = sc.measure(run, warmup=args.warmup, iters=args.iters,
                     check=None if args.no_check else check, test_graph=False)
    row.update(m=m, MODE=int(cfg.config["MODE"]),
               BLOCK_SIZE_M=int(cfg.config["BLOCK_SIZE_M"]))
    return row


def main() -> None:
    parser = sc.base_parser("Profile standalone W4A16 MoE stages",
                            e=384, n=256, k=7168, iters=50)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--has-zp", action="store_true")
    args = parser.parse_args()
    args.build_dir = sc.select_build_dir(args.build_dir, ["libmoe_w4a16.so"])
    sc.run_sweep(args, lambda m: run_case(m, args))


if __name__ == "__main__":
    main()
