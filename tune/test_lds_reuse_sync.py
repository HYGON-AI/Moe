"""Repeat independent GEMMs to check LDS reuse synchronization."""

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 1024])
    parser.add_argument("--quant", nargs="+",
                        choices=["int8_w8a8", "fp8_w8a8", "int8_w4a8"],
                        default=["int8_w8a8", "fp8_w8a8", "int8_w4a8"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    import torch
    from tune.tune_moe_hipc import (
        Shape, decode_mode, _make_w8a8_inputs, _bench_w8a8_stage,
    )

    cases = {
        "int8_w8a8": {
            "gemm1": [10202101, 10203101, 10204101, 10204102],
            "gemm2": [20102201, 20104202, 20001101, 20001102,
                      20001103, 20001104],
        },
        "fp8_w8a8": {
            "gemm1": [10204101],
            "gemm2": [20001101, 20001102, 20001103, 20001104],
        },
        "int8_w4a8": {
            "gemm1": [11201101, 11202101, 11203101, 11204101,
                      11204102, 11204104],
            "gemm2": [21001101, 21002102, 21003103, 21004104],
        },
    }
    saved = {}
    original = torch.allclose

    def capture(output, reference, **kwargs):
        saved["output"] = output.detach()
        return original(output, reference, **kwargs)

    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The stage benchmark owns the output buffer; capture it at validation.
    torch.allclose = capture
    try:
        for quant, stages in cases.items():
            if quant not in args.quant:
                continue
            for n in ((256, 320) if quant == "fp8_w8a8" else (256, 384)):
                for m in args.tokens:
                    shape = Shape(16, n, 4096, 4, m, "bf16", "gfx938", 72)
                    data = _make_w8a8_inputs(m, shape, quant)
                    for stage, modes in stages.items():
                        for mode in modes:
                            # Multi-tile N loops here require N=256.
                            if n != 256 and mode in (10204102, 11204102, 11204104):
                                continue
                            previous = None
                            for repeat in range(args.repeats):
                                saved.clear()
                                row = _bench_w8a8_stage(
                                    shape, data,
                                    {"MODE": mode, "BLOCK_SIZE_M": decode_mode(mode).bm},
                                    stage, quant, 1, 1,
                                )
                                output = saved.get("output")
                                changed = (None if previous is None or output is None
                                           else int((output != previous).sum().item()))
                                result = {
                                    "quant": quant, "n": n, "k": shape.k,
                                    "repeat": repeat, **asdict(row),
                                    "changed_elements": changed,
                                }
                                results.append(result)
                                print(json.dumps(result), flush=True)
                                previous = None if output is None else output.clone()
                            del previous
                    del data
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        torch.allclose = original
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    failed = [r for r in results if not r["ok"] or r["changed_elements"] not in (None, 0)]
    print(json.dumps({"runs": len(results), "failed": len(failed)}), flush=True)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
