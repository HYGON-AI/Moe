"""Compare before/after kernel timing in alternating rounds on one GPU."""

import argparse
import gc
import json
import multiprocessing as mp
import os
import statistics
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(pipe, build_dir, warmup, iterations):
    os.environ["MOE_C_BUILD_DIR"] = build_dir
    try:
        import torch
        import moe_utils.moe_ops as ops
        from tune.tune_moe_hipc import (
            Shape, decode_mode, _make_w8a8_inputs, _bench_w8a8_stage,
            _event_time_us,
        )

        captured = {}

        def wrap(op):
            def invoke(*args, **kwargs):
                captured["invoke"] = lambda: op(*args, **kwargs)
                captured["output"] = args[2]
                return op(*args, **kwargs)
            return invoke

        for name in ("moe_c_moe_gemm_marlin_w8a8", "moe_c_moe_gemm_marlin_w8a8_fp8",
                     "moe_c_moe_gemm_marlin_w4a8"):
            setattr(ops, name, wrap(getattr(ops, name)))
        pipe.send({"ready": True})
        while True:
            command = pipe.recv()
            if command["action"] == "stop":
                break
            if command["action"] == "prepare":
                captured.clear()
                gc.collect()
                torch.cuda.empty_cache()
                quant, stage, mode, m = command["case"]
                shape = Shape(16, 256, 4096, 4, m, "bf16", "gfx938", 72)
                data = _make_w8a8_inputs(m, shape, quant)
                row = _bench_w8a8_stage(
                    shape, data, {"MODE": mode, "BLOCK_SIZE_M": decode_mode(mode).bm},
                    stage, quant, 1, 1,
                )
                if "invoke" not in captured or "output" not in captured:
                    raise RuntimeError(row.error)
                output = captured["output"]
                previous = output.clone()
                changed = 0
                for _ in range(3):
                    captured["invoke"]()
                    torch.cuda.synchronize()
                    changed += int((output != previous).sum().item())
                    previous.copy_(output)
                for _ in range(warmup):
                    captured["invoke"]()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(iterations):
                        captured["invoke"]()
                pipe.send({"ok": row.ok, "error": row.error,
                           "max_delta": row.max_delta, "changed_elements": changed})
                del data, previous, output
            elif command["action"] == "time":
                pipe.send({"us": _event_time_us(graph.replay, 1, 1) / iterations})
    except BaseException:
        pipe.send({"exception": traceback.format_exc()})
    finally:
        pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 1024, 8192])
    parser.add_argument("--modes", type=int, nargs="+")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    profiles = [
        ("int8_w8a8", "gemm1", 10204101),
        ("int8_w8a8", "gemm1", 10204102),
        ("int8_w8a8", "gemm2", 20104202),
        ("int8_w8a8", "gemm2", 20001101),
        ("int8_w8a8", "gemm2", 20001104),
        ("fp8_w8a8", "gemm2", 20001101),
        ("fp8_w8a8", "gemm2", 20001104),
        ("int8_w4a8", "gemm1", 11204101),
        ("int8_w4a8", "gemm1", 11204102),
        ("int8_w4a8", "gemm1", 11204104),
        ("int8_w4a8", "gemm2", 21001101),
        ("int8_w4a8", "gemm2", 21004104),
        ("int8_w4a8", "gemm1", 11001104),
        ("int8_w4a8", "gemm1", 11104204),
        ("int8_w4a8", "gemm2", 21104204),
        ("int8_w4a8", "gemm1", 11201101),
        ("int8_w4a8", "gemm1", 11201102),
        ("int8_w4a8", "gemm2", 21201101),
        ("int8_w4a8", "gemm2", 21201102),
    ]
    if args.modes:
        profiles = [profile for profile in profiles if profile[2] in args.modes]
        if not profiles:
            parser.error("no benchmark profiles match --modes")
    ctx = mp.get_context("spawn")
    workers = {}

    def request(name, message):
        pipe, process = workers[name]
        pipe.send(message)
        result = pipe.recv()
        if "exception" in result:
            raise RuntimeError(result["exception"])
        return result

    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, path in (("before", args.before), ("after", args.after)):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(child, str(path.resolve()),
                                                       args.warmup, args.iterations))
            process.start()
            child.close()
            workers[name] = (parent, process)
            result = parent.recv()
            if "exception" in result:
                raise RuntimeError(result["exception"])
        for quant, stage, mode in profiles:
            for m in args.tokens:
                case = (quant, stage, mode, m)
                checks = {name: request(name, {"action": "prepare", "case": case})
                          for name in workers}
                times = {name: [] for name in workers}
                for round_i in range(args.rounds):
                    order = ("before", "after") if round_i % 2 == 0 else ("after", "before")
                    for name in order:
                        times[name].append(request(name, {"action": "time"})["us"])
                medians = {name: statistics.median(values) for name, values in times.items()}
                row = {"quant": quant, "stage": stage, "mode": mode, "m": m,
                       "e": 16, "n": 256, "k": 4096, "topk": 4,
                       "kernel_us": medians, "rounds_us": times, "checks": checks,
                       "change_percent": (medians["after"] / medians["before"] - 1) * 100}
                results.append(row)
                args.out.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    finally:
        for pipe, process in workers.values():
            if process.is_alive():
                pipe.send({"action": "stop"})
            process.join()
            pipe.close()
    return any(not r["checks"]["after"]["ok"] or
               r["checks"]["after"]["changed_elements"] for r in results)


if __name__ == "__main__":
    sys.exit(main())
