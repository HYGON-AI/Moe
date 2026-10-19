"""Merge independent tuning shards and check coverage against installed configs."""

import argparse
import csv
import json
from pathlib import Path

from tune_existing_moe_c_configs import (CONFIG_ROOT, discover_shapes, default_cu_for_arch,
                                         bottom_name, write_json)


def shape_key(shape):
    return (shape["quant"], shape["e"], shape["n"], shape["k"], shape["topk"],
            shape.get("cu") or default_cu_for_arch(shape["arch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, nargs="+", default=[CONFIG_ROOT])
    parser.add_argument("--arch", default="gfx938")
    args = parser.parse_args()
    expected, results = {}, {}
    for source in args.config_root:
        for quant in ("fp8_w8a8", "int8_w8a8", "int8_w4a8"):
            for shape in discover_shapes(args.arch, quant, source):
                fields = vars(shape)
                key = shape_key(fields)
                expected.setdefault(key, {"shape": fields, "m": set()})["m"].update(shape.m_values)
    for path in args.out_dir.rglob("best_summary.json"):
        data = json.loads(path.read_text())
        key = shape_key(data["shape"])
        if key in results:
            raise ValueError(f"duplicate shape summaries: {key}")
        results[key] = data
    stats, missing, selected = {}, [], []
    for key, item in sorted(expected.items()):
        shape = item["shape"]
        quant = key[0]
        stat = stats.setdefault(quant, {"shapes": 0, "expected_m": 0, "selected_m": 0})
        stat["shapes"] += 1
        stat["expected_m"] += len(item["m"])
        top, bottom = {}, {}
        summary = results.get(key, {})
        for m in sorted(item["m"]):
            best = summary.get("m_results", {}).get(str(m), {}).get("best")
            if best is None:
                missing.append({"shape": list(key), "m": m,
                                "error": summary.get("preparation_errors", {}).get(str(m), "no passing pair")})
                continue
            up, down = best["gemm1"], best["gemm2"]
            assert up["ok"] and down["ok"]
            assert up["bm"] == down["bm"] == best["bm"]
            assert up["sorting"] == down["sorting"] == best["sorting"]
            top[str(m)] = {"MODE": up["mode"], "BLOCK_SIZE_M": up["bm"]}
            bottom[str(m)] = {"MODE": down["mode"], "BLOCK_SIZE_M": down["bm"]}
            stat["selected_m"] += 1
            selected.append({"quant": quant, "e": key[1], "n": key[2], "k": key[3],
                             "topk": key[4], "cu": key[5], "m": m, "bm": up["bm"],
                             "routing": up["sorting"], "up_mode": up["mode"], "down_mode": down["mode"],
                             "up_us": up["latency_us"], "down_us": down["latency_us"],
                             "sum_us": best["latency_us"]})
        filename = f"E={key[1]},N={key[2]},K={key[3]},topk={key[4]},cu={key[5]},dtype={quant}.json"
        dest = args.out_dir / "configs" / args.arch / quant
        write_json(dest / filename, top)
        write_json(dest / bottom_name(filename), bottom)
    report = {"arch": args.arch, "stats": stats, "missing": missing,
              "objective": "independent GEMM1 + GEMM2 kernel time; same BM and routing"}
    write_json(args.out_dir / "summary.json", report)
    if selected:
        with (args.out_dir / "selected_pairs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    print(json.dumps(report, indent=2), flush=True)
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
