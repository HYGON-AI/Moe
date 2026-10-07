#!/usr/bin/env python3
"""Export legacy FP8 channelwise JSON configs to the HIPC CSV format.

The generated mode uses ``S F K MM VVV``:
stage, sorting, kernel family, BLOCK_SIZE_M / 16 and profile ID. The profile
descriptor owns the variant, N_LOOP and other static attributes. The
exporter intentionally skips legacy n160/kgrain64 IDs; those profiles must be
retuned with a normal FP8 launcher before entering the new namespace.
"""

import argparse
import csv
import json
from pathlib import Path


def encode(stage, sorting, family, bm, profile):
    if bm <= 0 or bm % 16 != 0 or bm // 16 >= 100:
        raise ValueError(f"BM must be a positive multiple of 16: {bm}")
    if not 1 <= profile <= 999:
        raise ValueError(f"profile ID must fit three decimal digits: {profile}")
    return (stage * 10_000_000 + sorting * 1_000_000 +
            family * 100_000 + (bm // 16) * 1_000 + profile)


def legacy_to_new(mode, stage):
    mode = int(mode)
    if 60000 <= mode < 70000:
        suffix = mode - 60000
        # 60098/60121 are the two ordinary GEMM1 decode profiles; 60038,
        # 60042, 60043 and 60046 are the ordinary GEMM2 profiles.
        decode_profiles = {
            (1, 98): (16, 101), (1, 121): (16, 102),
            (2, 38): (16, 101), (2, 42): (16, 102),
            (2, 43): (16, 103), (2, 46): (16, 104),
        }
        if (stage, suffix) not in decode_profiles:
            return None
        bm, profile = decode_profiles[(stage, suffix)]
        return encode(stage, 0, 0, bm, profile)
    if 80000 <= mode < 90000:
        suffix = mode - 80000
        bm, nloop = divmod(suffix, 10)
        if bm not in {32, 48, 64, 80, 96, 128} or nloop not in {1, 2, 3, 4, 5}:
            return None
        return encode(stage, 0, 1, bm, 100 + nloop)
    if 100000 <= mode < 110000:
        suffix = mode - 100000
        bm, nloop = divmod(suffix, 10)
        if bm not in {32, 48, 64} or nloop not in {1, 2, 4, 5}:
            return None
        return encode(stage, 0, 2, bm, 100 + nloop)
    return None


def load(path):
    with path.open() as f:
        return json.load(f)


def default_cu_for_arch(arch):
    if arch == "gfx938":
        return 72
    if arch == "gfx936":
        return 80
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--quant-type", default="fp8_w8a8_channel")
    parser.add_argument("--model-dim", type=int, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--cu", "--num-cu", dest="cu", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cu = args.cu if args.cu is not None else default_cu_for_arch(args.arch)

    rows = {}
    for path in sorted(args.config_dir.glob("E=*,dtype=fp8_w8a8.json")):
        bottom_path = path.with_name(path.stem + ",is_bottom=True.json")
        if not bottom_path.exists():
            continue
        name = path.name
        fields = dict(part.split("=", 1) for part in name[:-5].split(",") if "=" in part)
        e, n = int(fields["E"]), int(fields["N"])
        up, down = load(path), load(bottom_path)
        for token in sorted(set(up) & set(down), key=int):
            up_mode = legacy_to_new(up[token]["MODE"], 1)
            down_mode = legacy_to_new(down[token]["MODE"], 2)
            if up_mode is None or down_mode is None:
                continue
            key = (e, n, args.model_dim, int(token))
            rows[key] = {
                "arch": args.arch,
                "quant_type": args.quant_type,
                "indtype": "fp8",
                "token": int(token),
                "inter_dim": n,
                "model_dim": args.model_dim,
                "expert": e,
                "topk": args.topk,
                "cu": "" if cu is None else cu,
                "q_size_n": n,
                "q_size_k": args.model_dim,
                "sol_type": "moe_c",
                "sol_id": f"{up_mode}+{down_mode}",
                "time_us": "",
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arch", "quant_type", "indtype", "token", "inter_dim",
              "model_dim", "expert", "topk", "cu", "q_size_n", "q_size_k",
              "sol_type", "sol_id", "time_us"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows.values())
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
