"""Copy one existing up/down config pair per N into N-only defaults."""

import argparse
import json
import shutil
from pathlib import Path


def export_defaults(root):
    report = []
    for quant in ("int8_w4a8", "int8_w8a8", "fp8_w8a8"):
        for directory in sorted(root.glob("*/" + quant)):
            candidates = {}
            for up_path in sorted(directory.glob("E=*.json")):
                if "is_bottom=True" in up_path.name:
                    continue
                down_path = up_path.with_name(up_path.stem + ",is_bottom=True.json")
                if not down_path.exists():
                    continue
                up = json.loads(up_path.read_text())
                down = json.loads(down_path.read_text())
                if not up or up.keys() != down.keys():
                    continue
                if any(a["BLOCK_SIZE_M"] != down[m]["BLOCK_SIZE_M"] or
                       (int(a["MODE"]) // 1000000) % 10 !=
                       (int(down[m]["MODE"]) // 1000000) % 10
                       for m, a in up.items()):
                    continue
                fields = dict(part.split("=", 1) for part in up_path.stem.split(","))
                n = int(fields["N"])
                # Prefer broad M coverage, then the common tuning workload.
                rank = (-len(up), int(fields.get("K", 7168)) != 7168,
                        int(fields.get("topk", 8)) != 8,
                        abs(int(fields["E"]) - 256), up_path.name)
                candidates.setdefault(n, []).append((rank, up_path, down_path))
            expected = {int(dict(part.split("=", 1) for part in p.stem.split(","))["N"])
                        for p in directory.glob("E=*.json")}
            if expected != candidates.keys():
                raise ValueError("No compatible up/down pair for N: %s in %s" %
                                 (sorted(expected - candidates.keys()), directory))
            for n, pairs in sorted(candidates.items()):
                _, up_path, down_path = min(pairs, key=lambda item: item[0])
                for path, suffix in ((up_path, ""), (down_path, ",is_bottom=True")):
                    target = directory / ("N=%d,dtype=%s%s.json" % (n, quant, suffix))
                    shutil.copy2(path, target)
                report.append(dict(arch=directory.parent.name, quant=quant, n=n,
                                   source=up_path.name))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    result = export_defaults(args.config_root)
    print(json.dumps(result, indent=2))
    print("Default pairs:", len(result))
