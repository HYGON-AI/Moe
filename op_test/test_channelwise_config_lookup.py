"""Config filename precedence and legacy fallback for standalone MoE-C."""

import json
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_utils import fused_moe_c as runtime
from moe_utils import moe as dispatch


class ConfigLookupTest(unittest.TestCase):
    def test_bottom_lookup_uses_model_dim(self):
        for dtype in ("fp8_w8a8", "int8_w8a8", "int8_w4a8"):
            w1 = (4, 512, 2048 if dtype == "int8_w4a8" else 4096)
            w2 = (4, 4096, 128 if dtype == "int8_w4a8" else 256)
            with self.subTest(dtype=dtype), patch.object(
                runtime, "get_moe_configs_marlin",
                return_value={16: {"MODE": 20001102, "BLOCK_SIZE_M": 16}},
            ) as loader:
                runtime.try_get_optimal_moe_config_marlin(
                    w1, w2, 2, dtype, 16, is_bottom=True)
                self.assertEqual(loader.call_args.kwargs["K"], 4096)
                self.assertEqual(loader.call_args.kwargs["top_k"], 2)
                self.assertEqual(loader.call_args.args[1], 256)

    def test_exact_then_legacy_for_each_quant(self):
        load_runtime = inspect.unwrap(runtime.get_moe_configs_marlin)
        for dtype in ("fp8_w8a8", "int8_w8a8", "int8_w4a8"):
            for bottom in (False, True):
                with self.subTest(dtype=dtype, bottom=bottom), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    category = root / "gfx938" / dtype
                    category.mkdir(parents=True)
                    suffix = ",is_bottom=True" if bottom else ""
                    names = [
                        f"E=4,N=256,K=4096,topk=2,cu=72,dtype={dtype}{suffix}.json",
                        f"E=4,N=256,topk=2,cu=72,dtype={dtype}{suffix}.json",
                        f"E=4,N=256,K=4096,dtype={dtype}{suffix}.json",
                        f"E=4,N=256,dtype={dtype}{suffix}.json",
                    ]
                    for rank, name in enumerate(names):
                        (category / name).write_text(json.dumps({"16": {"MODE": rank, "BLOCK_SIZE_M": 16}}))
                        pair_name = name.replace(",is_bottom=True.json", ".json") if bottom else name[:-5] + ",is_bottom=True.json"
                        (category / pair_name).write_text(json.dumps({"16": {"MODE": rank, "BLOCK_SIZE_M": 16}}))
                    with patch.object(runtime, "_moe_c_config_root", return_value=str(root)), \
                         patch.object(runtime, "_get_gfx_version", return_value="gfx938"), \
                         patch.object(dispatch, "_config_root", return_value=root), \
                         patch.object(dispatch, "_gfx_version", return_value="gfx938"):
                        for rank, name in enumerate(names):
                            self.assertEqual(load_runtime(
                                4, 256, dtype, K=4096, top_k=2, num_cu=72,
                                is_bottom=bottom)[16]["MODE"], rank)
                            self.assertEqual(dispatch._find_moe_c_config(
                                4, 256, 4096, dtype, top_k=2, num_cu=72,
                                is_bottom=bottom)[16]["MODE"], rank)
                            (category / name).unlink()

    def test_relaxation_priority_and_pair_compatibility(self):
        for dtype in ("fp8_w8a8", "int8_w8a8", "int8_w4a8"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                category = root / "gfx938" / dtype
                category.mkdir(parents=True)
                names = []
                # Exact, K, CU, topk, E: remove each winner to expose the next level.
                for rank, (e, k, topk, cu) in enumerate([
                    (4, 4096, 2, 72), (4, 7168, 2, 72), (4, 4096, 2, 80),
                    (4, 4096, 3, 72), (8, 4096, 2, 72),
                ]):
                    name = f"E={e},N=256,K={k},topk={topk},cu={cu},dtype={dtype}.json"
                    names.append(name)
                    for bottom in (False, True):
                        path = category / (name[:-5] + ",is_bottom=True.json" if bottom else name)
                        path.write_text(json.dumps({"16": {
                            "MODE": (20001101 if bottom else 10001101),
                            "BLOCK_SIZE_M": 16, "winner": rank,
                        }}))
                with patch.object(runtime, "_moe_c_config_root", return_value=str(root)), \
                     patch.object(runtime, "_get_gfx_version", return_value="gfx938"), \
                     patch.object(dispatch, "_config_root", return_value=root), \
                     patch.object(dispatch, "_gfx_version", return_value="gfx938"):
                    loader = inspect.unwrap(runtime.get_moe_configs_marlin)
                    for rank, name in enumerate(names):
                        for bottom in (False, True):
                            self.assertEqual(loader(4, 256, dtype, K=4096, top_k=2,
                                num_cu=72, is_bottom=bottom)[16]["winner"], rank)
                            self.assertEqual(dispatch._find_moe_c_config(
                                4, 256, 4096, dtype, top_k=2, num_cu=72,
                                is_bottom=bottom)[16]["winner"], rank)
                        (category / name).unlink()
                        (category / (name[:-5] + ",is_bottom=True.json")).unlink()
                    self.assertIsNone(loader(4, 192, dtype, K=4096, top_k=2, num_cu=72))

    def test_nearest_k_rejects_incompatible_pairs_and_n_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            category = root / "gfx938" / "fp8_w8a8"
            category.mkdir(parents=True)
            for k, bm, routing in [(4096, 32, 0), (4160, 16, 1), (4224, 16, 0), (7168, 16, 0)]:
                name = f"E=4,N=256,K={k},topk=2,cu=72,dtype=fp8_w8a8"
                (category / (name + ".json")).write_text(json.dumps({"16": {
                    "MODE": 10001101, "BLOCK_SIZE_M": 16, "winner": k}}))
                (category / (name + ",is_bottom=True.json")).write_text(json.dumps({"16": {
                    "MODE": 20001101 + routing * 1000000, "BLOCK_SIZE_M": bm}}))
            config = runtime._find_closest_channelwise_config(
                4, 256, "fp8_w8a8", False, 4096, 2, 72, "gfx938", str(root))
            self.assertEqual(config[16]["winner"], 4224)
            self.assertIsNone(runtime._find_closest_channelwise_config(
                4, 256, "fp8_w8a8", False, 4096, 2, 72, "gfx936", str(root)))
            (category / "N=192,dtype=fp8_w8a8.json").write_text(json.dumps({"16": {
                "MODE": 10001101, "BLOCK_SIZE_M": 16}}))
            self.assertIsNone(runtime._find_closest_channelwise_config(
                4, 192, "fp8_w8a8", False, 4096, 2, 72, "gfx938", str(root)))


if __name__ == "__main__":
    unittest.main()
