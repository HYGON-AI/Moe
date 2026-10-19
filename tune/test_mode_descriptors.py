"""CPU-only checks for mode routing, C++ tiles, and shape restrictions."""

import re
import unittest
from dataclasses import replace
from pathlib import Path

from tune.tune_moe_hipc import PROFILES, Shape, DirectBenchRow, _best_bm_pairs, _valid_modes_for_shape, mode
from tune.tune_moe_w4a8_perchannel import Shape as W4Shape, generate_candidates


def shape(**changes):
    return replace(Shape(256, 256, 7168, 8, 1, "bf16", "gfx938", 72), **changes)


class ModeDescriptorsTest(unittest.TestCase):
    def test_pair_selection_requires_matching_routing(self):
        rows = [DirectBenchRow(1, "gemm1", 1, 16, "ck_sorting", 1, True),
                DirectBenchRow(1, "gemm2", 2, 16, "moe_align", 1, True),
                DirectBenchRow(1, "gemm2", 3, 16, "ck_sorting", 3, True),
                DirectBenchRow(1, "gemm1", 4, 16, "moe_align", 5, True)]
        best = min(_best_bm_pairs(rows, 1), key=lambda g: g["latency_us"])
        self.assertEqual((best["gemm1"]["mode"], best["gemm2"]["mode"]), (1, 3))
        self.assertEqual(best["sorting"], "ck_sorting")
        self.assertEqual(_best_bm_pairs(rows[:2], 1), [])
        from dataclasses import asdict
        from tune.tune_existing_moe_c_configs import best_rows_from_rows
        csv_rows = [{k: str(v) for k, v in asdict(r).items()} for r in rows]
        top, bottom = best_rows_from_rows(csv_rows)
        self.assertEqual((top["1"]["MODE"], bottom["1"]["MODE"]), (1, 3))
        self.assertEqual((top["1"]["BLOCK_SIZE_M"], bottom["1"]["BLOCK_SIZE_M"]), (16, 16))
        self.assertEqual(best_rows_from_rows(csv_rows[:2]), ({}, {}))

    def test_pair_selection_requires_matching_bm_and_passed_correctness(self):
        rows = [DirectBenchRow(1, "gemm1", 1, 16, "moe_align", 1, True),
                DirectBenchRow(1, "gemm2", 2, 16, "moe_align", 10, True),
                DirectBenchRow(1, "gemm1", 3, 32, "ck_sorting", 3, True),
                DirectBenchRow(1, "gemm2", 4, 32, "ck_sorting", 2, True),
                DirectBenchRow(1, "gemm2", 5, 32, "ck_sorting", 0.1, False)]
        best = min(_best_bm_pairs(rows, 1), key=lambda g: g["latency_us"])
        self.assertEqual((best["gemm1"]["mode"], best["gemm2"]["mode"]), (3, 4))
        self.assertEqual(best["latency_us"], 5)
        self.assertEqual(_best_bm_pairs(rows, 2), [])
        from dataclasses import asdict
        from tune.tune_existing_moe_c_configs import best_rows_from_rows
        csv_rows = [{k: str(v) for k, v in asdict(r).items()} for r in rows]
        top, bottom = best_rows_from_rows(csv_rows)
        self.assertEqual((top["1"]["MODE"], bottom["1"]["MODE"]), (3, 4))

    def test_stage_goldens_use_independent_route_inputs(self):
        import torch
        from tune.tune_moe_hipc import _w8a8_stage_reference
        s = shape(e=2, n=2, k=2, m=1, topk=2)
        data = {"topk_ids": torch.tensor([[1, 0]]), "topk_weights": torch.tensor([[0.25, 0.75]]),
                "reference_w1": torch.ones(2, 4, 2), "w1_scale": torch.ones(2, 4, 1),
                "reference_w2": torch.stack((torch.eye(2), 2 * torch.eye(2))),
                "w2_scale": torch.ones(2, 2, 1)}
        up = _w8a8_stage_reference(s, data, "gemm1", torch.tensor([[2, 3]]), torch.tensor([[2]]))
        self.assertTrue(torch.equal(up, torch.full((1, 2, 4), 10.)))
        down = _w8a8_stage_reference(s, data, "gemm2", torch.tensor([[4, 6], [8, 10]]), torch.ones(2, 1))
        self.assertTrue(torch.equal(down, torch.tensor([[[2., 3.], [6., 7.5]]])))

    def test_quant_routes_same_mode_to_different_tiles(self):
        self.assertEqual(PROFILES["int8_w8a8"].descriptor(10001103).bk, 1024)
        self.assertEqual(PROFILES["fp8_w8a8"].descriptor(10001103).bk, 256)
        self.assertIsNotNone(PROFILES["int8_w8a8"].descriptor(10001103).validate(shape(k=1280)))
        self.assertIsNone(PROFILES["fp8_w8a8"].descriptor(10001103).validate(shape(k=1280)))

    def test_explicit_cpp_entries_match_static_tiles(self):
        root = Path(__file__).resolve().parents[1] / "csrc_for_aiter/936_938"
        checked = 0
        for filename, prefix in (("moe_w8a8_config.h", "W8A8"), ("moe_w4a8_config.h", "W4A8")):
            source = (root / filename).read_text()
            maps = re.findall(r"static std::unordered_map[^\n]+ (kernel_maps_\w+) = \{(.*?)\n\};", source, re.S)
            for name, body in maps:
                quant = "int8_w4a8" if prefix == "W4A8" else ("fp8_w8a8" if "fp8" in name else "int8_w8a8")
                entries = re.findall(
                    r"MOE_" + prefix + r"_MODE\((\d+), (\d+), (\d+), (\d+), (\d+)\).*?"
                    r"\{ (launch_\w+)<([\d, ]+)>", body, re.S)
                for s, f, family, bm, profile, kernel, params in entries:
                    value = mode(*map(int, (s, f, family, bm, profile)))
                    desc = PROFILES[quant].descriptor(value, shape(n=128) if "gemm1n256" in name else shape(n=256))
                    static = tuple(map(int, params.split(",")))
                    self.assertEqual(static[:7], (desc.bm, desc.bn, desc.bk, desc.wm, desc.wn, desc.wk, desc.stages), value)
                    self.assertEqual(kernel, desc.kernel, value)
                    if len(static) == 8:
                        self.assertEqual(static[7], desc.nloop, value)
                    checked += 1
        self.assertGreater(checked, 60)

    def test_nloop_changes_tile_legality(self):
        desc = PROFILES["fp8_w8a8"].descriptor(10202105)
        self.assertEqual(desc.tile_n, 640)
        self.assertIsNone(desc.validate(shape(n=320)))
        self.assertIn("tile_n=640", desc.validate(shape(n=256)))

    def test_gemm2_k_tail_is_not_bk_divisibility(self):
        desc = PROFILES["fp8_w8a8"].descriptor(20202104)
        self.assertEqual((desc.k_step, desc.k_tail), (128, 64))
        self.assertIsNone(desc.validate(shape(n=320)))
        self.assertIsNotNone(desc.validate(shape(n=352)))

    def test_fp8_prefill_down_supports_k192(self):
        for mode_id in (20204101, 20204102, 20204104, 21104101, 21104102, 21104104):
            desc = PROFILES["fp8_w8a8"].descriptor(mode_id)
            self.assertIsNone(desc.validate(shape(e=384, n=192, k=7168, topk=6, m=4096)))
            self.assertEqual((desc.bm, desc.k_step, desc.k_tail), (64, 128, 64))
        self.assertIsNotNone(PROFILES["int8_w8a8"].descriptor(20204101).validate(shape(n=192)))

    def test_sorting_families_have_different_k_whitelists(self):
        ck = PROFILES["fp8_w8a8"].descriptor(20202104)
        align = PROFILES["fp8_w8a8"].descriptor(21102104)
        self.assertEqual((ck.sorting_name, align.sorting_name), ("ck_sorting", "moe_align"))
        self.assertIsNone(ck.validate(shape(n=1536)))
        self.assertIsNotNone(align.validate(shape(n=1536)))
        with self.assertRaises(ValueError):
            PROFILES["int8_w8a8"].descriptor(11102201)

    def test_metadata_index_limits_are_distinct(self):
        ck = PROFILES["fp8_w8a8"].descriptor(10202101)
        align = PROFILES["fp8_w8a8"].descriptor(11102101)
        self.assertIsNotNone(ck.validate(shape(m=1 << 24)))
        self.assertIsNone(align.validate(shape(m=1 << 24)))
        self.assertIsNotNone(align.validate(shape(m=(1 << 31) // 8)))

    def test_decode_fixed_nloop(self):
        desc = PROFILES["int8_w4a8"].descriptor(21001101)
        self.assertEqual((desc.nloop, desc.tile_n), (4, 512))
        self.assertIsNone(desc.validate(shape(n=192)))
        self.assertIsNotNone(desc.validate(shape(k=7296)))

    def test_w4_n256_launcher_variant(self):
        profile = PROFILES["int8_w4a8"]
        special = profile.descriptor(11101201, shape(n=128))
        normal = profile.descriptor(11101201, shape(n=256))
        self.assertEqual((special.nloop, special.tile_n), (2, 256))
        self.assertEqual((normal.nloop, normal.tile_n), (4, 512))
        self.assertNotEqual(special.kernel, normal.kernel)
        self.assertIsNone(normal.validate(shape(n=256)))

    def test_w4_k192_specialization(self):
        desc = PROFILES["int8_w4a8"].descriptor(21301104)
        self.assertEqual(desc.fixed_reduce_k, 192)
        self.assertIsNone(desc.validate(shape(n=192)))
        self.assertIsNotNone(desc.validate(shape(n=384)))

    def test_recommendation_is_optional(self):
        profile = PROFILES["fp8_w8a8"]
        s = shape(m=1024)
        self.assertIn(10001102, _valid_modes_for_shape(profile, s, 1))
        self.assertNotIn(10001102, _valid_modes_for_shape(profile, s, 1, recommended_only=True))
        self.assertIsNone(profile.descriptor(10001102).validate(s))

    def test_arch_dtype_and_topk_limits(self):
        desc = PROFILES["fp8_w8a8"].descriptor(10001102)
        self.assertIsNotNone(desc.validate(shape(arch="gfx946")))
        self.assertIsNone(desc.validate(shape(arch="native")))
        self.assertIsNotNone(desc.validate(shape(dtype="fp32")))
        self.assertIsNotNone(desc.validate(shape(topk=257)))

    def test_w4_candidates_use_descriptors_without_aliases(self):
        s = W4Shape(256, 640, 7168, 8, "bf16", [1, 1024], 72)
        candidates = generate_candidates(s, 9, "gfx938")
        self.assertEqual(len(candidates), len({c.mode for c in candidates}))
        self.assertTrue(any(c.gemm == "gemm2" for c in candidates))
        for c in candidates:
            desc = PROFILES["int8_w4a8"].descriptor(c.mode, shape(n=640))
            self.assertEqual(c.sorting, 1)
            self.assertEqual((c.block_size_n, c.block_size_k, c.n_loop), (desc.bn, desc.bk, desc.nloop))
            self.assertIsNone(desc.validate(shape(n=640)))
        self.assertFalse(any((c.mode // 100_000) % 10 == 3 for c in candidates))


if __name__ == "__main__":
    unittest.main()
