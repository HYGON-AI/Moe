"""CPU-only checks for mode routing, C++ tiles, and shape restrictions."""

import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tune.tune_moe_hipc import PROFILES, Shape, DirectBenchRow, _best_bm_pairs, _valid_modes_for_shape, mode
from tune.tune_moe_hipc import (_append_w4a16_checkpoint, _best_w4a16_pair,
                                _read_w4a16_checkpoint, _stage_descriptors, _supervise_w4a16)
from tune.w4a16_tune_utils import W4A16BenchRow
from tune.tune_moe_w4a8_perchannel import Shape as W4Shape, generate_candidates
from moe_utils.w4a16_modes import W4A16_CSV_NAME, W4A16_CSV_QUANT, mode_specs, w4a16_unfused_mode
from tune.gen_w4a16_mode_table import PIPELINES, TABLE, render, render_pipelines


def shape(**changes):
    return replace(Shape(256, 256, 7168, 8, 1, "bf16", "gfx938", 72), **changes)


class W4A16ModeDescriptorsTest(unittest.TestCase):
    profile = PROFILES["int4_w4a16"]

    def test_catalog_is_unique_and_fused_modes_have_exact_twins(self):
        specs = mode_specs()
        ids = [mode(s.stage, 1, s.family, s.bm, s.profile) for s in specs]
        self.assertEqual(len(specs), 80)
        self.assertEqual(list(self.profile.descriptors), sorted(ids))
        self.assertEqual(sum(d.implementation == "legacy_decode" for d in self.profile.descriptors.values()), 27)
        for desc in self.profile.descriptors.values():
            if desc.template.get("silu"):
                twin = self.profile.descriptor(desc.unfused_mode)
                self.assertEqual((twin.stage, twin.family, twin.bm), (1, desc.family, desc.bm))
                self.assertEqual(twin.template, dict(desc.template, silu=False))

    def test_generated_table_and_runtime_fused_twins(self):
        self.assertEqual(TABLE.read_text(), render())
        self.assertEqual(PIPELINES.read_text(), render_pipelines())
        self.assertIsNone(w4a16_unfused_mode(11103331))
        self.assertEqual(w4a16_unfused_mode(11103431), 11103331)
        self.assertEqual(w4a16_unfused_mode(11001218), 11001118)

    def test_csv_rejects_incompatible_quant_and_wrong_cu(self):
        from moe_utils.fused_moe_c import find_csv_config, rows_from_pair, write_config_csv

        rows = rows_from_pair({1: {"MODE": 11008101}}, {1: {"MODE": 21001101}},
                              "gfx936", "int4_w4a16", 8, 256, 1024, 4, 72,
                              ["torch.float16"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / W4A16_CSV_NAME
            write_config_csv(path, rows)
            read = lambda cu: find_csv_config(directory, "gfx936", "int4_w4a16",
                                               8, 256, 1024, 4, cu, False, "torch.float16")
            self.assertEqual(read(72), {1: {"MODE": 11008101, "BLOCK_SIZE_M": 128}})
            self.assertIsNone(read(80))
            bad = path.read_text().replace(W4A16_CSV_QUANT, "int4_w4a16_legacy")
            path.write_text(bad)
            with self.assertRaisesRegex(AssertionError, "invalid quant_type"):
                read(72)
            with self.assertRaisesRegex(ValueError, "incompatible quant_type"):
                write_config_csv(path, rows)
            self.assertEqual(path.read_text(), bad)

    def test_untuned_shape_uses_legal_modes(self):
        from moe_utils.fused_moe_c import _default_w4a16_config

        for n, k in ((256, 1024), (512, 7168), (512, 2304)):
            for bottom in (False, True):
                config = _default_w4a16_config(n, k, bottom)
                descriptor = self.profile.descriptor(config["MODE"])
                self.assertEqual(descriptor.bm, config["BLOCK_SIZE_M"])
                self.assertIsNone(descriptor.validate(shape(n=n, k=k)))
        self.assertIsNone(_default_w4a16_config(256, 1152, False))

    def test_legacy_single_stage_requires_complete_packed_scale_load(self):
        desc = self.profile.descriptor(11008191)
        self.assertIsNone(desc.validate(shape(k=256)))
        self.assertIsNone(desc.validate(shape(k=1024)))
        self.assertIsNotNone(desc.validate(shape(k=768)))

    def test_group_128_skips_legacy_decode_in_both_stages(self):
        for stage, legacy, decode in ((1, 11008101, 11001112), (2, 21001101, 21001122)):
            descriptors = _stage_descriptors(self.profile, stage, [legacy, decode])
            self.assertIn(legacy, _valid_modes_for_shape(self.profile, shape(), stage, descriptors=descriptors))
            self.assertEqual(
                _valid_modes_for_shape(self.profile, shape(group_size=128), stage, descriptors=descriptors),
                [decode],
            )

    def test_decode_depth_depends_on_reduction_k(self):
        desc = self.profile.descriptor(11001218)
        for k, depth in ((1024, 4), (2048, 8)):
            workload = shape(k=k)
            self.assertIsNone(desc.validate(workload))
            self.assertEqual(desc.for_shape(workload).template["pd"], depth)
        self.assertIsNotNone(desc.validate(shape(k=1536)))

    def test_sweep_uses_profile_and_picks_different_bms(self):
        self.assertEqual({d.family for d in _stage_descriptors(self.profile, 1)}, {0, 1})
        self.assertEqual([d.mode for d in _stage_descriptors(self.profile, 1, [11001218])], [11001218])
        with self.assertRaisesRegex(ValueError, "Unknown GEMM1"):
            _stage_descriptors(self.profile, 1, [21001101])
        rows = [W4A16BenchRow(8, "gemm1", 11103331, 48, "ok", 2.0, epilogue_us=2.0),
                W4A16BenchRow(8, "gemm1", 11001218, 16, "ok", 3.0),
                W4A16BenchRow(8, "gemm2", 21102611, 32, "ok", 1.0),
                W4A16BenchRow(8, "gemm2", 21001101, 16, "mismatch")]
        best = _best_w4a16_pair(rows, 8)
        self.assertEqual((best["gemm1"]["mode"], best["gemm2"]["mode"]), (11001218, 21102611))
        self.assertEqual(best["latency_us"], 4.0)
        self.assertTrue(best["dual_routing"])

    def test_w4a16_worker_resumes_without_overwriting_completed_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sweep.jsonl"
            unrelated = Path(directory) / "other.inflight.json"
            unrelated.write_text("unrelated")
            done = W4A16BenchRow(8, "gemm1", 11001218, 16, "ok", 3.0)
            _append_w4a16_checkpoint(checkpoint, done)
            missing = dict(m=8, stage="gemm2", mode=21102611, bm=32)
            calls = 0

            def worker(cmd):
                nonlocal calls
                marker = Path(cmd[-1])
                self.assertFalse(marker.exists())
                calls += 1
                if calls == 1:
                    # First exit occurs between journal append and marker removal.
                    fields = dict(m=done.m, stage=done.stage, mode=done.mode, bm=done.bm)
                elif calls == 2:
                    fields = missing
                else:
                    return 0
                marker.write_text(json.dumps(dict(checkpoint=str(checkpoint), **fields)))
                return -11

            with patch("tune.tune_moe_hipc.subprocess.call", side_effect=worker):
                self.assertEqual(_supervise_w4a16(SimpleNamespace(fresh=False), []), 0)
            results = _read_w4a16_checkpoint(checkpoint)
            self.assertEqual(results[(8, "gemm1", 11001218)], done)
            self.assertEqual(results[(8, "gemm2", 21102611)].status, "crashed")
            self.assertEqual(len(checkpoint.read_text().splitlines()), 2)
            self.assertEqual(unrelated.read_text(), "unrelated")
            self.assertEqual(calls, 3)

    def test_prefill_checks_shape_after_dtype_and_launcher_fallbacks(self):
        desc = self.profile.descriptor(21104522)
        workload = shape(k=1024)
        self.assertEqual(desc.for_shape(workload).template["nch"], 2)
        fp16 = replace(workload, dtype="fp16")
        self.assertEqual(desc.for_shape(fp16).template["nch"], 4)
        self.assertEqual(desc.for_shape(replace(fp16, k=768)).template["nch"], 1)
        self.assertIsNotNone(desc.validate(replace(fp16, k=640)))

        bstat = self.profile.descriptor(21102611)
        self.assertEqual(bstat.for_shape(workload).implementation, "bstat")
        self.assertIsNotNone(bstat.validate(replace(workload, k=256)))
        last_bstat_m = ((1 << 31) - 2) // (workload.topk * workload.k * 2)
        self.assertEqual(bstat.for_shape(replace(workload, m=last_bstat_m)).implementation, "bstat")
        wide_output = replace(workload, m=last_bstat_m + 1)
        effective = bstat.for_shape(wide_output)
        self.assertEqual((effective.implementation, effective.bm, effective.bn), ("prefill", 32, 128))
        self.assertIn("signed int32 byte limit", effective.selection_notes[-1])
        self.assertIsNone(bstat.validate(wide_output))
        fallback = replace(workload, n=512, k=256)
        self.assertEqual(bstat.for_shape(fallback).implementation, "prefill")
        self.assertIsNone(bstat.validate(fallback))
        self.assertIsNotNone(bstat.validate(replace(fallback, n=384)))


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
