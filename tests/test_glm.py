"""GLM registration, reports, Indexer topology, and parameter accounting."""

import copy
import json
import math
from pathlib import Path
import unittest

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry
from src.core.reporter import render_report


ROOT = Path(__file__).resolve().parents[1]
FULL_LAYERS = [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46,
               50, 54, 58, 62, 66, 70, 74]
# Projection shapes: [2048, 32*128], [6144, 128], [6144, 32], plus 256 norm params.
INDEXER_PARAMS = 9_371_904
EXPERT_PARAMS = 37_748_736
# GLM_future_analysis_v1.6.xlsx, GLM-5.2-743B-4k, H71/H72 (excluding MTP).
TOTAL_PARAMS = 743_377_000_704
ACTIVE_PARAMS = 41_250_511_104

# Frozen parameter goldens from GLM_future_analysis_v1.6.xlsx, the three
# GLM-Next *-4k sheets.  H71/H72 are total/active parameters; L10, L21+L35,
# L29, L46:L51 (scaled by D4), and L62 supply the grouped counts below.
# Keep these snapshots independent of the model and of optional Excel readers.
FUTURE_CASES = {
    "GLM-Next-2T-4k": {
        "config": {"num_layers": 82, "moe_intermediate_size": 2304,
                   "n_routed_experts": 512, "n_shared_experts": 1,
                   "num_experts_per_tok": 16, "num_indexer_layers": 22},
        "total_params": 2_026_692_957_696,
        "active_params": 85_313_213_952,
        "groups": {
            ("inputlayernorm", "postattentionlayernorm"): 1_175_552,
            ("mla", "dsa"): 15_127_884_800,
            ("indexer",): 209_786_368,
            ("sharedexperts",): 3_914_072_064,
            ("routedexperts",): 2_004_004_896_768,
            ("moerouter",): 289_931_264,
        },
        "layer_counts": (3, 19, 60),  # Dense+Indexer, MoE+Indexer, MoE+shared indices.
        "layer_params": (502_317_824, 25_614_402_304, 25_604_866_560),
    },
    "GLM-Next-10T-4k": {
        "config": {"num_layers": 98, "moe_intermediate_size": 3072,
                   "n_routed_experts": 2048, "n_shared_experts": 2,
                   "num_experts_per_tok": 18, "num_indexer_layers": 26},
        "total_params": 12_888_109_906_432,
        "active_params": 148_383_365_632,
        "groups": {
            ("inputlayernorm", "postattentionlayernorm"): 1_404_928,
            ("mla", "dsa"): 18_079_667_200,
            ("indexer",): 247_929_344,
            ("sharedexperts",): 12_551_454_720,
            ("routedexperts",): 12_852_689_633_280,
            ("moerouter",): 1_394_606_080,
        },
        "layer_counts": (3, 23, 72),
        "layer_params": (502_317_824, 135_632_306_944, 135_622_771_200),
    },
    "GLM-Next-20T-4k": {
        "config": {"num_layers": 102, "moe_intermediate_size": 3584,
                   "n_routed_experts": 3072, "n_shared_experts": 1,
                   "num_experts_per_tok": 32, "num_indexer_layers": 27},
        "total_params": 23_471_278_850_816,
        "active_params": 276_190_528_256,
        "groups": {
            ("inputlayernorm", "postattentionlayernorm"): 1_462_272,
            ("mla", "dsa"): 18_817_612_800,
            ("indexer",): 257_465_088,
            ("sharedexperts",): 7_629_963_264,
            ("routedexperts",): 23_439_247_147_008,
            ("moerouter",): 2_179_989_504,
        },
        "layer_counts": (3, 24, 75),
        "layer_params": (502_317_824, 237_053_199_104, 237_043_663_360),
    },
}


class TestGLMIndexer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigParser(ROOT / "configs/glm_5_2_config.json").load()
        cls.model = ModelRegistry.build("GLM-5.2", cls.config)
        cls.report = cls.model.analyze(16, 1)

    def test_analysis_schedule_matches_reference_config(self):
        reference = json.loads((ROOT / "src/models/glm/config.json").read_text())
        self.assertEqual(self.config["indexer_types"], reference["indexer_types"])
        self.assertEqual(self.config["num_indexer_layers"], 21)

    def test_full_layers_own_indexers_including_dense_layers(self):
        rows = self.report["layers"]
        owners = [r["layer_id"] for r in rows if r["use_indexer"]]
        self.assertEqual(owners, FULL_LAYERS)
        for row in rows:
            with self.subTest(layer=row["layer_id"]):
                is_full = row["layer_id"] in FULL_LAYERS
                indexers = [m for m in row["modules"] if m["type"] == "Indexer"]
                self.assertEqual(len(indexers), int(is_full))
                self.assertEqual(row["indexer_type"], "full" if is_full else "shared")
        self.assertTrue(all(row["is_dense"] and row["use_indexer"] for row in rows[:3]))
        self.assertEqual([r["indexer_source_layer"] for r in rows[:11]],
                         [0, 1, 2, 2, 2, 2, 6, 6, 6, 6, 10])

    def test_reuse_follows_nearest_full_layer_independent_of_ffn(self):
        cfg = copy.deepcopy(self.config)
        cfg["num_layers"] = 8
        cfg["indexer_types"] = ["full", "shared", "shared", "full",
                                "shared", "shared", "full", "shared"]
        # The per-layer schedule is authoritative, even with a stale count hint.
        model = ModelRegistry.build("GLM-5.2", cfg)
        report = model.analyze(16, 1)
        self.assertEqual(model.num_indexer_layers, 3)
        self.assertEqual([r["indexer_source_layer"] for r in report["layers"]],
                         [0, 0, 0, 3, 3, 3, 6, 6])
        self.assertEqual(report["per_module_summary_excel"]["indexer"]["count"], 3)

    def test_shared_layers_do_not_duplicate_indexer_costs(self):
        summary = self.report["per_module_summary"]["Indexer"]
        excel = self.report["per_module_summary_excel"]["indexer"]
        self.assertEqual(summary, excel)
        self.assertEqual(summary["count"], 21)
        self.assertEqual(summary["params"], 21 * INDEXER_PARAMS)
        self.assertEqual(summary["bytes"], 21 * INDEXER_PARAMS * 2)
        indexers = [m for r in self.report["layers"] for m in r["modules"]
                    if m["type"] == "Indexer"]
        for key in ("params", "bytes", "flops", "act_bytes"):
            self.assertEqual(summary[key], sum(m[key] for m in indexers))

    def test_new_full_layer_adds_exactly_one_indexer(self):
        cfg = copy.deepcopy(self.config)
        cfg["indexer_types"][4] = "full"
        changed = ModelRegistry.build("GLM-5.2", cfg).analyze(16, 1)
        before = self.report
        self.assertEqual([r["indexer_source_layer"] for r in changed["layers"][3:7]],
                         [2, 4, 4, 6])
        self.assertEqual(changed["per_module_summary_excel"]["indexer"]["count"], 22)
        for key in ("total_params", "active_params_per_token"):
            self.assertEqual(changed["model_params"][key] - before["model_params"][key],
                             INDEXER_PARAMS)
        self.assertEqual(changed["memory"]["weight_memory_bytes"]
                         - before["memory"]["weight_memory_bytes"], INDEXER_PARAMS * 2)
        # Projection/norm work plus Top-K comparisons at B=1, S=16.
        self.assertEqual(changed["compute"]["total_flops_per_sample"]
                         - before["compute"]["total_flops_per_sample"],
                         2 * 16 * INDEXER_PARAMS + 16 * 2048)

    def test_missing_schedule_defaults_to_all_full_and_consistent_totals(self):
        cfg = copy.deepcopy(self.config)
        del cfg["indexer_types"]
        model = ModelRegistry.build("GLM-5.2", cfg)
        report = model.analyze(16, 1)
        self.assertEqual(model.num_indexer_layers, 78)
        self.assertEqual([r["indexer_source_layer"] for r in report["layers"]],
                         list(range(78)))
        self.assertEqual(report["per_module_summary"]["Indexer"],
                         report["per_module_summary_excel"]["indexer"])

    def test_shared_layer_requires_an_earlier_full_layer(self):
        cfg = copy.deepcopy(self.config)
        cfg["indexer_types"][0] = "shared"
        with self.assertRaisesRegex(ValueError, "no earlier full Indexer layer"):
            ModelRegistry.build("GLM-5.2", cfg)

    def test_invalid_schedules_are_rejected(self):
        for schedule in (None, {}, "full", [], ["full"] * 77, ["full"] * 79,
                         ["full"] * 77 + ["unknown"]):
            with self.subTest(schedule=schedule):
                cfg = copy.deepcopy(self.config)
                cfg["indexer_types"] = schedule
                with self.assertRaisesRegex(ValueError, "indexer_types must cover every layer"):
                    ModelRegistry.build("GLM-5.2", cfg)


class TestGLMParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigParser(ROOT / "configs/glm_5_2_config.json").load()
        cls.model = ModelRegistry.build("GLM-5.2", cls.config)
        cls.report = cls.model.analyze(4096, 1)

    def test_vocabulary_matches_workbook_and_reference_config(self):
        reference = json.loads((ROOT / "src/models/glm/config.json").read_text())
        self.assertEqual(self.config["vocab_size"], 154_880)  # D22
        self.assertEqual(self.config["vocab_size"], reference["vocab_size"])
        self.assertEqual(self.report["model_params"]["embedding_params"], 951_582_720)
        self.assertEqual(self.report["per_module_summary_excel"]["lmhead"]["params"],
                         951_582_720)

    def test_workbook_total_and_active_parameters(self):
        self.assertEqual(self.report["model_params"]["total_params"], TOTAL_PARAMS)
        self.assertEqual(self.report["model_params"]["active_params_per_token"], ACTIVE_PARAMS)
        # Existing precision policy: BF16 weights, FP32 router weights.
        self.assertEqual(self.report["model_params"]["total_weight_bytes"],
                         TOTAL_PARAMS * 2 + 117_964_800 * 2)

    def test_workbook_grouped_parameters(self):
        summary = self.report["per_module_summary_excel"]
        # Workbook L4, L10, L21+L35, L29, L42, L53, L62, L68.
        expected = {
            ("wordembedding",): 951_582_720,
            ("inputlayernorm", "postattentionlayernorm"): 958_464,
            ("mla", "dsa"): 12_871_732_224,
            ("indexer",): 196_809_984,
            ("denseffn",): 679_477_248,
            ("sharedexperts", "routedexperts"): 727_606_886_400,
            ("moerouter",): 117_964_800,
            ("outputlayernorm", "lmhead"): 951_588_864,
        }
        for keys, count in expected.items():
            with self.subTest(modules=keys):
                self.assertEqual(sum(summary[k]["params"] for k in keys), count)

    def test_every_layer_contains_pre_and_post_norm(self):
        for row in self.report["layers"]:
            with self.subTest(layer=row["layer_id"]):
                types = [m["type"] for m in row["modules"]]
                self.assertEqual(types.count("InputLayerNorm"), 1)
                self.assertEqual(types.count("PostAttentionLayerNorm"), 1)
                self.assertEqual(types[0], "InputLayerNorm")
                self.assertEqual(types[types.index("DSA") + 1], "PostAttentionLayerNorm")
                for m in row["modules"]:
                    if m["type"] in ("InputLayerNorm", "PostAttentionLayerNorm"):
                        self.assertEqual(m["params"], 6144)
        summary = self.report["per_module_summary_excel"]
        for key in ("inputlayernorm", "postattentionlayernorm"):
            self.assertEqual(summary[key]["count"], 78)
            self.assertEqual(summary[key]["params"], 479_232)
        self.assertEqual(summary["outputlayernorm"]["count"], 1)

    def test_complete_layer_parameter_counts(self):
        # Shape-derived whole layers: two norms, MLA+DSA, optional Indexer,
        # and either the Dense FFN or all 257 experts plus the router.
        for row in self.report["layers"]:
            with self.subTest(layer=row["layer_id"]):
                expected = (400_898_816 if row["is_dense"] else
                            9_877_404_416 if row["use_indexer"] else 9_868_032_512)
                self.assertEqual(row["params"], expected)
                self.assertEqual(row["params"], sum(m["params"] for m in row["modules"]))
                self.assertEqual(row["bytes"], sum(m["bytes"] for m in row["modules"]))

    def test_layer_and_global_summaries_reconcile(self):
        self.assert_report_reconciles(self.report)
        summary = self.report["per_module_summary_excel"]
        self.assertEqual(summary["routedexperts"]["count"], 75)
        self.assertEqual(summary["routedexperts"]["params"], 724_775_731_200)

    def assert_report_reconciles(self, report):
        summary = report["per_module_summary_excel"]
        for module_type, stats in report["per_module_summary"].items():
            self.assertEqual(stats, summary[module_type.lower()])
        for row_key, total_key in (("params", "total_params"), ("bytes", "total_weight_bytes")):
            top = sum(summary[k][row_key] for k in ("wordembedding", "outputlayernorm", "lmhead"))
            total = top + sum(row[row_key] for row in report["layers"])
            self.assertEqual(total, report["model_params"][total_key])
            self.assertEqual(total, sum(stats[row_key] for stats in summary.values()))

    def test_expert_module_shapes_include_all_experts(self):
        for module in self.model.layers[3]:
            if module.module_type in ("RoutedExperts", "SharedExperts"):
                with self.subTest(module=module.module_type):
                    experts = 256 if module.module_type == "RoutedExperts" else 1
                    self.assertEqual(module.params_count(), experts * EXPERT_PARAMS)
                    for param in module.compute_params():
                        self.assertEqual(param.shape[0], experts)
                        self.assertEqual(math.prod(param.shape), param.num_elements)

    def test_expert_counts_scale_once_at_layer_and_model_levels(self):
        for key, value, delta, active_delta in (
            ("n_routed_experts", 128, -128 * (EXPERT_PARAMS + 6144), -128 * 6144),
            ("n_shared_experts", 2, EXPERT_PARAMS, EXPERT_PARAMS),
        ):
            with self.subTest(config_key=key):
                cfg = copy.deepcopy(self.config)
                cfg[key] = value
                report = ModelRegistry.build("GLM-5.2", cfg).analyze(4096, 1)
                self.assertEqual(report["model_params"]["total_params"], TOTAL_PARAMS + 75 * delta)
                self.assertEqual(report["model_params"]["active_params_per_token"],
                                 ACTIVE_PARAMS + 75 * active_delta)
                for before, after in zip(self.report["layers"], report["layers"]):
                    self.assertEqual(after["params"] - before["params"],
                                     0 if before["is_dense"] else delta)
                self.assert_report_reconciles(report)
                if key == "n_routed_experts":
                    before = self.report["per_module_summary"]["RoutedExperts"]
                    after = report["per_module_summary"]["RoutedExperts"]
                    self.assertEqual(after["flops"], before["flops"])
                    self.assertEqual(after["act_bytes"], before["act_bytes"])

    def test_topk_does_not_change_stored_parameters(self):
        cfg = copy.deepcopy(self.config)
        cfg["num_experts_per_tok"] = 4
        report = ModelRegistry.build("GLM-5.2", cfg).analyze(4096, 1)
        self.assertEqual(report["model_params"]["total_params"], TOTAL_PARAMS)
        self.assertEqual(report["model_params"]["total_weight_bytes"],
                         self.report["model_params"]["total_weight_bytes"])
        self.assertEqual(report["model_params"]["active_params_per_token"],
                         ACTIVE_PARAMS - 75 * 4 * EXPERT_PARAMS)


class TestGLMFutureParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Common architecture from D2:D23 and B23/B25 of the three sheets.
        common = {
            "model": "GLM-5.2", "hidden_size": 7168, "vocab_size": 154880,
            "max_seq_len": 4096, "dtype": "bf16", "tie_word_embeddings": False,
            "first_k_dense_replace": 3, "intermediate_size": 14336,
            "num_attention_heads": 64, "num_key_value_heads": 64,
            "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_head_dim": 256,
            "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
            "index_head_dim": 128, "index_n_heads": 32, "index_topk": 2048,
            "moe_router_dtype": "fp32",
        }
        cls.models = {}
        cls.reports = {}
        for name, case in FUTURE_CASES.items():
            cfg = {**common, **case["config"]}
            # Extend the existing 743B schedule.  The workbook specifies
            # counts (R80:R82), not exact Full layer IDs; parameter totals
            # depend only on those counts, including three Dense Full layers.
            full = {0, 1, 2, *range(6, cfg["num_layers"], 4)}
            cfg["indexer_types"] = ["full" if i in full else "shared"
                                    for i in range(cfg["num_layers"])]
            cfg = ConfigParser.load_from_dict(cfg)
            cls.models[name] = ModelRegistry.build("GLM-5.2", cfg)
            cls.reports[name] = cls.models[name].analyze(4096, 1)

    def test_future_total_and_active_parameters_match_workbook(self):
        for name, case in FUTURE_CASES.items():
            with self.subTest(sheet=name):
                report = self.reports[name]
                self.assertEqual(report["model_params"]["total_params"], case["total_params"])
                self.assertEqual(report["model_params"]["active_params_per_token"],
                                 case["active_params"])

    def test_future_grouped_parameters_match_workbook(self):
        # Common L4, L42 and L68 values; MLA and DSA are compared together
        # because the workbook splits the two parts of kv_b_proj across them.
        common_groups = {
            ("wordembedding",): 1_110_179_840,
            ("denseffn",): 924_844_032,
            ("outputlayernorm", "lmhead"): 1_110_187_008,
        }
        for name, case in FUTURE_CASES.items():
            summary = self.reports[name]["per_module_summary_excel"]
            for keys, expected in {**common_groups, **case["groups"]}.items():
                with self.subTest(sheet=name, modules=keys):
                    self.assertEqual(sum(summary[k]["params"] for k in keys), expected)

    def test_future_layer_counts_and_complete_parameters(self):
        for name, case in FUTURE_CASES.items():
            with self.subTest(sheet=name):
                counts = [0, 0, 0]
                rows = self.reports[name]["layers"]
                self.assertEqual(len(rows), case["config"]["num_layers"])
                self.assertEqual(self.models[name].num_indexer_layers,
                                 case["config"]["num_indexer_layers"])
                for row in rows:
                    kind = 0 if row["is_dense"] else 1 if row["use_indexer"] else 2
                    counts[kind] += 1
                    self.assertEqual(row["params"], case["layer_params"][kind],
                                     f"{name} layer {row['layer_id']}")
                    self.assertEqual(row["params"], sum(m["params"] for m in row["modules"]))
                self.assertEqual(tuple(counts), case["layer_counts"])

    def test_future_layer_and_global_parameter_sums_reconcile(self):
        for name, case in FUTURE_CASES.items():
            with self.subTest(sheet=name):
                report = self.reports[name]
                summary = report["per_module_summary_excel"]
                for module_type, stats in report["per_module_summary"].items():
                    self.assertEqual(stats["params"], summary[module_type.lower()]["params"])
                top = sum(summary[k]["params"]
                          for k in ("wordembedding", "outputlayernorm", "lmhead"))
                self.assertEqual(sum(row["params"] for row in report["layers"]) + top,
                                 case["total_params"])
                self.assertEqual(sum(stats["params"] for stats in summary.values()),
                                 case["total_params"])


# GLM-10T / 20T / 50T series configs (configs/glm_{10t,20t,50t}_config.json).
# The golden totals are the exact model outputs; they round to the analysis
# table's 10.000T / 19.984T / 49.990T totals and 185.4B / 296.1B / 535.0B
# active parameters per token.  All attention/indexer dims and hidden_size
# stay at the GLM-5.2 baseline; only layer count, expert dims/counts, and
# the indexer schedule change.
SERIES_CASES = {
    "GLM-10T": {
        "config_path": "configs/glm_10t_config.json",
        "num_layers": 96, "num_indexer_layers": 26,
        "total_params": 9_999_772_512_768,
        "active_params": 185_360_675_328,
    },
    "GLM-20T": {
        "config_path": "configs/glm_20t_config.json",
        "num_layers": 112, "num_indexer_layers": 30,
        "total_params": 19_984_031_782_400,
        "active_params": 296_126_617_088,
    },
    "GLM-50T": {
        "config_path": "configs/glm_50t_config.json",
        "num_layers": 128, "num_indexer_layers": 34,
        "total_params": 49_989_663_341_056,
        "active_params": 534_985_325_056,
    },
}


class TestGLMSeriesParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        cls.reports = {}
        for name, case in SERIES_CASES.items():
            cfg = ConfigParser(ROOT / case["config_path"]).load()
            cls.models[name] = ModelRegistry.build("GLM-5.2", cfg)
            cls.reports[name] = cls.models[name].analyze(4096, 1)

    def test_series_total_and_active_parameters(self):
        for name, case in SERIES_CASES.items():
            with self.subTest(sheet=name):
                report = self.reports[name]
                self.assertEqual(report["model_params"]["total_params"],
                                 case["total_params"])
                self.assertEqual(report["model_params"]["active_params_per_token"],
                                 case["active_params"])

    def test_series_indexer_schedule(self):
        for name, case in SERIES_CASES.items():
            with self.subTest(sheet=name):
                model = self.models[name]
                report = self.reports[name]
                self.assertEqual(model.num_indexer_layers, case["num_indexer_layers"])
                self.assertEqual(len(report["layers"]), case["num_layers"])
                self.assertEqual([r["indexer_type"] for r in report["layers"][:4]],
                                 ["full", "full", "full", "shared"])
                self.assertTrue(all(r["is_dense"] and r["use_indexer"]
                                    for r in report["layers"][:3]))

    def test_series_layer_and_global_sums_reconcile(self):
        for name, case in SERIES_CASES.items():
            with self.subTest(sheet=name):
                report = self.reports[name]
                summary = report["per_module_summary_excel"]
                for module_type, stats in report["per_module_summary"].items():
                    self.assertEqual(stats["params"], summary[module_type.lower()]["params"])
                top = sum(summary[k]["params"]
                          for k in ("wordembedding", "outputlayernorm", "lmhead"))
                self.assertEqual(sum(row["params"] for row in report["layers"]) + top,
                                 case["total_params"])


class TestGLMIntegration(unittest.TestCase):
    def test_registry_preserves_all_model_families(self):
        self.assertEqual(ModelRegistry.list_models(),
                         ["DSV41Flash", "DSV4Pro", "GLM-5.2", "Hy4", "Kimi-10T"])

    def test_shared_report_supports_glm_layer_schema(self):
        cfg = ConfigParser(ROOT / "configs/glm_5_2_config.json").load()
        report = ModelRegistry.build("GLM-5.2", cfg).analyze(4096, 1)
        before = copy.deepcopy(report)
        text = render_report(report)
        module_text = text.split("── Per-Module Summary", 1)[1].split("── Per-Layer Summary", 1)[0]
        modules = [line.split()[0] for line in module_text.splitlines()
                   if line.split() and line.split()[0] in report["per_module_summary_excel"]]
        self.assertEqual(modules, ["wordembedding", "inputlayernorm", "mla", "indexer", "dsa",
                                   "postattentionlayernorm", "denseffn", "sharedexperts",
                                   "routedexperts", "moerouter", "outputlayernorm", "lmhead"])
        layer_text = text.split("── Per-Layer Summary", 1)[1]
        header = next(line for line in layer_text.splitlines() if line.strip().startswith("Layer"))
        self.assertNotIn("Attn", header)
        rows = [line.split() for line in layer_text.splitlines()
                if line.split() and line.split()[0].isdigit()]
        self.assertEqual([int(row[0]) for row in rows], [0, 1, 2, 3, 4, 75, 76, 77])
        self.assertEqual([row[1] for row in rows[:5]], ["Dense", "Dense", "Dense", "MoE", "MoE"])
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
