"""Source-derived golden counts and accounting invariants; stdlib unittest.

Golden values are independently derived from inference/model.py tensor shapes,
not generated from the simulator's formulas. No torch or model weights required.
Run: python -m unittest tests.test_dsv41flash -v
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry
from src.core.reporter import render_report
from src.models.dsv41flash.modules.attention import Attention
from src.core.base_module import BaseModule, ModuleParamBreakdown, ModuleStats
from unittest.mock import patch
from src.models.dsv41flash.modules.report import prepare_flash_report

ROOT = Path(__file__).resolve().parents[1]


class TestDSV41Flash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigParser(ROOT / "configs/dsv41flash_config.json").load()
        cls.report = ModelRegistry.build("DSV41Flash", cls.config).analyze(4096, 1)

    def analyze(self, cfg=None, S=4096, B=1):
        return ModelRegistry.build("DSV41Flash", cfg or self.config).analyze(S, B)

    def test_registry_preserves_pro(self):
        self.assertEqual(ModelRegistry.list_models(),
                         ["DSV41Flash", "DSV4Pro", "GLM-5.2", "Hy4", "Kimi-10T"])

    def test_40_backbone_layer_types(self):
        self.assertEqual(self.report["scope"]["layer_counts"], {"SWA": 2, "Full": 4, "Reuse": 30, "Reindex": 4})
        self.assertEqual(len(self.report["layers"]), 40)

    def test_exact_attention_parameter_shapes(self):
        expected = {0: 126617408, 2: 137333184, 3: 126617408, 20: 134711744, 24: 132024128}
        for layer, count in expected.items():
            row = self.report["layers"][layer]["modules"][0]
            self.assertEqual(row["params"], count)
            wo = next(p for p in row["parameter_breakdown"] if p["name"].endswith("wo_a.weight"))
            self.assertEqual(wo["params"], 33554432)

    def test_exact_moe_residual_and_output(self):
        rows = self.report["per_module_summary"]
        self.assertEqual(rows["MoE"]["params"], 40 * 13626901248)
        self.assertEqual(rows["mHC_RMSNorm"]["params"], 40 * 993334)
        self.assertEqual(rows["WordEmbedding"]["params"], 661913600)
        self.assertEqual(rows["OutputLayer"]["params"], 661918720)

    def test_exact_total_and_bf16_bytes(self):
        self.assertEqual(self.report["model_params"]["total_params"], 748494684784)
        self.assertEqual(self.report["memory"]["weight_memory_bytes"], 2 * 748494684784)
        self.assertEqual(self.report["memory"]["training_breakdown"]["groups"]["engram"]["params"], 196613849600)

    def test_without_engram_removes_table_and_projection(self):
        cfg = copy.deepcopy(self.config)
        cfg["include_engram"] = False
        r = self.analyze(cfg)
        self.assertEqual(r["model_params"]["total_params"], 551566180464)
        self.assertNotIn("Engram", r["per_module_summary"])

    def test_dspark_weight_sharing_and_separate_compute(self):
        cfg = copy.deepcopy(self.config)
        cfg["include_dspark"] = True
        r = self.analyze(cfg)
        self.assertEqual(r["auxiliary"]["params"], 14225362530)
        self.assertEqual(r["model_params"]["total_params"], 762720047314)
        self.assertEqual(r["compute"], self.report["compute"])
        self.assertGreater(r["auxiliary"]["draft_forward_flops"], 0)
        self.assertEqual(r["per_module_summary"]["WordEmbedding"]["params"], 661913600)
        self.assertEqual(r["per_module_summary"]["DSpark"]["flops"], 0)
        self.assertEqual(sum(v["params"] for v in r["per_module_summary"].values()), r["model_params"]["total_params"])
        main_projection = next(m for m in r["auxiliary"]["modules"] if m["name"] == "DSpark input projection")
        self.assertEqual(main_projection["flops"], 2 * 3 * 5120 * 5120)

    def test_sum_invariants(self):
        r = self.report
        for key, expected in (("params", r["model_params"]["total_params"]),
                              ("bytes", r["memory"]["weight_memory_bytes"]),
                              ("flops", r["compute"]["total_forward_flops"]),
                              ("act_bytes", r["memory"]["activation_memory_bytes"]),
                              ("cache_bytes", r["memory"]["kv_cache_bytes"])):
            self.assertEqual(sum(m[key] for m in r["modules"]), expected)
            self.assertEqual(sum(m[key] for m in r["per_module_summary"].values()), expected)
        outside = [m for m in r["modules"] if m["type"] in ("WordEmbedding", "OutputLayer")]
        self.assertEqual(sum(m["flops"] for m in r["layers"]) + sum(m["flops"] for m in outside), r["compute"]["total_forward_flops"])

    def test_cache_ownership_and_exact_population(self):
        attention = [row["modules"][0] for row in self.report["layers"]]
        self.assertEqual(sum(m["details"]["owns_compressed_kv"] for m in attention), 4)
        self.assertEqual(sum(m["details"]["owns_index_k"] for m in attention), 4)
        self.assertEqual(sum(m["details"]["owns_indexer"] for m in attention), 8)
        # 40 windows + 3 half-length global/key caches + 1 full-length cache,
        # plus 3 owners' two FP32 pool-state buffers of shape [1,2,512].
        expected = 40 * 128 * 512 * 2 + (3 * 2048 + 4096) * (512 + 128) * 2 + 3 * 2 * 2 * 512 * 4
        self.assertEqual(self.report["memory"]["kv_cache_bytes"], expected)
        for layer in (3, 21, 24):
            self.assertEqual(attention[layer]["cache_bytes"], 128 * 512 * 2)

    def test_batch_scaling_and_per_token_denominator(self):
        r = self.analyze(B=2)
        self.assertEqual(r["model_params"], self.report["model_params"])
        self.assertEqual(r["compute"]["total_forward_flops"], 2 * self.report["compute"]["total_forward_flops"])
        self.assertEqual(r["compute"]["flops_per_token"], self.report["compute"]["flops_per_token"])
        self.assertEqual(r["memory"]["activation_memory_bytes"], 2 * self.report["memory"]["activation_memory_bytes"])

    def test_8k_parameters_static_unchanged(self):
        r = self.analyze(S=8192)
        self.assertEqual(r["model_params"], self.report["model_params"])
        self.assertEqual(r["memory"]["training_total_bytes"], self.report["memory"]["training_total_bytes"])
        self.assertGreater(r["compute"]["total_forward_flops"], self.report["compute"]["total_forward_flops"])

    def test_source_sensitive_fields(self):
        cfg = copy.deepcopy(self.config)
        cfg["architecture"]["window_size"] = 256
        r = self.analyze(cfg)
        self.assertGreater(r["memory"]["kv_cache_bytes"], self.report["memory"]["kv_cache_bytes"])
        self.assertGreater(r["compute"]["total_forward_flops"], self.report["compute"]["total_forward_flops"])
        cfg = copy.deepcopy(self.config)
        cfg["architecture"]["index_topk"] = 256
        self.assertLess(self.analyze(cfg)["compute"]["total_forward_flops"], self.report["compute"]["total_forward_flops"])
        cfg = copy.deepcopy(self.config)
        cfg["architecture"]["hc_mult"] = 8
        self.assertGreater(self.analyze(cfg)["model_params"]["total_params"], self.report["model_params"]["total_params"])

    def test_group_dtypes_use_elements_not_weight_bytes(self):
        cfg = copy.deepcopy(self.config)
        cfg["training"]["groups"]["expert"]["weight_dtype"] = "fp32"
        r = self.analyze(cfg)
        original = self.report["memory"]["training_breakdown"]["groups"]["expert"]
        changed = r["memory"]["training_breakdown"]["groups"]["expert"]
        self.assertEqual(changed["weights"], 2 * original["weights"])
        for key in ("gradients", "master_weights", "optimizer_states"):
            self.assertEqual(changed[key], original[key])

    def test_cpu_offload_preserves_gpu_projections(self):
        cfg = copy.deepcopy(self.config)
        cfg["training"]["groups"]["engram"]["device"] = "cpu"
        r = self.analyze(cfg)
        self.assertEqual(r["model_params"], self.report["model_params"])
        self.assertEqual(r["memory"]["activation_memory_bytes"], self.report["memory"]["activation_memory_bytes"])
        tb = r["memory"]["training_breakdown"]
        self.assertEqual(tb["per_device"]["cpu"]["weights"], 2 * 196613849600)
        self.assertEqual(tb["per_device"]["gpu"]["weights"], 2 * (551566180464 + 314654720))

    def test_momentum_has_one_state_and_freeze_keeps_weights(self):
        tb = self.report["memory"]["training_breakdown"]
        self.assertEqual(tb["groups"]["engram"]["optimizer_states"], 4 * 196613849600)
        cfg = copy.deepcopy(self.config)
        cfg["training"]["groups"]["engram"]["trainable"] = False
        group = self.analyze(cfg)["memory"]["training_breakdown"]["groups"]["engram"]
        self.assertEqual(group["weights"], 2 * 196613849600)
        self.assertEqual(group["gradients"] + group["optimizer_states"] + group["master_weights"], 0)

    def test_32_device_scenario_exact_buffer_gb(self):
        cfg = ConfigParser(ROOT / "configs/dsv41flash_32gpu_config.json").load()
        tb = self.analyze(cfg)["memory"]["training_breakdown"]
        expected_gpu = (551566180464 + 314654720) * 16 / 32
        self.assertEqual(tb["per_device"]["gpu"]["total"], expected_gpu)
        self.assertEqual(tb["per_device_gpu_static_with_buffers_bytes"], expected_gpu + 5e9)
        self.assertEqual(sorted(tb["per_device_gpu_buffers_bytes"].values()), [1e9, 2e9, 2e9])

    def test_causal_short_sequence_and_no_indexer_until_group_complete(self):
        attention = Attention("layers.2.attention", self.config, layer_id=2)
        for length in (1, 2, 3, 127, 128, 129):
            details = attention.analyze(length, 1).extra["details"]
            self.assertEqual(details["valid_window_pairs_per_sample"], sum(min(t, 128) for t in range(1, length + 1)))
            self.assertEqual(details["selected_compressed_pairs_per_sample"], sum(min(t // 2, 512) for t in range(1, length + 1)))
        self.assertNotIn("indexer_dense_reference_qk", attention.analyze(1, 1).extra["details"]["flops_breakdown"])

    def test_invalid_architecture_rejected(self):
        changes = [("n_heads", 63), ("n_activated_experts", 385),
                   ("kv_source_layers", [2, 8, 8, 20]), ("kv_source_layers", [1, 8, 14, 20]),
                   ("index_source_layers", [8, 14, 20, 24, 28, 32, 36]),
                   ("engram_num_embeddings", [10]), ("candidate_source_layer", 3), ("candidate_source_layer", 14)]
        for key, value in changes:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.config)
                cfg["architecture"][key] = value
                with self.assertRaises(ValueError):
                    self.analyze(cfg)

    def test_invalid_scenario_rejected(self):
        for key, value in (("shards", 0), ("shards", 3), ("device", "disk"), ("weight_dtype", "fp4")):
            cfg = copy.deepcopy(self.config)
            cfg["training"]["num_devices"] = 32
            cfg["training"]["groups"]["expert"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analyze(cfg)
        for S, B in ((0, 1), (-1, 1), (4096, 0)):
            with self.assertRaises(ValueError):
                self.analyze(S=S, B=B)

    def test_decimal_gb_and_json_serialization(self):
        text = render_report(self.report)
        self.assertNotIn("── Notes", text)
        self.assertIn("1,496.989370 GB", text)
        self.assertEqual(json.loads(json.dumps(self.report))["model_params"], self.report["model_params"])

    def test_report_uses_same_sections_and_columns_as_pro(self):
        cfg = ConfigParser(ROOT / "configs/dsv4pro_config.json").load()
        pro = render_report(ModelRegistry.build("DSV4Pro", cfg).analyze(4096, 1))
        flash = render_report(self.report)
        headings = ["Configuration", "Parameters", "Compute (per forward pass)",
                    "Memory", "Per-Module Summary", "Per-Layer Summary"]
        for text in (pro, flash):
            positions = [text.index("── " + heading + " ") for heading in headings]
            self.assertEqual(positions, sorted(positions))
        for prefix in ("  Module ", "  Layer "):
            self.assertEqual(next(line for line in pro.splitlines() if line.startswith(prefix)),
                             next(line for line in flash.splitlines() if line.startswith(prefix)))
        layer_text = flash.split("── Per-Layer Summary", 1)[1]
        shown = [int(line.split()[0]) for line in layer_text.splitlines()
                 if line.split() and line.split()[0].isdigit()]
        self.assertEqual(shown, [0, 1, 2, 3, 4, 24, 37, 38, 39])
        self.assertIn("Reindex", layer_text)
        self.assertIn("...", layer_text)
        self.assertNotIn("Parameter groups:", flash)

    def test_shared_report_supports_glm_layer_schema(self):
        cfg = ConfigParser(ROOT / "configs/glm_5_2_config.json").load()
        report = ModelRegistry.build("GLM-5.2", cfg).analyze(4096, 1)
        before = copy.deepcopy(report)
        text = render_report(report)
        module_text = text.split("── Per-Module Summary", 1)[1].split("── Per-Layer Summary", 1)[0]
        modules = [line.split()[0] for line in module_text.splitlines()
                   if line.split() and line.split()[0] in report["per_module_summary_excel"]]
        self.assertEqual(modules, ["wordembedding", "inputlayernorm", "mla", "indexer", "dsa",
                                   "denseffn", "sharedexperts", "routedexperts", "moerouter",
                                   "outputlayernorm", "lmhead"])
        layer_text = text.split("── Per-Layer Summary", 1)[1]
        header = next(line for line in layer_text.splitlines() if line.strip().startswith("Layer"))
        self.assertNotIn("Attn", header)
        rows = [line.split() for line in layer_text.splitlines()
                if line.split() and line.split()[0].isdigit()]
        self.assertEqual([int(row[0]) for row in rows], [0, 1, 2, 3, 4, 75, 76, 77])
        self.assertEqual([row[1] for row in rows[:5]], ["Dense", "Dense", "Dense", "MoE", "MoE"])
        self.assertEqual(report, before)

    def test_display_moe_split_conserves_all_totals_and_input(self):
        for expert_dtype in ("bf16", "fp32"):
            cfg = copy.deepcopy(self.config)
            cfg["training"]["groups"]["expert"]["weight_dtype"] = expert_dtype
            original = self.analyze(cfg)
            before = copy.deepcopy(original)
            view = prepare_flash_report(original)
            display = view["per_module_summary"]
            for metric in ("params", "bytes", "flops", "act_bytes", "cache_bytes"):
                self.assertEqual(sum(display[key][metric] for key in ("gate", "routedexpert", "sharedexpert")),
                                 original["per_module_summary"]["MoE"][metric])
                self.assertEqual(sum(row[metric] for row in display.values()),
                                 sum(row[metric] for row in original["per_module_summary"].values()))
            for key in ("gate", "routedexpert", "sharedexpert"):
                self.assertEqual(display[key]["count"], 40)
            self.assertEqual(display["sharedexpert"]["flops"], 40 * 6 * 4096 * 5120 * 2304)
            render_report(original)
            self.assertEqual(original, before)

    def test_display_batch_denominator_and_optional_sections(self):
        single = render_report(self.report)
        double = render_report(self.analyze(B=2))
        # Last four fields are per-token FLOPs and activation bytes, each with units.
        for name in ("gate", "routedexpert", "sharedexpert", "csa_full"):
            first = next(line.split() for line in single.splitlines() if line.strip().startswith(name + " "))
            second = next(line.split() for line in double.splitlines() if line.strip().startswith(name + " "))
            self.assertEqual(first[-4:-2], second[-4:-2])
        self.assertNotIn("── Per-Device Memory", single)
        cfg = ConfigParser(ROOT / "configs/dsv41flash_32gpu_config.json").load()
        distributed = render_report(self.analyze(cfg))
        self.assertIn("── Per-Device Memory (static)", distributed)
        self.assertIn("table: CPU", distributed)
        self.assertIn("1.000000 GB", distributed)
        self.assertIn("2.000000 GB", distributed)
        self.assertIn("GPU static + buffers", distributed)
        cfg = copy.deepcopy(self.config)
        cfg["include_dspark"] = True
        self.assertIn("── DSpark (separate draft pass)", render_report(self.analyze(cfg)))
        cfg["include_engram"] = False
        view = prepare_flash_report(self.analyze(cfg))
        self.assertNotIn("engram", view["per_module_summary"])

    def test_config_path_relative_to_file_and_cli_json(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "result.json"
            run = subprocess.run([sys.executable, "-B", "-m", "src.main", "--config",
                                  str(ROOT / "configs/dsv41flash_config.json"), "--seq-len", "8192",
                                  "--format", "json", "--output", str(target)],
                                 cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(target.read_text())["compute"]["seq_len"], 8192)

    def test_modules_implement_shared_api_and_independent_calculations(self):
        cfg = copy.deepcopy(self.config)
        cfg["include_dspark"] = True
        cfg["training"]["groups"]["expert"]["weight_dtype"] = "fp32"
        model = ModelRegistry.build("DSV41Flash", cfg)
        modules = [model.embedding, model.output, *model.draft_extras]
        modules += [module for layer in model.layers + model.draft_layers for module in layer]
        for module in modules:
            with self.subTest(module=module.name):
                self.assertIsInstance(module, BaseModule)
                for name in ("compute_params", "compute_flops", "compute_activations"):
                    self.assertIn(name, type(module).__dict__)
                # The shared API must remain usable independently, in either order.
                with patch.object(module, "compute_flops", side_effect=AssertionError("unexpected FLOPs call")), \
                     patch.object(module, "compute_activations", side_effect=AssertionError("unexpected activation call")):
                    params = module.compute_params()
                self.assertTrue(all(isinstance(p, ModuleParamBreakdown) for p in params))
                with patch.object(module, "compute_params", side_effect=AssertionError("unexpected params call")), \
                     patch.object(module, "compute_activations", side_effect=AssertionError("unexpected activation call")):
                    flops = module.compute_flops(129, 2)
                with patch.object(module, "compute_params", side_effect=AssertionError("unexpected params call")), \
                     patch.object(module, "compute_flops", side_effect=AssertionError("unexpected FLOPs call")):
                    activations = module.compute_activations(129, 2)
                stats = module.analyze(129, 2)
                self.assertIsInstance(stats, ModuleStats)
                self.assertEqual(stats.num_params, sum(p.num_elements for p in params))
                self.assertEqual(stats.num_bytes, sum(p.num_bytes for p in params))
                self.assertEqual(stats.num_bytes, module.params_bytes())
                self.assertEqual(stats.num_flops, flops)
                self.assertEqual(stats.num_activations_bytes, activations)
                for parameter in params:
                    size = 4 if parameter.group == "expert" else 2
                    self.assertEqual(parameter.num_bytes, parameter.num_elements * size)

    def test_module_analysis_does_not_retain_sequence_dependent_state(self):
        model = ModelRegistry.build("DSV41Flash", self.config)
        modules = [model.layers[i][0] for i in (0, 2, 3, 20, 24)]
        modules += model.layers[14][1:]
        for module in modules:
            with self.subTest(module=module.name):
                before = module.analyze(1, 1)
                module.analyze(8192, 2)
                self.assertEqual(module.analyze(1, 1), before)

    def test_pro_legacy_results_unchanged(self):
        cfg = ConfigParser(ROOT / "configs/dsv4pro_config.json").load()
        r = ModelRegistry.build("DSV4Pro", cfg).analyze(4096, 1)
        self.assertEqual(r["model_params"]["total_params"], 1573579339072)
        self.assertEqual(r["memory"]["weight_memory_bytes"], 1596163300736)
        self.assertEqual(r["compute"]["total_flops_per_sample"], 1607774349819904)


if __name__ == "__main__":
    unittest.main()
