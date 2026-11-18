"""Regression coverage for the 16 rows of model_struct_input.xlsx (non-Qwen)."""

import json
import unittest
from pathlib import Path

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
# Independent workbook facts: layers, expert width, E, Top-K, shared, heads.
EXPECTED = {
    "dsv4pro": (61, 3072, 384, 6, 1, 128),
    "dsv41flash": (40, 2304, 384, 6, 1, 64),
    "dsv41next10t": (120, 3072, 2048, 18, 2, 128),
    "dsv41next20t": (120, 4096, 3072, 20, 4, 160),
    "dsv41next40t": (120, 6144, 4096, 24, 4, 192),
    "dsv41next80t": (120, 6144, 8192, 32, 6, 224),
    "hy4_preview": (78, 2048, 256, 8, 1, 64),
    "hy4_10t": (120, 3072, 2048, 18, 2, 128),
    "hy4_30t": (120, 4608, 4096, 24, 2, 160),
    "kimi_k3": (93, 3072, 896, 16, 2, 96),
    "kimi_10t_wide_v2": (93, 4096, 2048, 32, 2, 96),
    "kimi_20t": (113, 4096, 3072, 48, 2, 96),
    "kimi_50t": (128, 6144, 4096, 48, 2, 96),
    "kimi_100t": (136, 8192, 6144, 56, 4, 96),
    "glm_5_2": (78, 2048, 256, 8, 1, 64),
    "glm_10t": (100, 3072, 2048, 18, 2, 128),
}


class TestPlanningInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        cls.saved = {}
        for stem in EXPECTED:
            cfg = ConfigParser(ROOT / "configs" / f"{stem}_config.json").load()
            cls.models[stem] = ModelRegistry.build(cfg["model"], cfg)
            cls.saved[stem] = json.loads((ROOT / "reports" / f"{stem}_report.json").read_text())

    def test_six_structure_fields_match_workbook(self):
        for stem, expected in EXPECTED.items():
            with self.subTest(model=stem):
                cfg = self.models[stem].config
                if stem == "dsv4pro":
                    a = cfg["layers"]["moe"]
                    actual = (cfg["num_layers"], a["moe_intermediate_size"], a["num_experts"],
                              a["num_activated_experts"], a["num_shared_experts"], cfg["layers"]["attention"]["num_heads"])
                elif stem.startswith("dsv41"):
                    a = cfg["architecture"]
                    actual = (a["n_layers"], a["moe_inter_dim"], a["n_routed_experts"],
                              a["n_activated_experts"], a["n_shared_experts"], a["n_heads"])
                elif stem.startswith(("hy4", "kimi")):
                    a = cfg["architecture"]
                    kimi = stem.startswith("kimi")
                    actual = (a["num_hidden_layers"], a["moe_intermediate_size"], a["n_routed_experts"],
                              a["top_k" if kimi else "num_experts_per_tok"], a["n_shared_experts"],
                              a["num_heads" if kimi else "num_attention_heads"])
                else:
                    actual = tuple(cfg[k] for k in ("num_layers", "moe_intermediate_size", "n_routed_experts",
                                                   "num_experts_per_tok", "n_shared_experts", "num_attention_heads"))
                self.assertEqual(actual, expected)

    def test_reports_reproduce_parameters_and_layer_topology(self):
        for stem, model in self.models.items():
            with self.subTest(model=stem):
                saved = self.saved[stem]
                new = model.analyze(saved["compute"]["seq_len"], saved["compute"]["batch_size"])
                self.assertEqual(new["model_params"], saved["model_params"])
                self.assertEqual([(l["layer_id"], l["params"]) for l in new["layers"]],
                                 [(l["layer_id"], l["params"]) for l in saved["layers"]])

    def test_aggregate_targets_are_not_silently_forced_to_pass(self):
        for stem, saved in self.saved.items():
            with self.subTest(model=stem):
                audit = saved["input_alignment"]
                total = saved["model_params"]["total_params"] / 1e9
                active = saved["model_params"]["active_params_per_token"] / 1e9
                total_pass = abs(total / audit["total_target_b"] - 1) <= 0.05
                active_pass = abs(active / audit["active_comparison_target_b"] - 1) <= 0.05
                self.assertEqual(audit["total_within_tolerance"], total_pass)
                self.assertEqual(audit["active_within_tolerance"], active_pass)
                self.assertEqual(audit["aggregate_status"], "within_tolerance" if total_pass and active_pass else "requires_review")

    def test_hunyuan_notes_follow_the_actual_variant(self):
        for stem, top_k in (("hy4_preview", 8), ("hy4_10t", 18), ("hy4_30t", 24)):
            self.assertIn(f"Top-{top_k}", self.saved[stem]["notes"][0])


if __name__ == "__main__":
    unittest.main()
