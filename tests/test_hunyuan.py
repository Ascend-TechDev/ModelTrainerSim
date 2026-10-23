"""Golden counts for the Hy4-preview 10T model, derived from the reference
workbook ``Hy4-preview显存占用分析.xlsx`` (sheets: Hy4-10T / 逐层拓扑 /
缓存与口径) — independently of the simulator's formulas.

Workbook cell references are cited per test.  GiB uses the workbook's
2^30 divisor (Hy4-10T sheet B52).  Run:
    python -m unittest tests.test_hunyuan -v
"""
import copy
import unittest
from pathlib import Path

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry
from src.core.reporter import render_report

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30

# Hy4-10T sheet D5-D16 (single-instance parameter counts, column F).
PER_INSTANCE_PARAMS = {
    "wordembedding": 742391808,
    "ihc": 196618,
    "sublayernorm": 6144,
    "mla": 265685568,
    "indexerproj": 9371904,
    "indexersel": 0,
    "denseffn": 339738624,
    "moe": 130535917568,
    "ihchead": 98309,
    "finalnorm": 6144,
    "outputlayer": 742391808,
    "ihcinit": 0,
}
# Hy4-10T sheet column E (counts).
EXCEL_COUNTS = {
    "wordembedding": 1, "ihcinit": 1, "ihc": 156, "sublayernorm": 156,
    "mla": 78, "indexerproj": 21, "indexersel": 21, "denseffn": 1,
    "moe": 77, "ihchead": 1, "finalnorm": 1, "outputlayer": 1,
}
# Hy4-10T sheet F17 / F20 / A57.
TOTAL_PARAMS = 10074042194589
TRAINABLE_PARAMS = 10074042036893          # total minus router bias buffer (D19)
ACTIVE_PARAMS = 342568891037               # A57: 每 token 激活估算
# Hy4-10T sheet H17 / H21 / H22 / H24 (GiB).
WEIGHT_GIB = 18764.365826899186
GRADIENT_GIB = 18764.365533167496
OPTIMIZER_GIB = 75057.46213266999
TRAINING_TOTAL_GIB = 112586.19349273667
# Hy4-10T sheet J17 (dynamic inventory total, GiB at B=1 S=4096).
ACTIVATION_GIB = 1005.3033447265625
# Hy4-10T sheet J column per module row (GiB at B=1 S=4096).
ACTIVATION_PER_MODULE_GIB = {
    "wordembedding": 0.046875, "ihcinit": 0.1875,
    "ihc": 36.6005859375, "sublayernorm": 7.3125,
    "mla": 82.3037109375, "indexerproj": 0.70751953125,
    "indexersel": 1.96875, "denseffn": 0.46875,
    "moe": 873.76953125, "ihchead": 0.0469970703125,
    "finalnorm": 0.046875, "outputlayer": 1.84375,
}
FULL_INDEXER_LAYERS = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53,
                       57, 61, 65, 69, 73, 77]


class TestHy4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigParser(ROOT / "configs/hy4_10t_config.json").load()
        cls.report = ModelRegistry.build("Hy4", cls.config).analyze(4096, 1)

    def analyze(self, cfg=None, S=4096, B=1):
        return ModelRegistry.build("Hy4", cfg or self.config).analyze(S, B)

    # -- registration & topology -------------------------------------------------

    def test_registry(self):
        models = ModelRegistry.list_models()
        self.assertIn("Hy4", models)
        for family in ("DSV41Flash", "DSV4Pro", "GLM-5.2"):
            self.assertIn(family, models)

    def test_layer_topology(self):
        rows = self.report["layers"]
        self.assertEqual(len(rows), 78)
        # 逐层拓扑 sheet: 21 Full / 57 Shared / 1 Dense / 77 MoE.
        self.assertEqual(sum(1 for r in rows if r["use_indexer"]), 21)
        self.assertEqual(sum(1 for r in rows if r["ffn_type"] == "Dense"), 1)
        self.assertEqual(sum(1 for r in rows if r["ffn_type"] == "MoE"), 77)
        self.assertEqual([r["layer_id"] for r in rows if r["use_indexer"]],
                         FULL_INDEXER_LAYERS)
        self.assertEqual(rows[0]["ffn_type"], "Dense")
        self.assertTrue(rows[0]["use_indexer"])

    # -- parameters --------------------------------------------------------------

    def test_excel_summary_counts(self):
        excel = self.report["per_module_summary_excel"]
        self.assertEqual(set(excel), set(EXCEL_COUNTS))
        for key, count in EXCEL_COUNTS.items():
            self.assertEqual(excel[key]["count"], count, f"count mismatch: {key}")

    def test_single_instance_parameter_counts(self):
        excel = self.report["per_module_summary_excel"]
        for key, params in PER_INSTANCE_PARAMS.items():
            if key == "ihcinit":
                self.assertEqual(excel[key]["params"], 0)
                continue
            self.assertEqual(excel[key]["params"] / excel[key]["count"], params,
                             f"per-instance params mismatch: {key}")

    def test_total_and_trainable_params(self):
        p = self.report["model_params"]
        self.assertEqual(p["total_params"], TOTAL_PARAMS)          # F17
        self.assertEqual(p["trainable_params"], TRAINABLE_PARAMS)  # F20
        self.assertEqual(p["embedding_params"], 742391808)         # F5
        self.assertEqual(p["active_params_per_token"], ACTIVE_PARAMS)  # A57

    # -- training memory (workbook H17/H21/H22/H23/H24) ---------------------------

    def test_training_breakdown(self):
        tb = self.report["memory"]["training_breakdown"]
        self.assertEqual(tb["weights"], 2 * TOTAL_PARAMS)
        self.assertEqual(tb["gradients"], 2 * TRAINABLE_PARAMS)
        self.assertEqual(tb["master_weights"], 0.0)                # H23: 主权重副本 0
        self.assertEqual(tb["optimizer_states"], 8 * TRAINABLE_PARAMS)
        self.assertAlmostEqual(tb["weights"] / GIB, WEIGHT_GIB, places=9)
        self.assertAlmostEqual(tb["gradients"] / GIB, GRADIENT_GIB, places=9)
        self.assertAlmostEqual(tb["optimizer_states"] / GIB, OPTIMIZER_GIB, places=9)
        self.assertAlmostEqual(tb["total"] / GIB, TRAINING_TOTAL_GIB, places=9)
        self.assertEqual(self.report["memory"]["training_total_bytes"], tb["total"])

    # -- KV cache (缓存与口径 sheet rows 5-8) --------------------------------------

    def test_kv_cache(self):
        # MLA latent KV 78 x S x 576 x 2; Indexer K 21 x S x 128 x 2;
        # one in-flight Top-K index S x min(2048, S) x 4.
        S = 4096
        expected = 78 * S * 576 * 2 + 21 * S * 128 * 2 + S * 2048 * 4
        self.assertEqual(self.report["memory"]["kv_cache_bytes"], expected)
        S2 = 2048
        r2 = self.analyze(S=S2)
        expected2 = 78 * S2 * 576 * 2 + 21 * S2 * 128 * 2 + S2 * min(2048, S2) * 4
        self.assertEqual(r2["memory"]["kv_cache_bytes"], expected2)

    # -- FLOPs (GEMM main terms, 2*M*N*Q; Hy4-10T D-block) -------------------------

    def test_flops_per_layer_modules(self):
        S, B = 4096, 1
        # MLA per layer (rows 64-76 + DSA): exact sum of the 8 GEMMs.
        mla = 2 * B * S * 2048 * 6144 + 2 * B * S * 16384 * 2048 \
            + 2 * B * S * 576 * 6144 + 2 * B * S * 28672 * 512 \
            + 2 * 64 * B * S * S * 256 + 2 * 64 * B * S * 256 * S \
            + 2 * B * S * 16384 * 6144 + 2 * B * S * 6144 * 16384
        self.assertEqual(mla, 3275986305024)                      # ≈ 3275.987 GFLOPs
        pms = self.report["per_module_summary"]
        self.assertEqual(pms["MLA"]["flops"], 78 * mla)
        # MoE per layer (rows 102-114).
        moe = 2 * B * S * 2048 * 6144 \
            + 2 * (B * S * 64) * 6912 * 6144 \
            + 2 * (B * S * 64) * 6144 * 3456 \
            + 3 * 2 * B * S * 3456 * 6144
        self.assertEqual(moe, 34022583435264)                     # ≈ 34022.583 GFLOPs
        self.assertEqual(pms["MoE"]["flops"], 77 * moe)
        # Dense FFN (rows 95-97): gate + up + down.
        dense = 3 * 2 * B * S * 18432 * 6144
        self.assertEqual(dense, 2783138807808)                     # ≈ 2783.139 GFLOPs
        self.assertEqual(pms["DenseFFN"]["flops"], dense)
        # Indexer (rows 81-84 + 89): projection + selection GEMMs.
        idx = 2 * B * S * 4096 * 2048 + 2 * B * S * 128 * 6144 \
            + 2 * B * S * 32 * 6144 \
            + 2 * B * S * 32 * S * 128 + 2 * B * S * 1 * S * 32
        self.assertEqual(idx, 215285235712)                       # ≈ 215.285 GFLOPs
        self.assertEqual(pms["Indexer"]["flops"], 21 * idx)
        # LM head (row 132).
        self.assertEqual(pms["OutputLayer"]["flops"], 2 * B * S * 120832 * 6144)
        # iHC fn per sub-layer (row 51) and final head (row 120).
        self.assertEqual(pms["iHC"]["flops"], 156 * 2 * B * S * 8 * 24576)
        self.assertEqual(pms["IHCHead"]["flops"], 2 * B * S * 4 * 24576)

    def test_flops_scale_with_batch_and_seq(self):
        r2 = self.analyze(B=2)
        self.assertEqual(r2["compute"]["total_flops_per_sample"],
                         2 * self.report["compute"]["total_flops_per_sample"])
        r8k = self.analyze(S=8192)
        self.assertGreater(r8k["compute"]["total_flops_per_sample"],
                           self.report["compute"]["total_flops_per_sample"])

    # -- activation inventory (Hy4-10T column J) -----------------------------------

    def test_activation_inventory(self):
        excel = self.report["per_module_summary_excel"]
        for key, gib in ACTIVATION_PER_MODULE_GIB.items():
            self.assertAlmostEqual(excel[key]["act_bytes"] / GIB, gib, places=9,
                                   msg=f"activation mismatch: {key}")
        self.assertAlmostEqual(self.report["memory"]["activation_memory_bytes"] / GIB,
                               ACTIVATION_GIB, places=9)          # J17

    def test_weights_static_under_seq_and_batch(self):
        r2 = self.analyze(S=8192, B=2)
        self.assertEqual(r2["model_params"]["total_params"],
                         self.report["model_params"]["total_params"])
        self.assertEqual(r2["memory"]["training_total_bytes"],
                         self.report["memory"]["training_total_bytes"])
        self.assertEqual(r2["model_params"]["active_params_per_token"],
                         self.report["model_params"]["active_params_per_token"])

    # -- report rendering ----------------------------------------------------------

    def test_render_report(self):
        text = render_report(self.report)
        self.assertIn("Hy4", text)
        self.assertIn("moe", text)
        self.assertIn("10.07T", text)

    # -- config validation ----------------------------------------------------------

    def test_config_validation_errors(self):
        base = copy.deepcopy(self.config)
        # Top-K cannot exceed routed experts.
        cfg = copy.deepcopy(base)
        cfg["architecture"]["num_experts_per_tok"] = cfg["architecture"]["n_routed_experts"] + 1
        with self.assertRaises(ValueError):
            self.analyze(cfg)
        # Full Indexer layer IDs must be valid.
        cfg = copy.deepcopy(base)
        cfg["architecture"]["full_indexer_layers"] = list(FULL_INDEXER_LAYERS) + [99]
        with self.assertRaises(ValueError):
            self.analyze(cfg)
        # Exactly one dense layer (workbook topology).
        cfg = copy.deepcopy(base)
        cfg["architecture"]["dense_layers"] = [0, 1]
        with self.assertRaises(ValueError):
            self.analyze(cfg)
        # Missing required architecture key (self-contained config, no
        # pinned base file to fill the gap).
        cfg = copy.deepcopy(base)
        cfg.pop("architecture_file", None)
        del cfg["architecture"]["hidden_size"]
        with self.assertRaises(ValueError):
            self.analyze(cfg)


if __name__ == "__main__":
    unittest.main()
