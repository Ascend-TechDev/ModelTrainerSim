"""Golden counts for the Kimi 10T model, derived from the reference workbook
``10T Kimi 显存建模.xlsx`` (sheets 10T-深度-v2 / 10T-宽度-v2) —
independently of the simulator's formulas.

Workbook cell references are cited per test:
  * Param totals follow the I-column formula chain
    I55 = I2+I3+I4+I13*D25+I18*D5+I32*D6+I51*D24+I34+I35 with
    I13 = SUM(I6:I12)+I71.  Deep-v2 = 9,283,004,226,880;
    Wide-v2 = 11,917,686,210,656.  (The cached J55 on 深度-v2 predates
    the v2 MLA g_proj 补档 and is stale — the formula is authoritative.)
  * FLOPs follow the R-column GEMMs at mbs.
  * Activation tensors follow the N-column element counts.
  * Training states: AdamW 12 / Muon 8 B per param (C27/C28);
    per-rank sharding per rows G63/G68/K66.

Run: python -m unittest tests.test_kimi -v
"""
import copy
import unittest
from pathlib import Path

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry
from src.core.reporter import render_report

ROOT = Path(__file__).resolve().parents[1]

# 10T-深度-v2 (formula chain; the sheet's cached J55 is stale).
DEEP_TOTAL = 9283004226880
# 10T-宽度-v2 (cached I55 matches the formula incl. g_proj).
WIDE_TOTAL = 11917686210656
GIB = 2 ** 30

# Single-instance params (I-column): I2/I13/I18/I32/I51/I35.
EMB_PARAMS = 163840 * 7168                      # I2 = I35 = D19*D9
MLA_PARAMS = (7168 * 1536 + 1536 + 7168 * 576 + 512 + 1536 * 96 * 192
              + 512 * 96 * 256 + 96 * 128 * 7168 + 7168 * 96 * 128)   # I13 (incl. I71)
KDA_PARAMS = (3 * 7168 * 96 * 128 + 3 * 96 * 128 * 4 + 96 + 7168 * 128
              + 128 * 96 * 128 + 96 * 128 + 7168 * 96 + 7168 * 96 * 128
              + 128 + 96 * 128 * 7168)          # I51
DENSE_PARAMS = 3 * 7168 * 33792                 # I18
DEEP_MOE_PARAMS = (2048 * 3 * 3584 * 3072 + 6 * 7168 * 3072
                   + 2048 * 7168 + 2 * 7168 * 3584)   # I32 (deep)
WIDE_MOE_PARAMS = (2048 * 3 * 5120 * 4096 + 6 * 7168 * 4096
                   + 2048 * 7168 + 2 * 7168 * 5120)   # I32 (wide)


class TestKimi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigParser(ROOT / "configs/kimi_10t_deep_v2_config.json").load()
        cls.report = ModelRegistry.build("Kimi-10T", cls.config).analyze(16384, 1)
        cls.wide_config = ConfigParser(ROOT / "configs/kimi_10t_wide_v2_config.json").load()
        cls.wide_report = ModelRegistry.build("Kimi-10T", cls.wide_config).analyze(4096, 1)

    def analyze(self, cfg=None, S=16384, B=1):
        return ModelRegistry.build("Kimi-10T", cfg or self.config).analyze(S, B)

    # -- registration & topology -------------------------------------------------

    def test_registry(self):
        models = ModelRegistry.list_models()
        self.assertIn("Kimi-10T", models)
        for family in ("DSV41Flash", "DSV4Pro", "GLM-5.2", "Hy4"):
            self.assertIn(family, models)

    def test_deep_topology(self):
        rows = self.report["layers"]
        self.assertEqual(len(rows), 137)
        self.assertEqual(rows[0]["ffn_type"], "Dense")
        self.assertEqual(rows[0]["attention_type"], "None")
        self.assertEqual(sum(1 for r in rows if r["attention_type"] == "MLA"), 34)
        self.assertEqual(sum(1 for r in rows if r["attention_type"] == "KDA"), 102)
        self.assertEqual(sum(1 for r in rows if r["ffn_type"] == "MoE"), 136)

    def test_wide_topology(self):
        rows = self.wide_report["layers"]
        self.assertEqual(len(rows), 93)
        # 宽度-v2: 24 MLA (incl. the Dense layer) + 69 KDA.
        self.assertEqual(rows[0]["ffn_type"], "Dense")
        self.assertEqual(rows[0]["attention_type"], "MLA")
        self.assertEqual(sum(1 for r in rows if r["attention_type"] == "MLA"), 24)
        self.assertEqual(sum(1 for r in rows if r["attention_type"] == "KDA"), 69)

    # -- parameters (I-column formula chain) ---------------------------------------

    def test_single_instance_params(self):
        excel = self.report["per_module_summary_excel"]
        self.assertEqual(excel["wordembedding"]["params"] / excel["wordembedding"]["count"],
                         EMB_PARAMS)
        self.assertEqual(excel["mla"]["params"] / excel["mla"]["count"], MLA_PARAMS)
        self.assertEqual(excel["kda"]["params"] / excel["kda"]["count"], KDA_PARAMS)
        self.assertEqual(excel["denseffn"]["params"], DENSE_PARAMS)
        self.assertEqual(excel["moe"]["params"] / excel["moe"]["count"], DEEP_MOE_PARAMS)
        self.assertEqual(excel["outputlayer"]["params"], EMB_PARAMS)
        for key in ("prenorm", "postnorm", "finalnorm"):
            self.assertEqual(excel[key]["params"], 7168)      # I3/I4/I34

    def test_deep_total_params(self):
        # I55 = I2+I3+I4+I13*D25+I18*D5+I32*D6+I51*D24+I34+I35
        expected = (EMB_PARAMS + 7168 + 7168 + MLA_PARAMS * 34 + DENSE_PARAMS
                    + DEEP_MOE_PARAMS * 136 + KDA_PARAMS * 102 + 7168 + EMB_PARAMS)
        self.assertEqual(self.report["model_params"]["total_params"], expected)
        self.assertEqual(self.report["model_params"]["total_params"], DEEP_TOTAL)
        # Static memory J55 = I55*2/2^30 (bf16).
        self.assertEqual(self.report["memory"]["weight_memory_bytes"], 2 * DEEP_TOTAL)
        # J55 口径 = I55*2/2^30（sheet 的缓存值是加 g_proj 前的旧值，公式为准）
        self.assertAlmostEqual(2 * DEEP_TOTAL / GIB, 17290.942793488503, places=9)

    def test_wide_total_params(self):
        self.assertEqual(self.wide_report["model_params"]["total_params"], WIDE_TOTAL)
        self.assertEqual(self.wide_report["memory"]["weight_memory_bytes"], 2 * WIDE_TOTAL)
        self.assertAlmostEqual(2 * WIDE_TOTAL / GIB, 22198.420410335, places=6)
        excel = self.wide_report["per_module_summary_excel"]
        self.assertEqual(excel["moe"]["params"] / excel["moe"]["count"], WIDE_MOE_PARAMS)
        self.assertEqual(excel["mla"]["count"], 24)
        self.assertEqual(excel["kda"]["count"], 69)
        self.assertEqual(excel["moe"]["count"], 92)

    # -- FLOPs (R-column GEMMs at mbs) ----------------------------------------------

    def test_flops_mla_kda_dense_moe(self):
        S, B = 16384, 1
        mla = (2 * B * S * 1536 * 7168 + 2 * B * S * 96 * 192 * 1536
               + 2 * B * S * 576 * 7168 + 2 * B * S * 96 * 256 * 512
               + 2 * B * 96 * S * S * 192 + 2 * B * 96 * S * S * 192
               + 2 * B * S * 7168 * 96 * 128 + 2 * B * S * 96 * 128 * 7168)
        kda = (3 * 2 * B * S * 7168 * 96 * 128 + 2 * B * S * 7168 * 128
               + 2 * B * S * 128 * 96 * 128 + 2 * B * S * 7168 * 96
               + 2 * B * S * 7168 * 96 * 128 + 2 * B * S * 96 * 128 * 7168)
        moe = (2 * B * S * 7168 * 3584 + 2 * B * S * 16 * 3584 * 3072 * 2
               + 2 * B * S * 16 * 3072 * 3584 + 2 * B * S * 3584 * 7168
               + 2 * B * S * 7168 * 3072 * 2 * 2 + 2 * B * S * 2 * 3072 * 7168)
        pms = self.report["per_module_summary"]
        self.assertEqual(pms["MLA"]["flops"], 34 * mla)
        self.assertEqual(pms["KDA"]["flops"], 102 * kda)
        self.assertEqual(pms["MoE"]["flops"], 136 * moe)
        self.assertEqual(pms["DenseFFN"]["flops"], 3 * 2 * B * S * 7168 * 33792)
        self.assertEqual(pms["OutputLayer"]["flops"], 2 * B * S * 163840 * 7168)
        # Cluster flops scale by gbs/mbs (S column uses gbs=32).
        self.assertEqual(self.report["compute"]["total_flops_at_gbs"],
                         32 * self.report["compute"]["total_flops_per_sample"])

    # -- activations (N-column element counts) ---------------------------------------

    def test_activation_tensors(self):
        S, B = 16384, 1
        a = 2.0
        # N-column tensors: q_a/q_norm (N6/N7), kv_a (N8), kv_norm (N9),
        # q_b (N10), kv_b (N11), residual (N12), PV input (N14; QK N13
        # excluded per M32), g_proj/gate (N71/N72), cats (N73), o_out (N74).
        mla = (B * S * 1536 * a * 2 + B * S * 576 * a + B * S * 512 * a
               + B * S * 96 * 192 * a + B * S * 96 * 256 * a + B * S * 7168 * a
               + B * S * 96 * 192 * a + B * S * 96 * 128 * a * 2
               + B * S * 96 * 192 * 3 * a + B * S * 7168 * a)
        # N37-42 (qkv out + conv out), N44 (f_a), N45 (f_b), N47 (b),
        # N48/N49 (g, norm), N50 (o_out), N76/N77 (silu3 + l2norm2),
        # N78 (b silu), N79/N80 (gated, kda_out).
        kda = (6 * B * S * 96 * 128 * a + B * S * 128 * a + B * S * 96 * 128 * a
               + B * S * 96 * a * 2 + B * S * 96 * 128 * a * 2 + B * S * 7168 * a
               + B * S * 96 * 128 * 5 * a + B * S * 96 * 128 * 2 * a)
        # N20-N29 minus N26 (router_expert_up excluded per M30).
        moe = (B * S * 3584 * a + B * S * 16 * 3584 * a + B * S * 16 * 3072 * 2 * a
               + B * S * 16 * 3072 * a + B * S * 16 * 3584 * a + B * S * 3584 * a
               + B * S * 3072 * 2 * 2 * a + B * S * 3072 * 2 * a)
        pms = self.report["per_module_summary"]
        self.assertEqual(pms["MLA"]["act_bytes"], 34 * mla)
        self.assertEqual(pms["KDA"]["act_bytes"], 102 * kda)
        self.assertEqual(pms["MoE"]["act_bytes"], 136 * moe)
        # Logits FP32 (M35): s*b*v*4.
        self.assertEqual(pms["OutputLayer"]["act_bytes"], B * S * 163840 * 4.0)

    # -- training memory & sharding (C27/C28, G63/G68/K66) ---------------------------

    def test_training_breakdown(self):
        tb = self.report["memory"]["training_breakdown"]
        self.assertEqual(tb["weights"], 2 * DEEP_TOTAL)
        self.assertEqual(tb["gradients"], 2 * DEEP_TOTAL)
        self.assertEqual(tb["master_weights"], 0.0)
        # AdamW 12 B/param on embedding+head; Muon 8 B/param elsewhere.
        adamw = 2 * EMB_PARAMS
        self.assertEqual(tb["groups"]["adamw"]["optimizer_states"], adamw * 12)
        self.assertEqual(tb["groups"]["muon"]["optimizer_states"],
                         (DEEP_TOTAL - adamw) * 8)
        self.assertEqual(tb["total"],
                         tb["weights"] + tb["gradients"] + tb["optimizer_states"])

    def test_per_rank_sharding(self):
        r = self.report["memory"]["per_rank"]
        # Dense (shard=tp) /TP + routed experts (shard=ep) /EP, bf16; grad 2B.
        params = []
        model = ModelRegistry.build("Kimi-10T", self.config)
        for modules in [model.top_modules] + model.layers:
            for m in modules:
                params.extend(m.compute_params())
        tp_bytes = sum(p.num_elements * 2 for p in params if p.shard == "tp")
        ep_bytes = sum(p.num_elements * 2 for p in params if p.shard == "ep")
        expected_w = tp_bytes / 1 + ep_bytes / 256
        self.assertAlmostEqual(r["weights"], expected_w, places=6)
        self.assertEqual(r["gradients"],
                         (sum(p.num_elements for p in params if p.shard == "tp") / 1
                          + sum(p.num_elements for p in params if p.shard == "ep") / 256) * 2)
        # Optimizer states divided by the total card count (row K66).
        tb = self.report["memory"]["training_breakdown"]
        self.assertAlmostEqual(r["optimizer_states"], tb["optimizer_states"] / 1024, places=6)
        self.assertEqual(r["total"], r["weights"] + r["gradients"]
                         + r["master_weights"] + r["optimizer_states"])

    def test_kv_cache(self):
        # MLA latent KV: kv_lora + rope per position, bf16, on MLA layers only.
        S = 16384
        self.assertEqual(self.report["memory"]["kv_cache_bytes"], 34 * S * (512 + 64) * 2)

    # -- report & validation ---------------------------------------------------------

    def test_render_report(self):
        text = render_report(self.report)
        self.assertIn("Kimi-10T", text)
        self.assertIn("kda", text)

    def test_config_validation_errors(self):
        base = copy.deepcopy(self.config)
        cfg = copy.deepcopy(base)
        cfg["architecture"]["top_k"] = cfg["architecture"]["n_routed_experts"] + 1
        with self.assertRaises(ValueError):
            self.analyze(cfg)
        cfg = copy.deepcopy(base)
        cfg["architecture"]["mla_layers"] += 2      # 36+102 既不等于 136 也不等于 137
        with self.assertRaises(ValueError):
            self.analyze(cfg)
        cfg = copy.deepcopy(base)
        cfg["parallelism"]["world_size"] = 1000     # not divisible by tp*ep*pp*cp
        with self.assertRaises(ValueError):
            self.analyze(cfg)
        cfg = copy.deepcopy(base)
        cfg.pop("architecture_file", None)
        del cfg["architecture"]["hidden_size"]
        with self.assertRaises(ValueError):
            self.analyze(cfg)


if __name__ == "__main__":
    unittest.main()
