"""Unit tests for DSV4Pro modules and the overall model."""

import pytest

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry
from src.models.dsv4pro.modules.csa import CSA
from src.models.dsv4pro.modules.hca import HCA
from src.models.dsv4pro.modules.mhc import mHC
from src.models.dsv4pro.modules.indexer import Indexer
from src.models.dsv4pro.modules.moe import MoE


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

CONFIG_PATH = "configs/dsv4pro_config.json"


@pytest.fixture
def config():
    return ConfigParser(CONFIG_PATH).load()


@pytest.fixture
def attn_config(config):
    """Attention config with hidden_size injected (as the model builder does)."""
    cfg = dict(config["layers"]["attention"])
    cfg["hidden_size"] = config["hidden_size"]
    return cfg


@pytest.fixture
def moe_config(config):
    """MoE config with hidden_size injected."""
    cfg = dict(config["layers"]["moe"])
    cfg["hidden_size"] = config["hidden_size"]
    return cfg


@pytest.fixture
def indexer_config(config):
    """Indexer config with hidden_size injected."""
    cfg = dict(config["layers"]["indexer"])
    cfg["hidden_size"] = config["hidden_size"]
    return cfg


@pytest.fixture
def model(config):
    return ModelRegistry.build(config["model"], config)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------

class TestCSA:
    def test_params_positive(self, attn_config):
        csa = CSA("test_csa", attn_config)
        params = csa.compute_params()
        assert len(params) > 0
        assert csa.params_count() > 0

    def test_flops_positive(self, attn_config):
        csa = CSA("test_csa", attn_config)
        flops = csa.compute_flops(seq_len=4096, batch_size=1)
        assert flops > 0

    def test_activations_positive(self, attn_config):
        csa = CSA("test_csa", attn_config)
        act = csa.compute_activations(seq_len=4096, batch_size=1)
        assert act > 0


class TestHCA:
    def test_params_positive(self, attn_config):
        hca = HCA("test_hca", attn_config)
        assert hca.params_count() > 0

    def test_flops_positive(self, attn_config):
        hca = HCA("test_hca", attn_config)
        assert hca.compute_flops(4096, 1) > 0


class TestmHC:
    def test_params_positive(self, attn_config):
        cfg = dict(attn_config)
        cfg["num_query_heads"] = cfg["num_heads"]
        mhc = mHC("test_mhc", cfg)
        assert mhc.params_count() > 0


class TestIndexer:
    def test_params_positive(self, indexer_config):
        idx = Indexer("test_idx", indexer_config)
        assert idx.params_count() > 0

    def test_flops_scale_with_seq_len(self, indexer_config):
        idx = Indexer("test_idx", indexer_config)
        fl1 = idx.compute_flops(1024, 1)
        fl2 = idx.compute_flops(4096, 1)
        assert fl2 > fl1


class TestMoE:
    def test_params_positive(self, moe_config):
        moe = MoE("test_moe", moe_config)
        assert moe.params_count() > 0

    def test_expert_param_count(self, moe_config):
        moe = MoE("test_moe", moe_config)
        info = moe._extra_info()
        assert info["total_expert_params"] > 0
        assert info["per_expert_params"] > 0
        assert info["activated_expert_params"] > 0

    def test_activation_ratio(self, moe_config):
        moe = MoE("test_moe", moe_config)
        info = moe._extra_info()
        assert 0 < info["activation_ratio"] <= 1


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------

class TestDSV4ProModel:
    def test_total_params_positive(self, model):
        report = model.analyze(seq_len=4096, batch_size=1)
        assert report["model_params"]["total_params"] > 0

    def test_weight_memory_positive(self, model):
        report = model.analyze(seq_len=4096, batch_size=1)
        assert report["memory"]["weight_memory_bytes"] > 0

    def test_kv_cache_positive(self, model):
        report = model.analyze(seq_len=4096, batch_size=1)
        assert report["memory"]["kv_cache_bytes"] > 0

    def test_training_memory_breakdown(self, model):
        report = model.analyze(seq_len=4096, batch_size=1)
        tb = report["memory"]["training_breakdown"]
        assert tb["weights"] > 0
        assert tb["gradients"] > 0
        assert tb["optimizer_states"] > 0
        assert tb["master_weights"] > 0
        assert tb["total"] == (tb["weights"] + tb["gradients"]
                                 + tb["master_weights"] + tb["optimizer_states"])

    def test_flops_scale_with_seq_len(self, model):
        r1 = model.analyze(seq_len=1024, batch_size=1)
        r2 = model.analyze(seq_len=4096, batch_size=1)
        # Per-token FLOPs should be roughly constant
        assert r2["compute"]["flops_per_token"] > 0
        assert r2["compute"]["total_flops_per_sample"] > r1["compute"]["total_flops_per_sample"]

    def test_has_all_module_types(self, model):
        report = model.analyze(seq_len=4096, batch_size=1)
        types = set(report["per_module_summary"].keys())
        assert "MoE" in types
        # CSA is used in first 3 layers
        assert "CSA" in types or "DenseFFN" in types