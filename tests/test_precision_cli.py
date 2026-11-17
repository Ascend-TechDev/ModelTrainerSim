import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestPrecisionCLI(unittest.TestCase):
    def run_cli(self, config, *options):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--config", config,
             "--format", "json", *options],
            cwd=ROOT, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def test_fp8_overrides_weight_and_activation_memory_for_each_family(self):
        configs = (
            "configs/dsv4pro_config.json",
            "configs/dsv41flash_config.json",
            "configs/glm_5_2_config.json",
            "configs/hy4_10t_config.json",
            "configs/kimi_10t_deep_v2_config.json",
        )
        for config in configs:
            with self.subTest(config=config):
                baseline = self.run_cli(config)
                overridden = self.run_cli(
                    config, "--weight-dtype", "fp8",
                    "--activation-dtype", "fp8")

                self.assertEqual(overridden["config"]["dtype"], "fp8")
                self.assertEqual(overridden["config"]["activation_dtype"], "fp8")
                self.assertLess(overridden["memory"]["weight_memory_bytes"],
                                baseline["memory"]["weight_memory_bytes"])
                self.assertLess(overridden["memory"]["activation_memory_bytes"],
                                baseline["memory"]["activation_memory_bytes"])

    def test_each_dtype_option_only_changes_its_own_precision(self):
        configs = (
            "configs/dsv4pro_config.json",
            "configs/dsv41flash_config.json",
            "configs/glm_5_2_config.json",
            "configs/hy4_10t_config.json",
            "configs/kimi_10t_deep_v2_config.json",
        )
        for config in configs:
            with self.subTest(config=config):
                baseline = self.run_cli(config)
                weights_only = self.run_cli(config, "--weight-dtype", "fp8")
                activations_only = self.run_cli(
                    config, "--activation-dtype", "fp8")

                self.assertEqual(
                    weights_only["memory"]["activation_memory_bytes"],
                    baseline["memory"]["activation_memory_bytes"])
                self.assertEqual(
                    activations_only["memory"]["weight_memory_bytes"],
                    baseline["memory"]["weight_memory_bytes"])
                self.assertLess(weights_only["memory"]["weight_memory_bytes"],
                                baseline["memory"]["weight_memory_bytes"])
                self.assertLess(
                    activations_only["memory"]["activation_memory_bytes"],
                    baseline["memory"]["activation_memory_bytes"])


if __name__ == "__main__":
    unittest.main()
