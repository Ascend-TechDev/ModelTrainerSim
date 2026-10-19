"""Text-path block inventory derived from V4.1-Flash ``inference/model.py``.

Shapes describe logical (unquantized, unsharded) parameters. Quantization scales,
packed checkpoint storage, vision tower, image aligner and image delimiters are
outside this inventory. FLOPs count forward GEMMs with one multiply-add = 2.
Activation bytes are an explicit retained-tensor budget, NOT an observed peak or
an assertion about PyTorch autograd's saved tensors. In-place kernels, aliasing,
checkpointing and communication can change the actual memory requirement.
"""

from math import prod

from ....core.base_module import BaseModule, ModuleStats
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown, element_bytes


def _positive(value, name):
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dimensions(arch, S, B):
    return (
        _positive(arch["dim"], "dim"),
        _positive(S, "sequence length"),
        _positive(B, "batch size"),
    )


def _term(name, shape, element_bytes):
    count = 1
    for dimension in shape:
        count *= dimension
    if element_bytes <= 0:
        raise ValueError("element_bytes must be positive")
    return {
        "name": name,
        "shape": list(shape),
        "element_bytes": element_bytes,
        "bytes": count * element_bytes,
    }


class FlashModule(BaseModule):
    """Use the shared module API with Flash parameter groups and cache details.

    Concrete modules implement params, FLOPs and activations independently.
    ``analyze`` only combines those results through the existing BaseModule API.
    """

    block_accounting = False

    def __init__(self, name: str, config: dict, *, layer_id=None, dspark=False):
        super().__init__(name, config)
        self.arch = config["architecture"]
        self.layer_id = layer_id
        self.dspark = dspark
        self.activation_element_bytes = element_bytes(config.get("activation_dtype", "bf16"))
        self.cache_element_bytes = element_bytes(config.get("cache_dtype", "bf16"))
        self.logits_element_bytes = element_bytes(config.get("logits_dtype", "fp32"))

    def parameter(self, name: str, shape: tuple[int, ...], group="dense") -> GroupedParamBreakdown:
        group_config = self.config.get("training", {}).get("groups", {}).get(group, {})
        dtype = DType(group_config.get("weight_dtype", self.config.get("dtype", "bf16")))
        return GroupedParamBreakdown(name, shape, prod(shape), dtype, group)

    def compute_cache_bytes(self, seq_len: int, batch_size: int = 1) -> float:
        return 0.0

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict:
        return {}

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict]:
        return []

    def analyze(self, seq_len: int = 4096, batch_size: int = 1) -> ModuleStats:
        _dimensions(self.arch, seq_len, batch_size)
        stats = super().analyze(seq_len, batch_size)
        details = self.compute_details(seq_len, batch_size)
        if self.block_accounting:
            details["activation_terms"] = self.compute_activation_terms(seq_len, batch_size)
            details["activation_policy"] = (
                "Explicit retained-tensor budget; not peak memory. Component inputs may "
                "alias neighbouring outputs; aggregate estimates intentionally do not "
                "deduplicate these boundary buffers or model lifetime overlap."
            )
            details["flops_scope"] = (
                "Forward GEMMs only (multiply-add=2); normalization, activation, "
                "routing/top-k, hashing, permutation, sampling and scalar arithmetic omitted."
            )
        stats.extra = {"details": details, "cache_bytes": self.compute_cache_bytes(seq_len, batch_size)}
        return stats
