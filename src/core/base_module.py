"""Abstract base class for all model modules.

Every module type (CSA, HCA, mHC, Indexer, MoE, etc.) inherits from
``BaseModule`` and implements ``compute_params`` and ``compute_flops``.

The design separates *parameter counting* from *memory accounting* so that
the same module can be evaluated under different precision / training-mode
configurations without re-instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .dtype import DType, get_dtype_size


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ModuleParamBreakdown:
    """Fine-grained parameter breakdown for a single sub-component."""

    name: str
    shape: tuple[int, ...]
    num_elements: int
    dtype: DType = DType.BF16

    @property
    def num_bytes(self) -> float:
        return self.num_elements * get_dtype_size(self.dtype)


@dataclass
class ModuleStats:
    """Aggregated statistics returned by a module's ``analyze`` call."""

    module_type: str
    num_params: int = 0
    num_bytes: float = 0.0          # weight bytes (at training precision)
    num_flops: float = 0.0          # FLOPs per forward pass (per token / per sample)
    num_activations_bytes: float = 0.0  # activation memory (forward)
    param_breakdown: list[ModuleParamBreakdown] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, other: "ModuleStats") -> "ModuleStats":
        self.num_params += other.num_params
        self.num_bytes += other.num_bytes
        self.num_flops += other.num_flops
        self.num_activations_bytes += other.num_activations_bytes
        self.param_breakdown.extend(other.param_breakdown)
        return self

    def __repr__(self) -> str:
        from .formatter import format_bytes, format_flops, format_number
        return (
            f"ModuleStats({self.module_type}:\n"
            f"  params={format_number(self.num_params)}\n"
            f"  weight_mem={format_bytes(self.num_bytes)}\n"
            f"  flops/token={format_flops(self.num_flops)}\n"
            f"  act_mem={format_bytes(self.num_activations_bytes)}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Base module
# ---------------------------------------------------------------------------

class BaseModule(ABC):
    """Abstract base for all architectural modules.

    Subclasses must implement :meth:`compute_params` and :meth:`compute_flops`.
    Optionally override :meth:`compute_activations` for activation memory.
    """

    #: Module type identifier (e.g. ``"CSA"``, ``"MoE"``)
    module_type: str = "Base"

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config

    # -- abstract -----------------------------------------------------------

    @abstractmethod
    def compute_params(self) -> list[ModuleParamBreakdown]:
        """Return a list of parameter breakdowns for this module."""
        ...

    @abstractmethod
    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        """Return FLOPs for a single forward pass."""
        ...

    # -- optional overrides -------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Return activation memory in bytes for the forward pass.

        Default implementation returns 0; subclasses override as needed.
        """
        return 0.0

    # -- public API ---------------------------------------------------------

    def params_count(self) -> int:
        return sum(b.num_elements for b in self.compute_params())

    def params_bytes(self) -> float:
        return sum(b.num_bytes for b in self.compute_params())

    def analyze(self, seq_len: int = 4096, batch_size: int = 1) -> ModuleStats:
        """Convenience: build a full :class:`ModuleStats` for this module."""
        breakdown = self.compute_params()
        return ModuleStats(
            module_type=self.module_type,
            num_params=sum(b.num_elements for b in breakdown),
            num_bytes=sum(b.num_bytes for b in breakdown),
            num_flops=self.compute_flops(seq_len, batch_size),
            num_activations_bytes=self.compute_activations(seq_len, batch_size),
            param_breakdown=breakdown,
            extra=self._extra_info(),
        )

    # -- hooks --------------------------------------------------------------

    def _extra_info(self) -> dict[str, Any]:
        """Subclasses can override to add extra metadata to ``ModuleStats``."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, type={self.module_type})"