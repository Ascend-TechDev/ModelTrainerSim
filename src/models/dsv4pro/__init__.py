"""DeepSeek V4 Pro model package."""

from ...core.model_registry import ModelRegistry
from .modules.model import DSV4ProModel

ModelRegistry.register("DSV4Pro")(DSV4ProModel)

__all__ = ["DSV4ProModel"]