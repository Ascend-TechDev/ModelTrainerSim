"""GLM-5.2 model package."""

from ...core.model_registry import ModelRegistry
from .modules.model import GLMModel

ModelRegistry.register("GLM-5.2")(GLMModel)

__all__ = ["GLMModel"]