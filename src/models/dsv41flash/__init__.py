"""DeepSeek V4.1 Flash analysis model."""
from ...core.model_registry import ModelRegistry
from .modules.model import DSV41FlashModel

ModelRegistry.register("DSV41Flash")(DSV41FlashModel)

__all__ = ["DSV41FlashModel"]
