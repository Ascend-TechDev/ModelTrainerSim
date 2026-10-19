"""Model registry.

Maps model family names (e.g. ``"DSV4Pro"``) to their builder classes.
New model families register themselves here.

Usage:

    from ..core.model_registry import ModelRegistry

    @ModelRegistry.register("DSV4Pro")
    class DSV4ProModel:
        ...
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


class ModelRegistry:
    """Central registry for model families."""

    _builders: dict[str, type] = {}
    _loaded: bool = False

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """Decorator to register a model builder class."""
        def decorator(builder_cls: type) -> type:
            cls._builders[name] = builder_cls
            return builder_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        cls._ensure_loaded()
        if name not in cls._builders:
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available: {list(cls._builders.keys())}"
            )
        return cls._builders[name]

    @classmethod
    def build(cls, name: str, config: dict[str, Any]) -> Any:
        """Instantiate a model from its name and config."""
        builder = cls.get(name)
        return builder(config)

    @classmethod
    def list_models(cls) -> list[str]:
        cls._ensure_loaded()
        return sorted(cls._builders.keys())

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Trigger auto-discovery of model modules."""
        if cls._loaded:
            return
        # Import all model packages so that their @register decorators run
        for package in ("src.models.dsv4pro", "src.models.glm"):
            try:
                importlib.import_module(package)
            except ImportError:
                pass
        importlib.import_module("src.models.dsv41flash")
        # Future: kimi, qwen, hunyuan will be imported here
        cls._loaded = True
