"""Model packages.  Each sub-package exposes a top-level model class."""

# pylint: disable=import-outside-toplevel


def get_supported_models() -> list[str]:
    """Return the list of registered model family names."""
    from ..core.model_registry import ModelRegistry
    return ModelRegistry.list_models()