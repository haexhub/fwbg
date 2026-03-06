"""Shared dependencies for the FWBG API."""
from functools import lru_cache
from pathlib import Path

from fwbg.pipeline.registry import PluginRegistry, get_registry
from fwbg.api.workspace import (  # noqa: F401 (re-exported for existing imports)
    get_strategies_dir,
    get_test_results_dir,
)


@lru_cache()
def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry with auto-discovery."""
    registry = get_registry()
    registry.auto_discover()
    return registry
