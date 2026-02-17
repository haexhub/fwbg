"""Shared dependencies for the FWBG API."""
import os
from functools import lru_cache
from pathlib import Path

from fwbg.pipeline.registry import PluginRegistry, get_registry


@lru_cache()
def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry with auto-discovery."""
    registry = get_registry()
    registry.auto_discover()
    return registry


def get_strategies_dir() -> Path:
    """Get the strategies directory path."""
    path = Path(os.environ.get("FWBG_STRATEGIES_DIR", "strategies"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_test_results_dir() -> Path:
    """Get the test results directory path."""
    path = Path(os.environ.get("FWBG_TEST_RESULTS_DIR", "test_results"))
    path.mkdir(parents=True, exist_ok=True)
    return path
