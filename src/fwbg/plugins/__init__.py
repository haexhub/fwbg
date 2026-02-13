"""
Plugin Base Classes.

Diese Module definieren die abstrakten Basisklassen für alle Plugin-Typen.
Entwickler importieren diese Klassen um eigene Plugins zu erstellen.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

from .indicator import BaseIndicator
from .exit_strategy import BaseExitStrategy
from .feature_selector import BaseFeatureSelector
from .preprocessor import BasePreprocessor
from .risk_manager import BaseRiskManager


def get_plugins_dir() -> Path:
    """Get the plugins directory path."""
    return Path(__file__).parent


def import_plugin_module(
    package: str,
    plugin_type: str,
    plugin_name: str
) -> Optional[ModuleType]:
    """
    Import a plugin module dynamically.

    Args:
        package: Plugin package name (e.g., 'fwbg-core', 'fwbg-premium')
        plugin_type: Plugin type directory (e.g., 'indicators', 'preprocessing')
        plugin_name: Plugin name (e.g., 'trend', 'fractional_diff')

    Returns:
        Loaded module or None if not found
    """
    plugins_dir = get_plugins_dir()
    module_path = plugins_dir / package / plugin_type / plugin_name / "__init__.py"

    if not module_path.exists():
        return None

    module_name = f"fwbg.plugins.{package.replace('-', '_')}.{plugin_type}.{plugin_name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_plugin_tests(verbose: bool = False) -> dict:
    """
    Run tests for all plugins.

    Args:
        verbose: If True, print verbose output

    Returns:
        Dictionary with test results per plugin
    """
    results = {}
    plugins_dir = get_plugins_dir()

    # Find all test files
    for test_file in plugins_dir.glob("**/tests.py"):
        # Get plugin info from path
        rel_path = test_file.relative_to(plugins_dir)
        parts = rel_path.parts[:-1]  # Remove 'tests.py'

        if len(parts) >= 3:
            package, plugin_type, plugin_name = parts[0], parts[1], parts[2]
            plugin_key = f"{package}:{plugin_type}/{plugin_name}"

            if verbose:
                print(f"\nRunning tests for {plugin_key}...")

            try:
                import pytest
                result = pytest.main([str(test_file), "-v" if verbose else "-q"])
                results[plugin_key] = {
                    "passed": result == 0,
                    "exit_code": result,
                }
            except ImportError:
                results[plugin_key] = {
                    "passed": False,
                    "error": "pytest not available",
                }

    return results


__all__ = [
    "BaseIndicator",
    "BaseExitStrategy",
    "BaseFeatureSelector",
    "BasePreprocessor",
    "BaseRiskManager",
    "get_plugins_dir",
    "import_plugin_module",
    "run_plugin_tests",
]
