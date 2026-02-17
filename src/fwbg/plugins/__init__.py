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

from fwbg_sdk import (
    BaseIndicator,
    BaseExitStrategy,
    BaseFeatureSelector,
    BasePreprocessor,
    BaseRiskManager,
    BaseDataLoader,
)


def get_plugins_dir() -> Path:
    """Get the plugins directory path."""
    return Path(__file__).parent


def _find_plugin_path(package: str, plugin_type: str, plugin_name: str) -> Optional[Path]:
    """Find a plugin's __init__.py across built-in, entry_point, and user dirs."""
    # 1. Built-in plugins
    builtin = get_plugins_dir() / package / plugin_type / plugin_name / "__init__.py"
    if builtin.exists():
        return builtin

    # 2. Entry-point-provided plugins
    try:
        from importlib.metadata import entry_points as _eps
        seen = set()
        for ep in _eps(group="fwbg.plugin_packages"):
            try:
                plugins_dir = ep.load()()
                resolved = plugins_dir.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidate = plugins_dir / package / plugin_type / plugin_name / "__init__.py"
                if candidate.exists():
                    return candidate
            except Exception:
                continue
    except Exception:
        pass

    # 3. User plugins
    user_dir = Path.home() / ".fwbg" / "plugins"
    user = user_dir / package / plugin_type / plugin_name / "__init__.py"
    if user.exists():
        return user

    return None


def import_plugin_module(
    package: str,
    plugin_type: str,
    plugin_name: str
) -> Optional[ModuleType]:
    """
    Import a plugin module dynamically.

    Searches built-in plugins, installed packages (via entry_points),
    and user plugins (~/.fwbg/plugins/).

    Args:
        package: Plugin package name (e.g., 'fwbg-core', 'fwbg-premium')
        plugin_type: Plugin type directory (e.g., 'indicators', 'preprocessing')
        plugin_name: Plugin name (e.g., 'trend', 'fractional_diff')

    Returns:
        Loaded module or None if not found
    """
    module_path = _find_plugin_path(package, plugin_type, plugin_name)
    if module_path is None:
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
    "BaseDataLoader",
    "get_plugins_dir",
    "import_plugin_module",
    "run_plugin_tests",
]
