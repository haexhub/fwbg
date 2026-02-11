"""Plugin registry with directory-based discovery."""
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from fwbg.pipeline.base import BasePlugin, PluginPhase

logger = logging.getLogger(__name__)


class PluginNotFoundError(Exception):
    """Raised when a plugin is not found in the registry."""

    pass


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""

    pass


class PluginRegistry:
    """
    Registry for managing and discovering pipeline plugins.

    Handles plugin registration, discovery from directories, and
    provides methods for querying plugin metadata.
    """

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._manifests: Dict[str, dict] = {}

    def register(self, plugin_cls: Type[BasePlugin]) -> None:
        """
        Register a plugin class.

        Args:
            plugin_cls: Plugin class to register (must be BasePlugin subclass)

        Raises:
            PluginValidationError: If plugin_cls is not a valid BasePlugin subclass
        """
        if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, BasePlugin):
            raise PluginValidationError(
                f"{plugin_cls} is not a valid BasePlugin subclass"
            )

        # Check it's not the abstract base class itself
        if plugin_cls is BasePlugin:
            raise PluginValidationError("Cannot register abstract BasePlugin class")

        self._plugins[plugin_cls.name] = plugin_cls

    def get(self, name: str) -> Type[BasePlugin]:
        """
        Get a plugin class by name.

        Args:
            name: Plugin name to look up

        Returns:
            The plugin class

        Raises:
            PluginNotFoundError: If plugin not found
        """
        if name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{name}' not found in registry")
        return self._plugins[name]

    def list_plugins(self, phase: Optional[PluginPhase] = None) -> List[str]:
        """
        List all registered plugin names.

        Args:
            phase: Optional phase to filter by

        Returns:
            List of plugin names
        """
        if phase is None:
            return list(self._plugins.keys())

        return [
            name
            for name, plugin_cls in self._plugins.items()
            if plugin_cls.phase == phase
        ]

    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Args:
            name: Plugin name

        Returns:
            Dict with name, version, phase, stateful, cacheable, default_params

        Raises:
            PluginNotFoundError: If plugin not found
        """
        plugin_cls = self.get(name)  # Raises PluginNotFoundError if not found

        return {
            "name": plugin_cls.name,
            "version": plugin_cls.version,
            "phase": plugin_cls.phase,
            "stateful": plugin_cls.stateful,
            "cacheable": plugin_cls.cacheable,
            "default_params": plugin_cls.get_default_params(),
        }

    def validate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all registered plugins.

        Returns:
            Dict mapping plugin name to {valid: bool, error: str}
        """
        results: Dict[str, Dict[str, Any]] = {}

        for name, plugin_cls in self._plugins.items():
            try:
                instance = plugin_cls()
                is_valid = instance.validate()
                results[name] = {
                    "valid": is_valid,
                    "error": "" if is_valid else "Validation returned False",
                }
            except Exception as e:
                results[name] = {
                    "valid": False,
                    "error": str(e),
                }

        return results

    def discover_from_directory(self, directory: Path) -> None:
        """
        Discover plugins from directories with manifest.json.

        Scans the directory for subdirectories containing manifest.json
        and __init__.py, loads them as modules, and registers any
        BasePlugin subclasses found.

        Args:
            directory: Directory to scan for plugin packages
        """
        directory = Path(directory)

        if not directory.is_dir():
            return

        # Scan for subdirectories
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue

            manifest_file = subdir / "manifest.json"
            init_file = subdir / "__init__.py"

            # Skip if missing required files
            if not manifest_file.exists() or not init_file.exists():
                continue

            # Load manifest
            try:
                with open(manifest_file) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load manifest from {manifest_file}: {e}")
                continue

            package_name = subdir.name
            self._manifests[package_name] = manifest

            # Dynamically import the module
            try:
                module_name = f"_plugin_{package_name}"
                spec = importlib.util.spec_from_file_location(
                    module_name, init_file
                )
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Find and register all BasePlugin subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BasePlugin)
                        and attr is not BasePlugin
                        and not inspect.isabstract(attr)
                    ):
                        self.register(attr)

            except Exception as e:
                logger.warning(f"Failed to load plugin package from {subdir}: {e}")
                continue

    def get_manifest(self, package_name: str) -> dict:
        """
        Get manifest for a plugin package.

        Args:
            package_name: Name of the plugin package directory

        Returns:
            The manifest dictionary
        """
        return self._manifests.get(package_name, {})


# Global registry singleton
_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry singleton.

    Returns:
        The global PluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None
