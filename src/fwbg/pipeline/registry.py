"""Plugin registry with directory-based discovery and namespace support."""
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

    Supports namespaced plugins in format "package:plugin" (e.g., "fwbg-core:trend").
    All plugin names must be fully qualified with namespace.
    """

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        # Store plugins with fully qualified names: "package:plugin"
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        # Package-level manifests
        self._package_manifests: Dict[str, dict] = {}
        # Plugin-level manifests
        self._plugin_manifests: Dict[str, dict] = {}

    def register(self, plugin_cls: Type[BasePlugin], namespace: str) -> str:
        """
        Register a plugin class with namespace.

        Args:
            plugin_cls: Plugin class to register (must be BasePlugin subclass)
            namespace: Package namespace (e.g., "fwbg-core")

        Returns:
            Fully qualified plugin name (e.g., "fwbg-core:trend")

        Raises:
            PluginValidationError: If plugin_cls is not a valid BasePlugin subclass
        """
        if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, BasePlugin):
            raise PluginValidationError(
                f"{plugin_cls} is not a valid BasePlugin subclass"
            )

        if plugin_cls is BasePlugin:
            raise PluginValidationError("Cannot register abstract BasePlugin class")

        fqn = f"{namespace}:{plugin_cls.name}"
        self._plugins[fqn] = plugin_cls
        return fqn

    def get(self, name: str) -> Type[BasePlugin]:
        """
        Get a plugin class by fully qualified name.

        Args:
            name: Fully qualified plugin name (e.g., "fwbg-core:trend")

        Returns:
            The plugin class

        Raises:
            PluginNotFoundError: If plugin not found
            ValueError: If name is not fully qualified (missing namespace)
        """
        if ":" not in name:
            raise ValueError(
                f"Plugin name '{name}' must be fully qualified with namespace "
                f"(e.g., 'fwbg-core:{name}'). Available: {self.list_plugins()}"
            )

        if name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{name}' not found in registry")
        return self._plugins[name]

    def resolve_name(self, name: str) -> str:
        """Resolve a short plugin name to fully qualified form.

        Returns as-is if already qualified (contains ':').
        Raises ValueError if ambiguous (multiple matches).
        Returns original name unchanged if no match found.
        """
        if ":" in name:
            return name
        matches = [fqn for fqn in self._plugins if fqn.split(":", 1)[1] == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous plugin name '{name}' matches: {matches}. "
                f"Use fully qualified name."
            )
        return name

    def list_plugins(
        self,
        phase: Optional[PluginPhase] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        List all registered plugin names.

        Args:
            phase: Optional phase to filter by
            namespace: Optional namespace to filter by

        Returns:
            List of fully qualified plugin names
        """
        plugins = list(self._plugins.keys())

        if namespace is not None:
            plugins = [p for p in plugins if p.startswith(f"{namespace}:")]

        if phase is not None:
            plugins = [
                name
                for name in plugins
                if self._plugins[name].phase == phase
            ]

        return plugins

    def list_namespaces(self) -> List[str]:
        """
        List all registered namespaces.

        Returns:
            List of namespace names
        """
        return list(self._package_manifests.keys())

    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Args:
            name: Fully qualified plugin name

        Returns:
            Dict with name, namespace, version, phase, stateful, cacheable, default_params

        Raises:
            PluginNotFoundError: If plugin not found
        """
        plugin_cls = self.get(name)
        namespace, plugin_name = name.split(":", 1)

        return {
            "name": plugin_cls.name,
            "namespace": namespace,
            "fqn": name,
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
            Dict mapping fully qualified name to {valid: bool, error: str}
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

    def discover_package(self, package_dir: Path) -> List[str]:
        """
        Discover plugins from a namespaced package directory.

        Expected structure:
        package_dir/
        ├── manifest.json       # Package manifest with "name" field
        ├── indicators/
        │   ├── trend/
        │   │   ├── __init__.py
        │   │   └── manifest.json
        │   └── ...
        ├── preprocessing/
        └── ...

        Args:
            package_dir: Directory containing the package

        Returns:
            List of fully qualified plugin names discovered
        """
        discovered: List[str] = []
        package_dir = Path(package_dir)

        if not package_dir.is_dir():
            return discovered

        # Load package manifest
        package_manifest_file = package_dir / "manifest.json"
        if not package_manifest_file.exists():
            logger.warning(f"No manifest.json in package directory: {package_dir}")
            return discovered

        try:
            with open(package_manifest_file) as f:
                package_manifest = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load package manifest from {package_manifest_file}: {e}")
            return discovered

        namespace = package_manifest.get("name", package_dir.name)
        self._package_manifests[namespace] = package_manifest

        # Scan for plugin categories
        for category in ["data_loading", "indicators", "preprocessing", "feature_selection", "exit_strategies", "risk_management"]:
            category_dir = package_dir / category
            if category_dir.is_dir():
                discovered.extend(
                    self._discover_plugins_in_category(category_dir, namespace)
                )

        return discovered

    def _discover_plugins_in_category(
        self, category_dir: Path, namespace: str
    ) -> List[str]:
        """
        Discover plugins within a category directory.

        Args:
            category_dir: Directory containing plugin subdirectories
            namespace: Package namespace

        Returns:
            List of fully qualified plugin names
        """
        discovered: List[str] = []

        for plugin_dir in category_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            manifest_file = plugin_dir / "manifest.json"
            init_file = plugin_dir / "__init__.py"

            if not manifest_file.exists() or not init_file.exists():
                continue

            # Load plugin manifest
            try:
                with open(manifest_file) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load manifest from {manifest_file}: {e}")
                continue

            plugin_name = manifest.get("name", plugin_dir.name)
            fqn = f"{namespace}:{plugin_name}"
            self._plugin_manifests[fqn] = manifest

            # Dynamically import the module
            try:
                module_name = f"_plugin_{namespace}_{plugin_name}"
                spec = importlib.util.spec_from_file_location(module_name, init_file)
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
                        registered_fqn = self.register(attr, namespace)
                        discovered.append(registered_fqn)
                        # Propagate manifest attributes to plugin class
                        if "benefits_from_stationary" in manifest:
                            attr.benefits_from_stationary = manifest["benefits_from_stationary"]

            except Exception as e:
                logger.warning(f"Failed to load plugin from {plugin_dir}: {e}")
                continue

        return discovered

    def get_package_manifest(self, namespace: str) -> dict:
        """
        Get manifest for a package namespace.

        Args:
            namespace: Package namespace (e.g., "fwbg-core")

        Returns:
            The package manifest dictionary
        """
        return self._package_manifests.get(namespace, {})

    def get_plugin_manifest(self, fqn: str) -> dict:
        """
        Get manifest for a specific plugin.

        Args:
            fqn: Fully qualified plugin name

        Returns:
            The plugin manifest dictionary
        """
        return self._plugin_manifests.get(fqn, {})

    def auto_discover(self) -> List[str]:
        """
        Automatically discover plugins from core packages, installed packages,
        and user directory.

        Discovers plugins from:
        1. Built-in packages (fwbg-core) in plugins/
        2. Installed packages via entry_points (fwbg-premium etc.)
        3. User packages (~/.fwbg/plugins/)

        Returns:
            List of fully qualified plugin names discovered
        """
        discovered: List[str] = []
        plugins_dir = get_core_plugins_dir()

        # 1. Built-in packages
        if plugins_dir.exists():
            for package_dir in plugins_dir.iterdir():
                if package_dir.is_dir() and (package_dir / "manifest.json").exists():
                    discovered.extend(self.discover_package(package_dir))

        # 2. Installed packages via entry_points
        discovered.extend(self._discover_from_entry_points())

        # 3. User plugins
        user_dir = get_user_plugins_dir()
        if user_dir.exists():
            for package_dir in user_dir.iterdir():
                if package_dir.is_dir() and (package_dir / "manifest.json").exists():
                    discovered.extend(self.discover_package(package_dir))

        return discovered

    def _discover_from_entry_points(self) -> List[str]:
        """
        Discover plugins from pip-installed packages via entry_points.

        Looks for packages registered under the 'fwbg.plugin_packages' group.
        Each entry point should be a callable returning a Path to a plugins directory.

        Returns:
            List of fully qualified plugin names discovered
        """
        from importlib.metadata import entry_points

        discovered: List[str] = []
        seen_dirs: set = set()
        try:
            eps = entry_points(group="fwbg.plugin_packages")
            for ep in eps:
                try:
                    get_dir = ep.load()
                    plugins_dir = get_dir()
                    resolved = plugins_dir.resolve()
                    if resolved in seen_dirs:
                        continue
                    seen_dirs.add(resolved)
                    if plugins_dir.is_dir():
                        for package_dir in plugins_dir.iterdir():
                            if package_dir.is_dir() and (package_dir / "manifest.json").exists():
                                discovered.extend(self.discover_package(package_dir))
                except Exception as e:
                    logger.warning(f"Failed to load plugin package '{ep.name}': {e}")
        except Exception:
            pass
        return discovered

def get_user_plugins_dir() -> Path:
    """
    Get user plugins directory (~/.fwbg/plugins/).

    Returns:
        Path to user plugins directory
    """
    return Path.home() / ".fwbg" / "plugins"


def get_core_plugins_dir() -> Path:
    """
    Get core plugins directory.

    Returns:
        Path to core plugins directory (src/fwbg/plugins/)
    """
    return Path(__file__).parent.parent / "plugins"


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
