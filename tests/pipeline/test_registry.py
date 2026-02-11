"""Tests for PluginRegistry with directory-based discovery."""
import json
from pathlib import Path

import pytest

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.registry import (
    PluginNotFoundError,
    PluginRegistry,
    PluginValidationError,
    get_registry,
    reset_registry,
)


# Test fixtures - concrete plugin implementations for testing
class ValidTestPlugin(BasePlugin):
    """A valid test plugin for registry tests."""

    name = "valid_test_plugin"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        return ctx

    def validate(self) -> bool:
        return True

    @classmethod
    def get_default_params(cls):
        return {"param1": "value1", "param2": 42}


class AnotherValidPlugin(BasePlugin):
    """Another valid plugin with different phase."""

    name = "another_valid_plugin"
    version = "2.0.0"
    phase = PluginPhase.INDICATORS
    stateful = True
    cacheable = False

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        return ctx

    def validate(self) -> bool:
        return True


class InvalidValidationPlugin(BasePlugin):
    """Plugin that fails validation."""

    name = "invalid_validation_plugin"
    version = "1.0.0"
    phase = PluginPhase.DATA_LOADING

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        return ctx

    def validate(self) -> bool:
        return False


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return PluginRegistry()


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset global registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    def test_registry_register_plugin(self, registry):
        """Test manual plugin registration works."""
        registry.register(ValidTestPlugin)

        # Should be registered
        assert "valid_test_plugin" in registry.list_plugins()

        # Should be able to retrieve it
        plugin_cls = registry.get("valid_test_plugin")
        assert plugin_cls is ValidTestPlugin

    def test_registry_get_plugin(self, registry):
        """Test get plugin by name returns correct class."""
        registry.register(ValidTestPlugin)
        registry.register(AnotherValidPlugin)

        # Get first plugin
        plugin1 = registry.get("valid_test_plugin")
        assert plugin1 is ValidTestPlugin
        assert plugin1.name == "valid_test_plugin"
        assert plugin1.version == "1.0.0"

        # Get second plugin
        plugin2 = registry.get("another_valid_plugin")
        assert plugin2 is AnotherValidPlugin
        assert plugin2.name == "another_valid_plugin"

    def test_registry_get_unknown_plugin(self, registry):
        """Test get unknown plugin raises PluginNotFoundError."""
        registry.register(ValidTestPlugin)

        with pytest.raises(PluginNotFoundError) as exc_info:
            registry.get("nonexistent_plugin")

        assert "nonexistent_plugin" in str(exc_info.value)

    def test_registry_list_by_phase(self, registry):
        """Test list plugins filtered by phase."""
        registry.register(ValidTestPlugin)  # PREPROCESSING
        registry.register(AnotherValidPlugin)  # INDICATORS
        registry.register(InvalidValidationPlugin)  # DATA_LOADING

        # List all plugins
        all_plugins = registry.list_plugins()
        assert len(all_plugins) == 3

        # Filter by phase
        preprocessing_plugins = registry.list_plugins(phase=PluginPhase.PREPROCESSING)
        assert preprocessing_plugins == ["valid_test_plugin"]

        indicators_plugins = registry.list_plugins(phase=PluginPhase.INDICATORS)
        assert indicators_plugins == ["another_valid_plugin"]

        data_loading_plugins = registry.list_plugins(phase=PluginPhase.DATA_LOADING)
        assert data_loading_plugins == ["invalid_validation_plugin"]

        # Empty phase
        model_plugins = registry.list_plugins(phase=PluginPhase.MODEL)
        assert model_plugins == []

    def test_registry_discover_from_directory(self, registry, tmp_path):
        """Test discover plugins from directory with manifest.json."""
        # Create plugin package directory structure
        plugin_dir = tmp_path / "my_test_plugin"
        plugin_dir.mkdir()

        # Create manifest.json
        manifest = {
            "name": "my_test_plugin",
            "version": "1.0.0",
            "description": "A test plugin for discovery",
            "author": "Test Author",
        }
        manifest_file = plugin_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        # Create __init__.py with a valid plugin class
        init_file = plugin_dir / "__init__.py"
        init_file.write_text('''
"""Test plugin module."""
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext


class DiscoveredPlugin(BasePlugin):
    """A plugin that will be discovered."""

    name = "discovered_plugin"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        return ctx

    def validate(self) -> bool:
        return True
''')

        # Discover plugins from directory
        registry.discover_from_directory(tmp_path)

        # Should have discovered the plugin
        assert "discovered_plugin" in registry.list_plugins()

        # Should be able to get the plugin
        plugin_cls = registry.get("discovered_plugin")
        assert plugin_cls.name == "discovered_plugin"
        assert plugin_cls.version == "1.0.0"

        # Should have stored the manifest
        stored_manifest = registry.get_manifest("my_test_plugin")
        assert stored_manifest["name"] == "my_test_plugin"
        assert stored_manifest["description"] == "A test plugin for discovery"

    def test_registry_validate_all(self, registry):
        """Test validate all plugins handles valid and invalid."""
        registry.register(ValidTestPlugin)
        registry.register(AnotherValidPlugin)
        registry.register(InvalidValidationPlugin)

        results = registry.validate_all()

        # Should have results for all three plugins
        assert len(results) == 3

        # Valid plugins should pass
        assert results["valid_test_plugin"]["valid"] is True
        assert results["valid_test_plugin"]["error"] == ""

        assert results["another_valid_plugin"]["valid"] is True
        assert results["another_valid_plugin"]["error"] == ""

        # Invalid plugin should fail
        assert results["invalid_validation_plugin"]["valid"] is False
        assert results["invalid_validation_plugin"]["error"] != ""

    def test_registry_get_plugin_info(self, registry):
        """Test get plugin metadata."""
        registry.register(ValidTestPlugin)
        registry.register(AnotherValidPlugin)

        # Get info for first plugin
        info1 = registry.get_info("valid_test_plugin")
        assert info1["name"] == "valid_test_plugin"
        assert info1["version"] == "1.0.0"
        assert info1["phase"] == PluginPhase.PREPROCESSING
        assert info1["stateful"] is False
        assert info1["cacheable"] is True
        assert info1["default_params"] == {"param1": "value1", "param2": 42}

        # Get info for second plugin with custom attributes
        info2 = registry.get_info("another_valid_plugin")
        assert info2["name"] == "another_valid_plugin"
        assert info2["version"] == "2.0.0"
        assert info2["phase"] == PluginPhase.INDICATORS
        assert info2["stateful"] is True
        assert info2["cacheable"] is False
        assert info2["default_params"] == {}


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_returns_singleton(self):
        """Test get_registry returns same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_reset_registry_creates_new_instance(self):
        """Test reset_registry creates new instance."""
        reg1 = get_registry()
        reg1.register(ValidTestPlugin)
        assert "valid_test_plugin" in reg1.list_plugins()

        reset_registry()

        reg2 = get_registry()
        assert reg1 is not reg2
        assert "valid_test_plugin" not in reg2.list_plugins()


class TestRegistryValidation:
    """Tests for registry validation."""

    def test_register_non_baseplugin_raises_error(self, registry):
        """Test registering non-BasePlugin class raises PluginValidationError."""

        class NotAPlugin:
            name = "not_a_plugin"

        with pytest.raises(PluginValidationError):
            registry.register(NotAPlugin)

    def test_get_info_unknown_plugin_raises_error(self, registry):
        """Test get_info for unknown plugin raises PluginNotFoundError."""
        with pytest.raises(PluginNotFoundError):
            registry.get_info("unknown_plugin")
